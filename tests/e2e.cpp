// End-to-end tests: the real router on an ephemeral port, with in-process
// mock servers standing in for the decider and both model tiers.
// Port of tests/proxy.rs.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.hpp"
#include "json.hpp"
#include "server.hpp"

using namespace s1;
using config::Config;
using config::DeciderConfig;
using config::OnError;
using config::ServerConfig;
using config::TierConfig;
using s1::Json;

// ------------------------------------------------------------------ mocks ---

struct Recorded {
    std::string path;
    httplib::Headers headers;  // copied request headers
    std::string body;
};

/// One in-process HTTP mock: scriptable per-route responders, everything else
/// 404s like an unmatched wiremock mount.
class MockServer {
public:
    using Responder = std::function<void(const httplib::Request&, httplib::Response&)>;

    MockServer() {
        svr_.set_payload_max_length(64 * 1024 * 1024);
        auto guard = [this](const char* path, Responder& responder) {
            svr_.Post(path, [this, path, &responder](const httplib::Request& req,
                                                     httplib::Response& res) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    recorded_.push_back(Recorded{path, req.headers, req.body});
                }
                std::lock_guard<std::mutex> lock(mu_);
                if (responder) {
                    responder(req, res);
                } else {
                    res.status = 404;
                }
            });
        };
        guard("/v1/decisions", on_decisions_);
        guard("/v1/chat/completions", on_chat_);
        guard("/v1/messages", on_messages_);
    }

    void start() {
        port_ = svr_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { svr_.listen_after_bind(); });
    }

    ~MockServer() {
        svr_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string uri() const { return "http://127.0.0.1:" + std::to_string(port_); }

    std::vector<Recorded> requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return recorded_;
    }
    size_t count(const std::string& path) const {
        size_t n = 0;
        for (const auto& r : requests()) {
            if (r.path == path) ++n;
        }
        return n;
    }

    Responder on_decisions_;
    Responder on_chat_;
    Responder on_messages_;

private:
    httplib::Server svr_;
    std::thread thread_;
    int port_ = 0;
    mutable std::mutex mu_;
    std::vector<Recorded> recorded_;
};

void respond_json(httplib::Response& res, int status, const Json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void respond_raw(httplib::Response& res, int status, const std::string& body,
                 const char* content_type) {
    res.status = status;
    res.set_content(body, content_type);
}

/// wiremock's body_partial_json: every expected key/value (recursively, arrays
/// element-wise) is present in the actual body.
bool json_subset(const Json& expected, const Json& actual) {
    if (expected.is_object()) {
        if (!actual.is_object()) return false;
        for (auto it = expected.begin(); it != expected.end(); ++it) {
            if (!actual.contains(it.key())) return false;
            if (!json_subset(it.value(), actual.at(it.key()))) return false;
        }
        return true;
    }
    if (expected.is_array()) {
        if (!actual.is_array() || actual.size() != expected.size()) return false;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!json_subset(expected[i], actual[i])) return false;
        }
        return true;
    }
    return expected == actual;
}

Json parse_body(const std::string& body) {
    Json value = Json::parse(body, nullptr, false);
    if (value.is_discarded()) return Json::object();
    return value;
}

// ---------------------------------------------------------------- harness ---

struct Harness {
    std::unique_ptr<MockServer> decider;
    std::unique_ptr<MockServer> top;
    std::unique_ptr<MockServer> flash;
    std::unique_ptr<server::Router> router;
    std::unique_ptr<httplib::Server> proxy;
    std::thread proxy_thread;
    int proxy_port = 0;

    explicit Harness(std::function<void(Config&)> tweak = nullptr) {
        decider = std::make_unique<MockServer>();
        top = std::make_unique<MockServer>();
        flash = std::make_unique<MockServer>();
        decider->start();
        top->start();
        flash->start();

        auto tier = [](const MockServer& server, const char* model,
                       const char* key) {
            return TierConfig{server.uri() + "/v1",
                              key ? std::optional<std::string>(key) : std::nullopt,
                              model, std::nullopt, std::nullopt};
        };
        Config cfg;
        cfg.server = ServerConfig{"127.0.0.1", 0};
        cfg.top = tier(*top, "top-model", "top-secret");
        cfg.flash = tier(*flash, "flash-model", nullptr);
        cfg.decider = DeciderConfig{decider->uri() + "/v1/decisions", std::nullopt,
                                    "clef-flash", 300, 0.5, OnError::Top, 900};
        if (tweak) tweak(cfg);

        router = std::make_unique<server::Router>(std::move(cfg));
        proxy = std::make_unique<httplib::Server>();
        router->install(*proxy);
        proxy_port = proxy->bind_to_any_port("127.0.0.1");
        proxy_thread = std::thread([this] { proxy->listen_after_bind(); });
    }

    ~Harness() {
        proxy->stop();
        if (proxy_thread.joinable()) proxy_thread.join();
    }

    /// The decider answers every request with this p(top).
    void decide(double p_top) {
        const char* choice = p_top >= 0.5 ? "top" : "flash";
        decider->on_decisions_ = [p_top, choice](const httplib::Request&,
                                                 httplib::Response& res) {
            respond_json(res, 200, Json{
                {"model", "clef-flash"},
                {"answers", Json{{"tier", Json{{"type", "choice"},
                                                {"choice", choice},
                                                {"confidence", std::max(p_top, 1.0 - p_top)},
                                                {"probabilities", Json{{"flash", 1.0 - p_top},
                                                                       {"top", p_top}}}}}}},
                {"usage", Json{{"input_tokens", 200}, {"output_tokens", 0}}}});
        };
    }

    struct Reply {
        int status = 0;
        std::string body;
        httplib::Headers headers;
        bool has_header(const std::string& name) const {
            return headers.find(name) != headers.end();
        }
        std::string header_of(const std::string& name) const {
            return headers.count(name) ? headers.find(name)->second : "";
        }
        Json json() const { return parse_body(body); }
    };

    Reply post(const std::string& route, const Json& body) {
        httplib::Client cli("127.0.0.1", proxy_port);
        cli.set_connection_timeout(std::chrono::seconds(30));
        cli.set_read_timeout(std::chrono::seconds(60));
        httplib::Headers headers{{"authorization", "Bearer client-key"},
                                 {"x-api-key", "client-key"}};
        auto res = cli.Post(route.c_str(), headers, body.dump(), "application/json");
        REQUIRE(res);
        return Reply{res->status, res->body, res->headers};
    }

    Reply get(const std::string& route) {
        httplib::Client cli("127.0.0.1", proxy_port);
        cli.set_connection_timeout(std::chrono::seconds(30));
        cli.set_read_timeout(std::chrono::seconds(60));
        auto res = cli.Get(route.c_str());
        REQUIRE(res);
        return Reply{res->status, res->body, res->headers};
    }
};

Json completion_body(const char* text) {
    return Json{{"id", "chatcmpl-1"},
                {"object", "chat.completion"},
                {"choices", Json::array({Json{
                    {"index", 0},
                    {"finish_reason", "stop"},
                    {"message", Json{{"role", "assistant"}, {"content", text}}}}})},
                {"usage", Json{{"prompt_tokens", 9},
                               {"completion_tokens", 3},
                               {"total_tokens", 12}}}};
}

/// A chat-completions mock that answers `completion` when the body matches
/// `expected` (subset), else 404 — wiremock's match + mount.
void chat_mock(MockServer& server, const Json& expected, int status, const Json& completion) {
    server.on_chat_ = [expected, status, completion](const httplib::Request& req,
                                                     httplib::Response& res) {
        if (json_subset(expected, parse_body(req.body))) {
            respond_json(res, status, completion);
        } else {
            res.status = 404;
        }
    };
}

Json chat_request(const char* prompt) {
    return Json{{"model", "anything"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", prompt}}})}};
}

// ----------------------------------------------------------------- tests ---

TEST_CASE("complex prompt goes to top with its key and model") {
    Harness h;
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request& req, httplib::Response& res) {
        auto auth = req.get_header_value("authorization");
        if (auth == "Bearer top-secret" &&
            json_subset(Json::parse(R"({"model": "top-model", "temperature": 0.2})"),
                        parse_body(req.body))) {
            respond_json(res, 200, completion_body("from top"));
        } else {
            res.status = 404;
        }
    };
    h.flash->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 200, completion_body("from flash"));
    };

    Json request = chat_request("design a distributed lock service");
    request["temperature"] = 0.2;
    auto response = h.post("/v1/chat/completions", request);

    CHECK(response.status == 200);
    CHECK(response.header_of("x-s1-tier") == "top");
    CHECK(response.header_of("x-s1-decision") == "p=0.9000;src=decider");
    CHECK(response.json()["choices"][0]["message"]["content"] == "from top");

    // The decider saw the prompt, and was asked the tier question.
    auto requests = h.decider->requests();
    REQUIRE(requests.size() == 1);
    Json asked = parse_body(requests[0].body);
    CHECK(asked["state"]["latest_user_request"] == "design a distributed lock service");
    CHECK(asked["questions"]["tier"]["type"] == "choice");
    CHECK(asked["questions"]["tier"]["criteria"]["top"].is_string());
    CHECK(h.top->count("/v1/chat/completions") == 1);
    CHECK(h.flash->count("/v1/chat/completions") == 0);
}

TEST_CASE("simple prompt goes to flash without credentials") {
    Harness h;
    h.decide(0.1);
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 200, completion_body("from top"));
    };
    chat_mock(*h.flash, Json::parse(R"({"model": "flash-model"})"), 200,
              completion_body("from flash"));

    auto response = h.post("/v1/chat/completions", chat_request("say hi"));

    CHECK(response.header_of("x-s1-tier") == "flash");
    auto seen = h.flash->requests();
    REQUIRE(seen.size() == 1);
    CHECK(!seen[0].headers.contains("authorization"));
    CHECK(!seen[0].headers.contains("x-api-key"));
    CHECK(h.top->count("/v1/chat/completions") == 0);
}

TEST_CASE("threshold is configurable") {
    Harness h([](Config& cfg) { cfg.decider.threshold = 0.8; });
    h.decide(0.7);
    chat_mock(*h.flash, Json::object(), 200, completion_body("from flash"));

    auto response = h.post("/v1/chat/completions", chat_request("medium task"));
    CHECK(response.header_of("x-s1-tier") == "flash");
}

TEST_CASE("decider failures fall back to top") {
    struct Failure {
        int status;
        std::string body;
        bool raw;
        int delay_ms;
    };
    auto failures = std::vector<Failure>{
        {400, R"({"error": "schema requires too many tokens"})", true, 0},
        {200, R"({"answers": {}})", true, 0},
        {200, R"({"answers": {"tier": {"choice": "top"}}})", true, 1500},
    };
    for (const auto& failure : failures) {
        CAPTURE(failure.status);
        Harness h;
        h.decider->on_decisions_ = [failure](const httplib::Request&,
                                             httplib::Response& res) {
            if (failure.delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(failure.delay_ms));
            }
            respond_raw(res, failure.status, failure.body, "application/json");
        };
        chat_mock(*h.top, Json::object(), 200, completion_body("from top"));

        auto response = h.post("/v1/chat/completions", chat_request("say hi"));
        CHECK(response.status == 200);
        CHECK(response.header_of("x-s1-tier") == "top");
        CHECK(response.header_of("x-s1-decision") == "src=fallback");
        CHECK(h.top->count("/v1/chat/completions") == 1);
    }
}

TEST_CASE("unreachable decider can fail the request instead") {
    Harness h([](Config& cfg) {
        cfg.decider.url = "http://127.0.0.1:1/v1/decisions";
        cfg.decider.on_error = OnError::Fail;
    });
    chat_mock(*h.top, Json::object(), 200, completion_body("x"));

    auto openai = h.post("/v1/chat/completions", chat_request("hi"));
    CHECK(openai.status == 503);
    CHECK(openai.json()["error"]["message"].is_string());

    auto anthropic = h.post(
        "/v1/messages",
        Json{{"max_tokens", 10},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}});
    CHECK(anthropic.status == 503);
    Json body = anthropic.json();
    CHECK(body["type"] == "error");
    CHECK(body["error"]["type"] == "overloaded_error");
    CHECK(h.top->count("/v1/chat/completions") == 0);
}

TEST_CASE("ha fails over to the other tier when the model errors") {
    Harness h([](Config& cfg) { cfg.decider.on_error = OnError::Ha; });
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_raw(res, 503, "overloaded", "text/plain");
    };
    chat_mock(*h.flash, Json::parse(R"({"model": "flash-model"})"), 200,
              completion_body("from flash"));

    auto response = h.post("/v1/chat/completions", chat_request("refactor everything"));
    CHECK(response.status == 200);
    CHECK(response.header_of("x-s1-tier") == "flash");
    CHECK(response.header_of("x-s1-decision") == "p=0.9000;src=decider;failover_from=top");
    CHECK(response.json()["choices"][0]["message"]["content"] == "from flash");
    CHECK(h.top->count("/v1/chat/completions") == 1);
    CHECK(h.flash->count("/v1/chat/completions") == 1);
}

TEST_CASE("ha survives decider and top tier both being down") {
    Harness h([](Config& cfg) {
        cfg.decider.on_error = OnError::Ha;
        cfg.decider.url = "http://127.0.0.1:1/v1/decisions";
        cfg.top.base_url = "http://127.0.0.1:1/v1";
    });
    chat_mock(*h.flash, Json::parse(R"({"model": "flash-model"})"), 200,
              completion_body("still here"));

    auto openai = h.post("/v1/chat/completions", chat_request("hi"));
    CHECK(openai.status == 200);
    CHECK(openai.header_of("x-s1-decision") == "src=fallback;failover_from=top");

    // The Anthropic body is rebuilt for the tier that actually answers.
    auto anthropic = h.post(
        "/v1/messages",
        Json{{"max_tokens", 10},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}});
    CHECK(anthropic.status == 200);
    CHECK(anthropic.header_of("x-s1-tier") == "flash");
    Json body = anthropic.json();
    CHECK(body["model"] == "flash-model");
    CHECK(body["content"][0]["text"] == "still here");
    CHECK(h.flash->count("/v1/chat/completions") == 2);
}

TEST_CASE("ha does not retry requests the model rejected") {
    Harness h([](Config& cfg) { cfg.decider.on_error = OnError::Ha; });
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 400, Json{{"error", Json{{"message", "bad tool schema"}}}});
    };
    h.flash->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 200, completion_body("x"));
    };

    auto response = h.post("/v1/chat/completions", chat_request("hi"));
    CHECK(response.status == 400);
    CHECK(response.header_of("x-s1-decision") == "p=0.9000;src=decider");
    CHECK(h.top->count("/v1/chat/completions") == 1);
    CHECK(h.flash->count("/v1/chat/completions") == 0);
}

TEST_CASE("ha reports the last error when both tiers fail") {
    Harness h([](Config& cfg) { cfg.decider.on_error = OnError::Ha; });
    h.decide(0.1);
    h.flash->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 500, Json::object());
    };
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 429, Json{{"error", Json{{"message", "slow down"}}}});
    };

    auto response = h.post("/v1/chat/completions", chat_request("hi"));
    CHECK(response.status == 429);
    CHECK(response.header_of("x-s1-tier") == "top");
    CHECK(response.header_of("x-s1-decision") == "p=0.1000;src=decider;failover_from=flash");
    CHECK(h.flash->count("/v1/chat/completions") == 1);
    CHECK(h.top->count("/v1/chat/completions") == 1);
}

TEST_CASE("without ha a failing model is not replaced") {
    Harness h;
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 503, Json::object());
    };
    h.flash->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 200, completion_body("x"));
    };

    auto response = h.post("/v1/chat/completions", chat_request("hi"));
    CHECK(response.status == 503);
    CHECK(response.header_of("x-s1-tier") == "top");
    CHECK(h.top->count("/v1/chat/completions") == 1);
    CHECK(h.flash->count("/v1/chat/completions") == 0);
}

TEST_CASE("tool loop reuses the decision and new turns ask again") {
    Harness h;
    h.decide(0.9);
    chat_mock(*h.top, Json::object(), 200, completion_body("ok"));

    Json first = chat_request("refactor the parser");
    Json follow_up = first;
    follow_up["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", Json::array({Json{{"id", "c1"},
                                         {"type", "function"},
                                         {"function", Json{{"name", "read"},
                                                           {"arguments", "{}"}}}}})}});
    follow_up["messages"].push_back(
        Json{{"role", "tool"}, {"tool_call_id", "c1"}, {"content", "fn parse() {}"}});
    Json next_turn = follow_up;
    next_turn["messages"].push_back(
        Json{{"role", "assistant"}, {"content", "done"}});
    next_turn["messages"].push_back(
        Json{{"role", "user"}, {"content", "thanks"}});

    auto a = h.post("/v1/chat/completions", first);
    auto b = h.post("/v1/chat/completions", follow_up);
    CHECK(a.header_of("x-s1-decision") == "p=0.9000;src=decider");
    CHECK(b.header_of("x-s1-decision") == "p=0.9000;src=cache");
    CHECK(h.decider->count("/v1/decisions") == 1);

    auto c = h.post("/v1/chat/completions", next_turn);
    CHECK(c.header_of("x-s1-decision") == "p=0.9000;src=decider");
    CHECK(h.decider->count("/v1/decisions") == 2);
}

TEST_CASE("openai stream and upstream errors pass through untouched") {
    Harness h;
    h.decide(0.1);
    const std::string sse =
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
    h.flash->on_chat_ = [sse](const httplib::Request& req, httplib::Response& res) {
        if (json_subset(Json::parse(R"({"stream": true})"), parse_body(req.body))) {
            respond_raw(res, 200, sse, "text/event-stream");
        } else {
            respond_json(res, 429, Json{{"error", Json{{"message", "slow down"}}}});
        }
    };

    Json streaming = chat_request("say hi");
    streaming["stream"] = true;
    auto streamed = h.post("/v1/chat/completions", streaming);
    CHECK(streamed.header_of("content-type") == "text/event-stream");
    CHECK(streamed.body == sse);

    auto response = h.post("/v1/chat/completions", chat_request("say hi again"));
    CHECK(response.status == 429);
    CHECK(response.header_of("x-s1-tier") == "flash");
    CHECK(response.json()["error"]["message"] == "slow down");
}

TEST_CASE("anthropic request is translated both ways") {
    Harness h;
    h.decide(0.1);
    h.flash->on_chat_ = [](const httplib::Request& req, httplib::Response& res) {
        Json expected = Json::parse(R"({
            "model": "flash-model",
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "what is in main.rs?"}
            ],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        })");
        if (!json_subset(expected, parse_body(req.body))) {
            res.status = 404;
            return;
        }
        respond_json(res, 200, Json{
            {"choices", Json::array({Json{
                {"index", 0},
                {"finish_reason", "tool_calls"},
                {"message", Json{{"role", "assistant"}, {"content", nullptr},
                                 {"tool_calls", Json::array({Json{
                                     {"id", "call_1"}, {"type", "function"},
                                     {"function", Json{{"name", "read"},
                                                       {"arguments", "{\"path\":\"main.rs\"}"}}}}})}}}}})},
            {"usage", Json{{"prompt_tokens", 30}, {"completion_tokens", 8}}}});
    };

    auto response = h.post(
        "/v1/messages",
        Json{{"model", "claude-something"},
             {"max_tokens", 256},
             {"system", "be terse"},
             {"tools", Json::array({Json{{"name", "read"},
                                         {"input_schema", Json{{"type", "object"}}}}})},
             {"messages", Json::array({Json{
                 {"role", "user"},
                 {"content", Json::array({Json{{"type", "text"},
                                                {"text", "what is in main.rs?"}}})}}})}});

    CHECK(response.status == 200);
    CHECK(response.header_of("x-s1-tier") == "flash");
    Json body = response.json();
    CHECK(body["type"] == "message");
    CHECK(body["role"] == "assistant");
    CHECK(body["model"] == "flash-model");
    CHECK(body["stop_reason"] == "tool_use");
    CHECK(body["content"] == Json::parse(
                                R"([{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "main.rs"}}])"));
    CHECK(body["usage"]["input_tokens"] == 30);
    CHECK(body["usage"]["output_tokens"] == 8);
    CHECK(h.flash->count("/v1/chat/completions") == 1);
}

TEST_CASE("anthropic stream is translated to message events") {
    Harness h;
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request& req, httplib::Response& res) {
        Json expected =
            Json::parse(R"({"stream": true, "stream_options": {"include_usage": true}})");
        if (!json_subset(expected, parse_body(req.body))) {
            res.status = 404;
            return;
        }
        std::string upstream;
        for (const char* chunk :
             {R"({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})",
              R"({"choices": [{"index": 0, "delta": {"content": "Hel"}}]})",
              R"({"choices": [{"index": 0, "delta": {"content": "lo"}}]})",
              R"({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})",
              R"({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}})"}) {
            upstream += std::string("data: ") + chunk + "\n\n";
        }
        upstream += "data: [DONE]\n\n";
        respond_raw(res, 200, upstream, "text/event-stream");
    };

    auto response = h.post(
        "/v1/messages",
        Json{{"max_tokens", 64},
             {"stream", true},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "greet me"}}})}});
    CHECK(response.header_of("content-type") == "text/event-stream");
    CHECK(response.header_of("x-s1-tier") == "top");

    std::vector<std::pair<std::string, Json>> events;
    size_t start = 0;
    std::string text = response.body;
    while (start < text.size()) {
        size_t end = text.find("\n\n", start);
        if (end == std::string::npos) break;
        std::string frame = text.substr(start, end - start);
        start = end + 2;
        if (frame.empty()) continue;
        size_t nl = frame.find('\n');
        REQUIRE(nl != std::string::npos);
        std::string event = frame.substr(7, nl - 7);  // after "event: "
        std::string data = frame.substr(nl + 7);      // after "data: "
        events.emplace_back(event, parse_body(data));
    }
    std::vector<std::string> names;
    for (auto& [name, data] : events) names.push_back(name);
    CHECK(names == std::vector<std::string>{"message_start", "content_block_start",
                                            "content_block_delta",
                                            "content_block_delta", "content_block_stop",
                                            "message_delta", "message_stop"});
    std::string said;
    for (auto& [name, data] : events) {
        if (data.contains("delta") && data["delta"].contains("text")) {
            said += data["delta"]["text"].get<std::string>();
        }
    }
    CHECK(said == "Hello");
    CHECK(events[0].second["message"]["model"] == "top-model");
    CHECK(events[5].second["delta"]["stop_reason"] == "end_turn");
    CHECK(events[5].second["usage"]["output_tokens"] == 2);
    CHECK(h.top->count("/v1/chat/completions") == 1);
}

TEST_CASE("anthropic callers get anthropic shaped errors") {
    Harness h;
    h.decide(0.9);
    h.top->on_chat_ = [](const httplib::Request&, httplib::Response& res) {
        respond_json(res, 429, Json{{"error", Json{{"message", "slow down"}}}});
    };

    auto upstream_error = h.post(
        "/v1/messages",
        Json{{"max_tokens", 10},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}});
    CHECK(upstream_error.status == 429);
    CHECK(upstream_error.json() ==
          Json::parse(R"({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})"));

    auto bad_request = h.post("/v1/messages", Json{{"messages", "nope"}});
    CHECK(bad_request.status == 400);
    CHECK(bad_request.json()["error"]["type"] == "invalid_request_error");
}

TEST_CASE("auxiliary endpoints") {
    Harness h;

    auto models = h.get("/v1/models");
    std::vector<std::string> ids;
    Json models_body = models.json();
    for (const auto& entry : models_body["data"]) {
        ids.push_back(entry["id"].get<std::string>());
    }
    CHECK(ids == std::vector<std::string>{"s1-auto", "top-model", "flash-model"});

    auto health = h.get("/health");
    CHECK(health.body.find("\"status\":\"ok\"") != std::string::npos);
    CHECK(health.body.find("top-secret") == std::string::npos);

    auto count = h.post(
        "/v1/messages/count_tokens",
        Json{{"messages", Json::array({Json{{"role", "user"}, {"content", "hello there"}}})}});
    CHECK(count.json()["input_tokens"].get<uint64_t>() > 0);
    CHECK(h.decider->count("/v1/decisions") == 0);

    auto missing = h.post("/v1/responses", Json::object());
    CHECK(missing.status == 404);
}

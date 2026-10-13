#include "server.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "anthropic/mod.hpp"
#include "anthropic/request.hpp"
#include "anthropic/response.hpp"
#include "anthropic/stream.hpp"
#include "decider.hpp"
#include "error.hpp"
#include "http.hpp"
#include "json.hpp"
#include "log.hpp"
#include "openai.hpp"
#include "upstream.hpp"
#include "utf8.hpp"

namespace s1::server {

/// Shared per-instance state: the configuration, the decider and both tiers.
struct State {
    config::Config cfg;
    decider::Decider decider;
    upstream::Upstream top;
    upstream::Upstream flash;

    explicit State(config::Config config)
        : cfg(std::move(config)), decider(cfg.decider), top(cfg.top), flash(cfg.flash) {}

    upstream::Upstream& upstream(upstream::Tier tier) {
        return tier == upstream::Tier::Top ? top : flash;
    }
};

namespace {

constexpr size_t kMaxBodyBytes = 64 * 1024 * 1024;
constexpr auto kPingEvery = std::chrono::seconds(15);
/// Listed by /v1/models for clients that insist on picking a model; the
/// requested model name never affects routing.
const char* const kVirtualModel = "s1-auto";

bool hop_by_hop(const std::string& name) {
    std::string lower = name;
    for (char& c : lower) {
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    return lower == "connection" || lower == "content-length" || lower == "keep-alive" ||
           lower == "transfer-encoding" || lower == "upgrade";
}

bool is_success(long status) { return status >= 200 && status <= 299; }

/// Failures that say the model is unavailable, not that the request is wrong:
/// the other tier may well succeed.
bool model_failed_status(long status) {
    return status >= 500 || status == 401 || status == 403 || status == 404 || status == 408 ||
           status == 429;
}

/// Status codes that say the upstream does not serve Messages at all (as
/// opposed to rejecting this particular request).
bool native_unsupported(long status) { return status == 404 || status == 405 || status == 501; }

std::string fmt_double(double value) {
    char buf[64];
    auto result = std::to_chars(buf, buf + sizeof(buf), value);
    return std::string(buf, result.ptr);
}

// --- response plumbing ---

void render_api_error(httplib::Response& res, Protocol protocol, int status,
                      const std::string& message) {
    res.status = status;
    res.set_content(api_error_body(protocol, status, message).dump(), "application/json");
}

/// Tell the caller which tier answered and why.
void stamp(httplib::Response& res, const decider::Decision& decision) {
    res.set_header("x-s1-tier", upstream::tier_str(decision.tier));
    res.set_header("x-s1-decision", decision.header());
}

void log_routed(Protocol protocol, const decider::Decision& decision, long status,
                std::chrono::steady_clock::time_point started) {
    if (!log::enabled(log::Level::Info)) return;
    auto now = std::chrono::steady_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(now - started);
    std::string line =
        std::string("routed protocol=") + protocol_str(protocol) +
        " tier=" + upstream::tier_str(decision.tier) +
        " p_top=" + (decision.p_top ? fmt_double(*decision.p_top) : "null") +
        " source=" + decider::source_str(decision.source) + " failover_from=" +
        (decision.failed_over_from ? upstream::tier_str(*decision.failed_over_from) : "null") +
        " decider_ms=" + std::to_string(decision.elapsed.count()) +
        " status=" + std::to_string(status) + " total_ms=" + std::to_string(total.count());
    log::write(log::Level::Info, line);
}

struct ParsedBody {
    bool ok = false;
    Json value = Json::object();
    std::string message;
};

ParsedBody parse_object(const std::string& body) {
    ParsedBody out;
    Json value;
    try {
        value = Json::parse(body);
    } catch (const std::exception& err) {
        out.message = std::string("invalid JSON body: ") + err.what();
        return out;
    }
    if (!value.is_object()) {
        out.message = "request body must be a JSON object";
        return out;
    }
    out.ok = true;
    out.value = std::move(value);
    return out;
}

// The decision is always resolved (decider, cache or fallback) before any
// upstream call.
struct DecisionOutcome {
    bool ok = false;
    decider::Decision decision{};
};

DecisionOutcome decide(State& state, Protocol protocol, const decider::Conversation& conv,
                       httplib::Response& res) {
    auto result = state.decider.decide(conv);
    if (!result.ok) {
        render_api_error(res, protocol, 503, "routing decision failed: " + result.error.message);
        return {};
    }
    return {true, result.decision};
}

// Forward an upstream head + streaming body to the client verbatim.
void forward_stream(httplib::Response& res, long status,
                    const std::vector<http::HeaderEntry>& headers,
                    std::shared_ptr<http::StreamRequest> call) {
    res.status = static_cast<int>(status);
    std::string content_type = "application/octet-stream";
    for (const auto& header : headers) {
        if (hop_by_hop(header.name)) continue;
        std::string lower = header.name;
        for (char& c : lower) {
            c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
        }
        if (lower == "content-type") {
            content_type = header.value;
            continue;
        }
        res.set_header(header.name, header.value);
    }
    res.set_chunked_content_provider(
        content_type, [call = std::move(call)](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            std::string chunk;
            bool timed_out = false;
            while (call->next(chunk, std::chrono::hours(24), timed_out)) {
                if (!sink.write(chunk.data(), chunk.size())) return false;
            }
            sink.done();
            return true;
        });
}

// --- chat-completions route ---

struct ChatDispatch {
    enum class Kind { Forwarded, Dead };
    Kind kind = Kind::Dead;
    std::shared_ptr<http::StreamRequest> call;
    long status = 0;
    std::vector<http::HeaderEntry> headers;
    std::string message;  // Dead: transport error text
};

/// Send to the decided tier. With on_error = "ha", a failing model is replaced
/// by the other tier, and `decision` is updated to say so.
ChatDispatch dispatch_chat(State& state, decider::Decision& decision, const Json& request) {
    auto attempt = [&](upstream::Tier tier) -> ChatDispatch {
        const upstream::Upstream& up = state.upstream(tier);
        Json body = request;
        openai::prepare(body, up);
        auto call = std::make_shared<http::StreamRequest>(up.chat_url(), body.dump(),
                                                          up.auth_headers(), up.timeout_ms(),
                                                          upstream::Upstream::kConnectTimeoutMs);
        auto head = call->wait_head();
        if (!head.ok) return ChatDispatch{ChatDispatch::Kind::Dead, {}, 0, {}, head.error};
        return ChatDispatch{
            ChatDispatch::Kind::Forwarded, std::move(call), head.status, head.headers, {}};
    };

    // Failures that say the model is unavailable, not that the request is
    // wrong: the other tier may well succeed.
    auto model_failed = [](const ChatDispatch& result) -> std::optional<std::string> {
        if (result.kind == ChatDispatch::Kind::Dead) return result.message;
        if (model_failed_status(result.status)) {
            return "HTTP " + std::to_string(result.status);
        }
        return std::nullopt;
    };

    ChatDispatch result = attempt(decision.tier);
    if (state.cfg.decider.on_error == config::OnError::Ha) {
        if (auto reason = model_failed(result)) {
            auto failed = decision.tier;
            S1_LOG(log::Level::Warn,
                   "model failed, failing over failed=" + std::string(upstream::tier_str(failed)) +
                       " failover=" + upstream::tier_str(upstream::other_tier(failed)) +
                       " reason=" + *reason);
            if (result.call) result.call->cancel();
            decision.tier = upstream::other_tier(failed);
            decision.failed_over_from = failed;
            result = attempt(decision.tier);
        }
    }
    return result;
}

void handle_chat(const std::shared_ptr<State>& state, const httplib::Request& req,
                 httplib::Response& res) {
    const Protocol protocol = Protocol::OpenAi;
    auto started = std::chrono::steady_clock::now();

    auto parsed = parse_object(req.body);
    if (!parsed.ok) {
        render_api_error(res, protocol, 400, parsed.message);
        return;
    }
    auto decided = decide(*state, protocol, openai::conversation(parsed.value), res);
    if (!decided.ok) return;
    auto& decision = decided.decision;

    auto dispatch = dispatch_chat(*state, decision, parsed.value);
    if (dispatch.kind == ChatDispatch::Kind::Dead) {
        S1_LOG(log::Level::Warn, "upstream unreachable protocol=openai tier=" +
                                     std::string(upstream::tier_str(decision.tier)) +
                                     " error=" + dispatch.message);
        render_api_error(res, protocol, 502,
                         std::string(upstream::tier_str(decision.tier)) +
                             " tier request failed: " + dispatch.message);
        stamp(res, decision);
        return;
    }

    log_routed(protocol, decision, dispatch.status, started);
    // Success or error, the upstream already speaks the caller's protocol.
    forward_stream(res, dispatch.status, dispatch.headers, std::move(dispatch.call));
    stamp(res, decision);
}

// --- messages route ---

/// How a tier answered a Messages request.
struct TierReply {
    enum class Kind { Answered, Dead, Bad };
    enum class Answer { Native, ChatStream, ChatBuffered };
    Kind kind = Kind::Dead;
    Answer answer = Answer::Native;
    long status = 0;
    std::vector<http::HeaderEntry> headers;
    std::shared_ptr<http::StreamRequest> call;  // Native or ChatStream
    std::string buffered;                       // ChatBuffered body
    std::string message;                        // Dead / Bad text

    /// Failures that say the model is unavailable, not that the request is
    /// wrong: the other tier may well succeed.
    std::optional<std::string> failed() const {
        if (kind == Kind::Dead) return message;
        if (kind == Kind::Bad) return std::nullopt;
        if (model_failed_status(status)) return "HTTP " + std::to_string(status);
        return std::nullopt;
    }
};

/// Ask one tier, preferring the Messages API it was addressed on and
/// transparently converting to chat-completions if it refuses the dialect.
TierReply ask_tier(State& state, upstream::Tier tier, const Json& request,
                   const upstream::MessagesHeaders& dialect) {
    const upstream::Upstream& up = state.upstream(tier);

    Json native = up.native_body(request);
    auto native_call = std::make_shared<http::StreamRequest>(
        up.messages_url(), native.dump(), up.messages_headers(dialect), up.timeout_ms(),
        upstream::Upstream::kConnectTimeoutMs);
    auto head = native_call->wait_head();
    if (!head.ok) {
        TierReply reply;
        reply.kind = TierReply::Kind::Dead;
        reply.message = head.error;
        return reply;
    }
    if (!native_unsupported(head.status)) {
        TierReply reply;
        reply.kind = TierReply::Kind::Answered;
        reply.answer = TierReply::Answer::Native;
        reply.status = head.status;
        reply.headers = head.headers;
        reply.call = std::move(native_call);
        return reply;
    }
    native_call->cancel();

    // No Messages API here; fall back to the converted path.
    auto converted = anthropic::request::to_chat(request, up.model, up.max_output_tokens);
    if (!converted.ok) {
        TierReply reply;
        reply.kind = TierReply::Kind::Bad;
        reply.message = converted.error;
        return reply;
    }
    const Json& chat = converted.value;
    if (get_bool(chat, "stream") == true) {
        auto call = std::make_shared<http::StreamRequest>(up.chat_url(), chat.dump(),
                                                          up.auth_headers(), up.timeout_ms(),
                                                          upstream::Upstream::kConnectTimeoutMs);
        auto chat_head = call->wait_head();
        if (!chat_head.ok) {
            TierReply reply;
            reply.kind = TierReply::Kind::Dead;
            reply.message = chat_head.error;
            return reply;
        }
        TierReply reply;
        reply.kind = TierReply::Kind::Answered;
        reply.answer = TierReply::Answer::ChatStream;
        reply.status = chat_head.status;
        reply.headers = chat_head.headers;
        reply.call = std::move(call);
        return reply;
    }
    auto reply_body = up.send(chat);
    if (!reply_body.ok) {
        TierReply reply;
        reply.kind = TierReply::Kind::Dead;
        reply.message = reply_body.error;
        return reply;
    }
    TierReply reply;
    reply.kind = TierReply::Kind::Answered;
    reply.answer = TierReply::Answer::ChatBuffered;
    reply.status = reply_body.status;
    reply.buffered = std::move(reply_body.body);
    return reply;
}

/// Translate an upstream SSE body into an Anthropic SSE body, with pings when
/// the upstream stays silent and a closing batch when it ends.
void forward_translated_stream(httplib::Response& res, std::string model,
                               std::shared_ptr<http::StreamRequest> call) {
    res.status = 200;
    res.set_header("cache-control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [call = std::move(call), model = std::move(model)](size_t /*offset*/,
                                                           httplib::DataSink& sink) -> bool {
            anthropic::stream::StreamTranslator translator(model);
            auto write_all = [&](const std::vector<anthropic::stream::SseEvent>& events) {
                for (const auto& event : events) {
                    std::string frame = event.encode();
                    if (!sink.write(frame.data(), frame.size())) return false;
                }
                return true;
            };
            if (!write_all(translator.start())) return false;

            anthropic::stream::SseParser parser;
            auto ping_deadline = std::chrono::steady_clock::now() + kPingEvery;
            bool ended = false;
            bool failed = false;
            std::string fail_message;
            while (!ended && !failed) {
                auto now = std::chrono::steady_clock::now();
                auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(ping_deadline - now);
                if (remaining <= std::chrono::milliseconds(0)) {
                    anthropic::stream::SseEvent ping{"ping", Json{{"type", "ping"}}};
                    std::string frame = ping.encode();
                    if (!sink.write(frame.data(), frame.size())) return false;
                    ping_deadline = std::chrono::steady_clock::now() + kPingEvery;
                    continue;
                }
                std::string chunk;
                bool timed_out = false;
                if (!call->next(chunk, remaining, timed_out)) {
                    if (timed_out) continue;
                    break;  // upstream stream ended
                }
                parser.feed(chunk);
                for (auto& data : parser.drain()) {
                    if (utf8::trim_copy(data) == "[DONE]") {
                        ended = true;
                        break;
                    }
                    Json parsed = Json::parse(data, nullptr, false);
                    if (parsed.is_discarded()) continue;
                    if (parsed.contains("error")) {
                        Json wrapped = Json{{"error", parsed.at("error")}};
                        fail_message = upstream_message(wrapped.dump());
                        failed = true;
                        break;
                    }
                    if (!write_all(translator.on_chunk(parsed))) return false;
                }
                ping_deadline = std::chrono::steady_clock::now() + kPingEvery;
            }
            if (failed) {
                S1_LOG(log::Level::Warn, "anthropic stream aborted message=" + fail_message);
                std::string frame =
                    anthropic::stream::StreamTranslator::error_event(fail_message).encode();
                if (!sink.write(frame.data(), frame.size())) return false;
                sink.done();
                return true;
            }
            if (!write_all(translator.finish())) return false;
            sink.done();
            return true;
        });
}

void handle_messages(const std::shared_ptr<State>& state, const httplib::Request& req,
                     httplib::Response& res) {
    const Protocol protocol = Protocol::Anthropic;
    auto started = std::chrono::steady_clock::now();

    auto parsed = parse_object(req.body);
    if (!parsed.ok) {
        render_api_error(res, protocol, 400, parsed.message);
        return;
    }
    auto decided = decide(*state, protocol, anthropic::conversation(parsed.value), res);
    if (!decided.ok) return;
    auto& decision = decided.decision;
    auto dialect = upstream::MessagesHeaders::extract(req.headers);

    TierReply reply = ask_tier(*state, decision.tier, parsed.value, dialect);
    if (state->cfg.decider.on_error == config::OnError::Ha) {
        if (auto reason = reply.failed()) {
            auto failed = decision.tier;
            S1_LOG(log::Level::Warn,
                   "model failed, failing over failed=" + std::string(upstream::tier_str(failed)) +
                       " failover=" + upstream::tier_str(upstream::other_tier(failed)) +
                       " reason=" + *reason);
            if (reply.call) reply.call->cancel();
            decision.tier = upstream::other_tier(failed);
            decision.failed_over_from = failed;
            reply = ask_tier(*state, decision.tier, parsed.value, dialect);
        }
    }

    if (reply.kind == TierReply::Kind::Dead) {
        S1_LOG(log::Level::Warn, "upstream unreachable protocol=anthropic tier=" +
                                     std::string(upstream::tier_str(decision.tier)) +
                                     " error=" + reply.message);
        render_api_error(res, protocol, 502,
                         std::string(upstream::tier_str(decision.tier)) +
                             " tier request failed: " + reply.message);
        stamp(res, decision);
        return;
    }
    if (reply.kind == TierReply::Kind::Bad) {
        render_api_error(res, protocol, 400, reply.message);
        return;
    }

    const upstream::Upstream& up = state->upstream(decision.tier);
    log_routed(protocol, decision, reply.status, started);

    if (!is_success(reply.status)) {
        // An error arrived in one dialect; hand it to the client in the shape
        // the client expects.
        std::string text = reply.call ? reply.call->drain() : reply.buffered;
        render_api_error(res, protocol, static_cast<int>(reply.status), upstream_message(text));
        stamp(res, decision);
        return;
    }

    if (reply.answer == TierReply::Answer::Native) {
        // Verbatim Messages traffic.
        forward_stream(res, reply.status, reply.headers, std::move(reply.call));
    } else if (reply.answer == TierReply::Answer::ChatStream) {
        forward_translated_stream(res, up.model, std::move(reply.call));
    } else {
        Json completion = Json::parse(reply.buffered, nullptr, false);
        if (completion.is_discarded()) {
            render_api_error(
                res, protocol, 502,
                std::string(upstream::tier_str(decision.tier)) +
                    " tier returned an unreadable response: " + reply.buffered.substr(0, 200));
            stamp(res, decision);
            return;
        }
        res.status = 200;
        res.set_content(anthropic::response::from_chat(completion, up.model).dump(),
                        "application/json");
    }
    stamp(res, decision);
}

// --- auxiliary routes ---

void handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    auto parsed = parse_object(req.body);
    if (!parsed.ok) {
        render_api_error(res, Protocol::Anthropic, 400, parsed.message);
        return;
    }
    res.status = 200;
    res.set_content(Json{{"input_tokens", anthropic::estimate_tokens(parsed.value)}}.dump(),
                    "application/json");
}

void handle_models(const std::shared_ptr<State>& state, httplib::Response& res) {
    std::vector<std::string> ids{kVirtualModel, state->top.model, state->flash.model};
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    Json data = Json::array();
    for (const auto& id : ids) {
        // Each entry carries both the OpenAI and the Anthropic field names.
        data.push_back(Json{{"id", id},
                            {"object", "model"},
                            {"type", "model"},
                            {"display_name", id},
                            {"created", 0},
                            {"created_at", "1970-01-01T00:00:00Z"},
                            {"owned_by", "s1-irrlab"}});
    }
    res.status = 200;
    res.set_content(Json{{"object", "list"}, {"data", data}, {"has_more", false}}.dump(),
                    "application/json");
}

void handle_health(const std::shared_ptr<State>& state, httplib::Response& res) {
    auto tier = [](const config::TierConfig& cfg) {
        return Json{
            {"base_url", cfg.base_url}, {"model", cfg.model}, {"api_key", cfg.api_key.has_value()}};
    };
    const config::Config& cfg = state->cfg;
    res.status = 200;
    res.set_content(Json{{"status", "ok"},
                         {"top", tier(cfg.top)},
                         {"flash", tier(cfg.flash)},
                         {"decider", Json{{"url", cfg.decider.url},
                                          {"model", cfg.decider.model},
                                          {"threshold", cfg.decider.threshold},
                                          {"timeout_ms", cfg.decider.timeout_ms},
                                          {"sticky_ttl_secs", cfg.decider.sticky_ttl_secs}}}}
                        .dump(),
                    "application/json");
}

}  // namespace

Router::Router(config::Config cfg) : state_(std::make_shared<State>(std::move(cfg))) {}

Router::~Router() = default;

void Router::install(httplib::Server& svr) {
    svr.set_payload_max_length(kMaxBodyBytes);
    svr.Get("/health", [state = state_](const httplib::Request&, httplib::Response& res) {
        handle_health(state, res);
    });
    svr.Get("/v1/models", [state = state_](const httplib::Request&, httplib::Response& res) {
        handle_models(state, res);
    });
    svr.Post("/v1/chat/completions",
             [state = state_](const httplib::Request& req, httplib::Response& res) {
                 handle_chat(state, req, res);
             });
    svr.Post("/v1/messages", [state = state_](const httplib::Request& req, httplib::Response& res) {
        handle_messages(state, req, res);
    });
    svr.Post("/v1/messages/count_tokens", [](const httplib::Request& req, httplib::Response& res) {
        handle_count_tokens(req, res);
    });
    // Everything else: the axum fallback, shaping the 404 for the caller's
    // dialect. Only untouched routing failures (404, empty body) are replaced;
    // error bodies produced by the handlers above pass through.
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (res.status == 404 && res.body.empty()) {
            Protocol protocol =
                req.path.rfind("/v1/messages", 0) == 0 ? Protocol::Anthropic : Protocol::OpenAi;
            render_api_error(res, protocol, 404, "no route for " + req.path);
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
}

}  // namespace s1::server

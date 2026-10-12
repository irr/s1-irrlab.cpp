// Unit tests: direct ports of the #[cfg(test)] modules of the Rust sources
// (config.rs, decider.rs, upstream.rs, openai.rs, anthropic/*).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <httplib.h>

#include "anthropic/mod.hpp"
#include "anthropic/request.hpp"
#include "anthropic/response.hpp"
#include "anthropic/stream.hpp"
#include "config.hpp"
#include "decider.hpp"
#include "json.hpp"
#include "openai.hpp"
#include "upstream.hpp"
#include "utf8.hpp"

using namespace s1;
using config::ConfigResult;
using decider::Conversation;
using decider::Role;
using http::HeaderEntry;
using s1::Json;

// ---------------------------------------------------------------- config ---

const std::string kFull = R"(
        [top]
        base_url = "https://gw.example.com/v1/"
        api_key = "{env:GW_KEY}"
        model = "big"

        [flash]
        base_url = "http://192.168.1.209:8000/v1"
        model = "Qwen/Qwen3.8-Flash-Next"

        [decider]
        url = "http://127.0.0.1:8000/v1/decisions"
    )";

std::optional<std::string> env(const std::string& name) {
    if (name == "GW_KEY") return std::string("secret");
    return std::nullopt;
}

std::vector<std::string> problems(const std::string& text) {
    ConfigResult result = config::parse(text, env);
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.kind == config::ConfigError::Kind::Invalid);
    return result.error.problems;
}

bool has_problem(const std::vector<std::string>& list, const std::string& needle) {
    for (const auto& problem : list) {
        if (problem.find(needle) != std::string::npos) return true;
    }
    return false;
}

TEST_CASE("config: parses with defaults") {
    ConfigResult result = config::parse(kFull, env);
    REQUIRE(result.ok);
    const auto& cfg = result.value;
    CHECK(cfg.server.host == "127.0.0.1");
    CHECK(cfg.server.port == 1970);
    CHECK(cfg.top.base_url == "https://gw.example.com/v1");
    CHECK(cfg.top.api_key == std::optional<std::string>("secret"));
    CHECK(cfg.flash.api_key == std::nullopt);
    CHECK(cfg.flash.timeout_secs == std::nullopt);
    CHECK(cfg.decider.model == "clef-flash");
    CHECK(cfg.decider.threshold == 0.5);
    CHECK(cfg.decider.on_error == config::OnError::Top);
    CHECK(cfg.decider.sticky_ttl_secs == 900);
}

TEST_CASE("config: reports every missing piece") {
    auto list = problems("[top]\nmodel = \"big\"\n");
    CHECK(has_problem(list, "top.base_url is required"));
    CHECK(has_problem(list, "[flash] section is required"));
    CHECK(has_problem(list, "[decider] section is required"));
}

TEST_CASE("config: rejects placeholders and bad urls") {
    std::string text = kFull;
    const std::string from = "model = \"big\"";
    const std::string to = "model = \"REPLACE_ME\"";
    text.replace(text.find(from), from.size(), to);
    const std::string bad_url = "http://127.0.0.1:8000/v1/decisions";
    const std::string with_bad = "localhost:8000";
    text.replace(text.find(bad_url), bad_url.size(), with_bad);
    auto list = problems(text);
    CHECK(has_problem(list, "top.model is not set"));
    CHECK(has_problem(list, "decider.url is not a valid"));
}

TEST_CASE("config: rejects unset env var") {
    std::string text = kFull;
    const std::string from = "GW_KEY";
    const std::string to = "MISSING_KEY";
    text.replace(text.find(from), from.size(), to);
    CHECK(has_problem(problems(text), "MISSING_KEY"));
}

TEST_CASE("config: rejects unknown keys") {
    ConfigResult result = config::parse(kFull + "\nsticky = 1\n", env);
    CHECK(!result.ok);
    CHECK(result.error.kind == config::ConfigError::Kind::Parse);
}

// --------------------------------------------------------------- decider ---

Conversation conv_of(std::vector<std::pair<Role, std::string>> turns) {
    Conversation conv;
    for (auto& [role, text] : turns) conv.push(role, std::move(text));
    return conv;
}

TEST_CASE("decider: state uses latest user text not tool result") {
    Conversation conv = conv_of({
        {Role::User, "refactor the parser"},
        {Role::Assistant, "[tool call: read_file]"},
        {Role::Tool, "fn parse() {}"},
    });
    Json state = conv.state();
    CHECK(state["latest_user_request"] == "refactor the parser");
    CHECK(state["last_turn_is_tool_result"] == true);
    CHECK(state["recent_context"].size() == 2);
    CHECK(state["turns"] == 3);
}

TEST_CASE("decider: state strips reminders and clips") {
    std::string long_text(10000, 'x');
    Conversation conv =
        conv_of({{Role::User, "<system-reminder>ignore me</system-reminder>" + long_text}});
    std::string request = conv.state()["latest_user_request"].get<std::string>();
    CHECK(request.find("ignore me") == std::string::npos);
    CHECK(utf8::length(request) == 4000 + 2000 + 3);
}

TEST_CASE("decider: sticky key is stable across a tool loop") {
    Conversation first = conv_of({{Role::User, "fix the bug"}});
    Conversation later = conv_of({
        {Role::User, "fix the bug"},
        {Role::Assistant, "[tool call: grep]"},
        {Role::Tool, "match"},
    });
    Conversation next = conv_of({
        {Role::User, "fix the bug"},
        {Role::Assistant, "done"},
        {Role::User, "thanks"},
    });
    CHECK(first.sticky_key() == later.sticky_key());
    CHECK(first.sticky_key() != next.sticky_key());
}

TEST_CASE("decider: reads probability then choice") {
    Json full = Json::parse(
        R"({"answers": {"tier": {"choice": "flash", "probabilities": {"flash": 0.3, "top": 0.7}}}})");
    CHECK(decider::p_top(full) == 0.7);
    CHECK(decider::p_top(Json::parse(R"({"answers": {"tier": {"choice": "flash"}}})")) ==
          std::optional<double>(0.0));
    CHECK(decider::p_top(Json::parse(R"({"answers": {}})")) == std::nullopt);
}

// -------------------------------------------------------------- upstream ---

config::TierConfig test_tier() {
    return config::TierConfig{"http://127.0.0.1:1/v1", std::nullopt, "qwen", std::nullopt,
                              std::optional<uint64_t>(32768)};
}

TEST_CASE("upstream: native body keeps the client body and rewrites tier fields") {
    upstream::Upstream up(test_tier());
    Json request = Json::parse(R"({
        "model": "claude-something",
        "max_tokens": 64000,
        "stream": true,
        "system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}],
        "metadata": {"user_id": "u"}
    })");
    Json body = up.native_body(request);
    CHECK(body["model"] == "qwen");
    CHECK(body["max_tokens"] == 32768);
    CHECK(body["stream"] == true);
    CHECK(body["metadata"]["user_id"] == "u");
    // Dialect fields travel verbatim, cache_control included.
    CHECK(body["system"][0]["cache_control"]["type"] == "ephemeral");
}

TEST_CASE("upstream: native body without cap or max_tokens is untouched apart from model") {
    config::TierConfig tier = test_tier();
    tier.max_output_tokens = std::nullopt;
    upstream::Upstream up(tier);
    Json body = up.native_body(Json::parse(R"({"model": "claude", "messages": []})"));
    CHECK(body == Json::parse(R"({"model": "qwen", "messages": []})"));
}

TEST_CASE("upstream: dialect headers only keep anthropic metadata") {
    httplib::Headers headers;
    headers.emplace("anthropic-version", "2023-06-01");
    headers.emplace("anthropic-beta", "tools-2024-04-04");
    headers.emplace("x-api-key", "secret");
    auto kept = upstream::MessagesHeaders::extract(headers);
    CHECK(kept.version == std::optional<std::string>("2023-06-01"));
    CHECK(kept.beta == std::optional<std::string>("tools-2024-04-04"));
}

TEST_CASE("upstream: tier alternation") {
    CHECK(upstream::other_tier(upstream::Tier::Top) == upstream::Tier::Flash);
    CHECK(upstream::other_tier(upstream::Tier::Flash) == upstream::Tier::Top);
    CHECK(std::string_view(upstream::tier_str(upstream::Tier::Top)) == "top");
}

// ---------------------------------------------------------------- openai ---

TEST_CASE("openai: builds conversation from mixed content") {
    Json body = Json::parse(R"({
        "tools": [{"type": "function"}, {"type": "function"}],
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "a cat"}
        ]
    })");
    auto conv = openai::conversation(body);
    CHECK(conv.system == "be brief");
    CHECK(conv.tools == 2);
    CHECK(conv.has_images);
    CHECK(conv.turns.size() == 3);
    CHECK(conv.turns[0].text == "what is this?");
    CHECK(conv.turns[1].text == "[tool call: look]");
    CHECK(conv.turns[2].role == Role::Tool);
}

// ------------------------------------------------- anthropic: mod + request ---

TEST_CASE("anthropic: tool results are not user text") {
    Json body = Json::parse(R"({
        "system": [{"type": "text", "text": "you are helpful"}],
        "tools": [{"name": "read", "input_schema": {}}],
        "messages": [
            {"role": "user", "content": "open main.rs"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "main.rs"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "fn main() {}"}
            ]}
        ]
    })");
    auto conv = anthropic::conversation(body);
    CHECK(conv.system == "you are helpful");
    CHECK(conv.tools == 1);
    CHECK(conv.turns.back().role == Role::Tool);
    CHECK(conv.state()["latest_user_request"] == "open main.rs");
}

TEST_CASE("anthropic request: translates a tool round trip") {
    Json req = Json::parse(R"({
        "model": "claude-whatever",
        "max_tokens": 64000,
        "system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}],
        "stream": true,
        "stop_sequences": ["END"],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [
            {"name": "read", "description": "read a file", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search"}
        ],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
        "messages": [
            {"role": "user", "content": "open main.rs"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "main.rs"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "fn main() {}"}]},
                {"type": "text", "text": "now explain it"}
            ]}
        ]
    })");
    auto result = anthropic::request::to_chat(req, "qwen", std::optional<uint64_t>(32768));
    REQUIRE(result.ok);
    const Json& chat = result.value;
    CHECK(chat["model"] == "qwen");
    CHECK(chat["max_tokens"] == 32768);
    CHECK(chat["stop"] == Json::array({"END"}));
    CHECK(chat["stream_options"] == Json::parse(R"({"include_usage": true})"));
    CHECK(chat["tool_choice"] == "required");
    CHECK(chat["parallel_tool_calls"] == false);
    CHECK(!chat.contains("thinking"));
    CHECK(chat["tools"].size() == 1);
    CHECK(chat["tools"][0]["function"]["parameters"] == Json::parse(R"({"type": "object"})"));

    const Json& messages = chat["messages"];
    REQUIRE(messages.size() == 5);
    CHECK(messages[0] == Json::parse(R"({"role": "system", "content": "be terse"})"));
    CHECK(messages[1] == Json::parse(R"({"role": "user", "content": "open main.rs"})"));
    CHECK(messages[2]["content"] == "reading");
    CHECK(messages[2]["tool_calls"][0]["id"] == "t1");
    CHECK(messages[2]["tool_calls"][0]["function"]["arguments"] == R"({"path":"main.rs"})");
    CHECK(messages[3] ==
          Json::parse(R"({"role": "tool", "tool_call_id": "t1", "content": "fn main() {}"})"));
    CHECK(messages[4] == Json::parse(R"({"role": "user", "content": "now explain it"})"));
}

TEST_CASE("anthropic request: translates images and tool-only turns") {
    Json req = Json::parse(R"({
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "look", "input": {}}
            ]}
        ]
    })");
    auto result = anthropic::request::to_chat(req, "m", std::nullopt);
    REQUIRE(result.ok);
    const Json& chat = result.value;
    CHECK(chat["max_tokens"] == 100);
    CHECK(!chat.contains("tools"));
    const Json& messages = chat["messages"];
    CHECK(messages[0]["content"][1] ==
          Json::parse(R"({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})"));
    CHECK(messages[1]["content"] == nullptr);
    CHECK(messages[1]["tool_calls"][0]["function"]["arguments"] == "{}");
}

TEST_CASE("anthropic request: rejects malformed requests") {
    CHECK(!anthropic::request::to_chat(Json::parse(R"({"messages": "hi"})"), "m",
                                       std::nullopt)
               .ok);
    CHECK(!anthropic::request::to_chat(Json::parse(R"({"messages": [{"role": "robot", "content": "x"}]})"),
                                       "m", std::nullopt)
               .ok);
}

// ----------------------------------------------------- anthropic: response ---

TEST_CASE("anthropic response: translates text reasoning and tool calls") {
    Json resp = Json::parse(R"({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "reasoning_content": "let me look",
                "content": "checking",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.rs\"}"}}
                ]
            }
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}
    })");
    Json msg = anthropic::response::from_chat(resp, "qwen");
    CHECK(msg["type"] == "message");
    CHECK(msg["model"] == "qwen");
    CHECK(msg["stop_reason"] == "tool_use");
    CHECK(msg["content"][0]["type"] == "thinking");
    CHECK(msg["content"][1] == Json::parse(R"({"type": "text", "text": "checking"})"));
    CHECK(msg["content"][2] ==
          Json::parse(R"({"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.rs"}})"));
    CHECK(msg["usage"]["input_tokens"] == 60);
    CHECK(msg["usage"]["cache_read_input_tokens"] == 40);
    CHECK(msg["usage"]["output_tokens"] == 20);
}

TEST_CASE("anthropic response: maps stop reasons") {
    using anthropic::response::stop_reason;
    CHECK(std::string_view(stop_reason("stop", false)) == "end_turn");
    CHECK(std::string_view(stop_reason("stop", true)) == "tool_use");
    CHECK(std::string_view(stop_reason("length", true)) == "max_tokens");
    CHECK(std::string_view(stop_reason(std::nullopt, false)) == "end_turn");
}

TEST_CASE("anthropic response: empty reply still has a block") {
    Json msg = anthropic::response::from_chat(
        Json::parse(R"({"choices": [{"finish_reason": "stop", "message": {"content": null}}]})"),
        "m");
    CHECK(msg["content"] == Json::parse(R"([{"type": "text", "text": ""}])"));
    CHECK(msg["usage"]["input_tokens"] == 0);
}

// ------------------------------------------------------- anthropic: stream ---

std::vector<anthropic::stream::SseEvent> run_translator(const std::vector<Json>& chunks) {
    anthropic::stream::StreamTranslator translator("qwen");
    std::vector<anthropic::stream::SseEvent> events = translator.start();
    for (const auto& chunk : chunks) {
        auto batch = translator.on_chunk(chunk);
        events.insert(events.end(), batch.begin(), batch.end());
    }
    auto tail = translator.finish();
    events.insert(events.end(), tail.begin(), tail.end());
    return events;
}

Json chunk_delta(Json delta) {
    return Json{{"choices", Json::array({Json{{"index", 0},
                                              {"delta", std::move(delta)},
                                              {"finish_reason", nullptr}}})}};
}

std::vector<std::string> event_names(const std::vector<anthropic::stream::SseEvent>& events) {
    std::vector<std::string> names;
    for (const auto& event : events) names.emplace_back(event.event);
    return names;
}

TEST_CASE("anthropic stream: text stream") {
    auto events = run_translator({
        chunk_delta(Json{{"role", "assistant"}, {"content", ""}}),
        chunk_delta(Json{{"content", "Hel"}}),
        chunk_delta(Json{{"content", "lo"}}),
        Json::parse(R"({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})"),
        Json::parse(R"({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}})"),
    });
    CHECK(event_names(events) ==
          std::vector<std::string>{"message_start", "content_block_start",
                                    "content_block_delta", "content_block_delta",
                                    "content_block_stop", "message_delta", "message_stop"});
    CHECK(events[0].data["message"]["model"] == "qwen");
    CHECK(events[2].data["delta"] == Json::parse(R"({"type": "text_delta", "text": "Hel"})"));
    CHECK(events[5].data["delta"]["stop_reason"] == "end_turn");
    CHECK(events[5].data["usage"]["input_tokens"] == 12);
    CHECK(events[5].data["usage"]["output_tokens"] == 2);
    CHECK(events[5].data["type"] == "message_delta");
}

TEST_CASE("anthropic stream: thinking then text then parallel tools") {
    auto tool = [](uint64_t index, std::optional<std::string> id,
                   std::optional<std::string> name, const std::string& args) {
        Json call = Json{{"index", index}};
        call["id"] = id ? Json(*id) : Json(nullptr);
        call["function"] = Json{{"name", name ? Json(*name) : Json(nullptr)},
                               {"arguments", args}};
        return chunk_delta(Json{{"tool_calls", Json::array({call})}});
    };
    auto events = run_translator({
        chunk_delta(Json{{"reasoning_content", "think"}}),
        chunk_delta(Json{{"content", "ok"}}),
        tool(0, "call_a", "read", ""),
        tool(0, std::nullopt, std::nullopt, "{\"path\":"),
        tool(0, std::nullopt, std::nullopt, "\"a.rs\"}"),
        tool(1, "call_b", "grep", "{}"),
        Json::parse(R"({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})"),
    });
    std::vector<Json> starts;
    for (const auto& event : events) {
        if (std::string_view(event.event) == "content_block_start") starts.push_back(event.data);
    }
    REQUIRE(starts.size() == 4);
    CHECK(starts[0]["content_block"]["type"] == "thinking");
    CHECK(starts[1]["content_block"]["type"] == "text");
    CHECK(starts[2]["content_block"] ==
          Json::parse(R"({"type": "tool_use", "id": "call_a", "name": "read", "input": {}})"));
    CHECK(starts[3]["index"] == 3);
    CHECK(starts[3]["content_block"]["id"] == "call_b");

    // Every block is stopped before the next one starts.
    std::optional<Json> open;
    for (const auto& event : events) {
        const std::string name = event.event;
        if (name == "content_block_start") {
            CHECK(!open.has_value());
            open = event.data["index"];
        } else if (name == "content_block_delta") {
            REQUIRE(open.has_value());
            CHECK(*open == event.data["index"]);
        } else if (name == "content_block_stop") {
            REQUIRE(open.has_value());
            CHECK(*open == event.data["index"]);
            open.reset();
        }
    }
    std::string json_parts;
    for (const auto& event : events) {
        if (event.data.contains("index") && event.data["index"] == 2 &&
            event.data.contains("delta") &&
            event.data["delta"]["type"] == "input_json_delta") {
            json_parts += event.data["delta"]["partial_json"].get<std::string>();
        }
    }
    CHECK(json_parts == "{\"path\":\"a.rs\"}");
    const anthropic::stream::SseEvent* message_delta = nullptr;
    for (const auto& event : events) {
        if (std::string_view(event.event) == "message_delta") message_delta = &event;
    }
    REQUIRE(message_delta != nullptr);
    CHECK(message_delta->data["delta"]["stop_reason"] == "tool_use");
}

TEST_CASE("anthropic stream: same index new id starts a new tool block") {
    auto call = [](const std::string& id) {
        Json entry = Json{{"index", 0},
                           {"id", id},
                           {"function", Json{{"name", "f"}, {"arguments", "{}"}}}};
        return chunk_delta(Json{{"tool_calls", Json::array({entry})}});
    };
    auto events = run_translator({call("a"), call("b")});
    size_t starts = 0;
    for (const auto& event : events) {
        if (std::string_view(event.event) == "content_block_start") ++starts;
    }
    CHECK(starts == 2);
}

TEST_CASE("anthropic stream: empty stream emits one empty text block") {
    auto events = run_translator({});
    CHECK(event_names(events) ==
          std::vector<std::string>{"message_start", "content_block_start",
                                    "content_block_stop", "message_delta",
                                    "message_stop"});
}

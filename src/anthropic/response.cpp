#include "response.hpp"

#include <cstring>
#include <random>

namespace s1::anthropic::response {

namespace {

// 32 hex chars of a v4-format UUID (uuid::Uuid::new_v4().simple()).
std::string new_uuid_simple() {
    thread_local std::mt19937_64 gen(std::random_device{}());
    uint64_t words[2] = {gen(), gen()};
    auto* bytes = reinterpret_cast<unsigned char*>(words);
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (size_t i = 0; i < sizeof(words); ++i) {
        out += hex[bytes[i] >> 4];
        out += hex[bytes[i] & 0x0F];
    }
    return out;
}

uint64_t u64_at(const Json* value, const std::string& pointer) {
    if (!value) return 0;
    const Json* found = json_ptr(*value, pointer);
    if (!found) return 0;
    if (auto n = as_u64(*found)) return *n;
    return 0;
}

}  // namespace

std::string message_id() { return "msg_" + new_uuid_simple(); }

std::string tool_use_id() { return "toolu_" + new_uuid_simple(); }

const char* stop_reason(std::optional<std::string> finish_reason, bool has_tool_use) {
    if (finish_reason == "length") return "max_tokens";
    if (finish_reason == "tool_calls" || finish_reason == "function_call") {
        return "tool_use";
    }
    if (has_tool_use) return "tool_use";
    return "end_turn";
}

std::optional<std::string> reasoning(const Json& message) {
    for (const char* key : {"reasoning_content", "reasoning"}) {
        if (auto text = get_str(message, key)) {
            if (!text->empty()) return text;
        }
    }
    return std::nullopt;
}

Json usage(const Json* usage_value) {
    uint64_t cached = u64_at(usage_value, "/prompt_tokens_details/cached_tokens");
    uint64_t prompt = u64_at(usage_value, "/prompt_tokens");
    return Json{{"input_tokens", prompt >= cached ? prompt - cached : 0},
                {"output_tokens", u64_at(usage_value, "/completion_tokens")},
                {"cache_creation_input_tokens", 0},
                {"cache_read_input_tokens", cached}};
}

Json from_chat(const Json& resp, const std::string& model) {
    const Json* choice = json_ptr(resp, "/choices/0");
    const Json* message = choice && choice->contains("message") ? &choice->at("message") : nullptr;

    Json content = Json::array();
    if (message) {
        if (auto thinking = reasoning(*message)) {
            content.push_back(
                Json{{"type", "thinking"}, {"thinking", *thinking}, {"signature", ""}});
        }
    }
    std::string text;
    if (message && message->contains("content")) {
        const Json& value = message->at("content");
        if (value.is_string()) {
            text = value.get<std::string>();
        } else if (value.is_array()) {
            // Text parts are concatenated with no separator, like the Rust
            // Vec<&str> collect.
            for (const auto& part : value) {
                if (auto t = get_str(part, "text")) text += *t;
            }
        }
    }
    if (!text.empty()) {
        content.push_back(Json{{"type", "text"}, {"text", text}});
    }
    bool has_tool_use = false;
    if (message && message->contains("tool_calls") && message->at("tool_calls").is_array()) {
        for (const auto& call : message->at("tool_calls")) {
            Json input = Json::object();
            if (auto arguments = get_str_at(call, "/function/arguments")) {
                Json parsed = Json::parse(*arguments, nullptr, false);
                if (!parsed.is_discarded() && parsed.is_object()) input = parsed;
            }
            std::string id = tool_use_id();
            if (auto given = get_str(call, "id")) {
                if (!given->empty()) id = *given;
            }
            content.push_back(Json{{"type", "tool_use"},
                                   {"id", id},
                                   {"name", get_str_at(call, "/function/name").value_or("")},
                                   {"input", input}});
            has_tool_use = true;
        }
    }
    if (content.empty()) {
        content.push_back(Json{{"type", "text"}, {"text", ""}});
    }

    std::optional<std::string> finish;
    if (choice && choice->contains("finish_reason")) {
        finish = get_str(*choice, "finish_reason");
    }
    const Json* usage_value = resp.contains("usage") ? &resp.at("usage") : nullptr;

    return Json{{"id", message_id()},       {"type", "message"},
                {"role", "assistant"},      {"model", model},
                {"content", content},       {"stop_reason", stop_reason(finish, has_tool_use)},
                {"stop_sequence", nullptr}, {"usage", usage(usage_value)}};
}

}  // namespace s1::anthropic::response

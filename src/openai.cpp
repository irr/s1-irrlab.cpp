#include "openai.hpp"

#include "utf8.hpp"

namespace s1::openai {

namespace {

// Text of a message's `content` (string or parts) and whether it carries media.
// Mirrors the Rust content_text: parts without a string `text` field mark the
// message as carrying media (images, audio, ...).
std::pair<std::string, bool> content_text(const Json* content) {
    if (!content) return {"", false};
    if (content->is_string()) return {content->get<std::string>(), false};
    if (content->is_array()) {
        std::string text;
        bool media = false;
        for (const auto& part : *content) {
            if (auto t = get_str(part, "text")) {
                if (!text.empty()) text += "\n";
                text += *t;
            } else {
                media = true;
            }
        }
        return {text, media};
    }
    return {"", false};
}

}  // namespace

decider::Conversation conversation(const Json& body) {
    decider::Conversation conv;
    if (body.contains("tools") && body.at("tools").is_array()) {
        conv.tools = body.at("tools").size();
    }
    if (body.contains("messages") && body.at("messages").is_array()) {
        for (const auto& message : body.at("messages")) {
            auto [text, images] =
                content_text(message.contains("content") ? &message.at("content") : nullptr);
            conv.has_images = conv.has_images || images;
            auto role = get_str(message, "role");
            if (!role) continue;
            if (*role == "system" || *role == "developer") {
                if (!conv.system.empty()) conv.system += "\n\n";
                conv.system += text;
            } else if (*role == "user") {
                conv.push(decider::Role::User, std::move(text));
            } else if (*role == "assistant") {
                if (message.contains("tool_calls") && message.at("tool_calls").is_array()) {
                    for (const auto& call : message.at("tool_calls")) {
                        auto name = get_str_at(call, "/function/name");
                        text += "\n[tool call: " + name.value_or("?") + "]";
                    }
                }
                conv.push(decider::Role::Assistant, utf8::trim_copy(text));
            } else if (*role == "tool" || *role == "function") {
                conv.push(decider::Role::Tool, std::move(text));
            }
        }
    }
    return conv;
}

void prepare(Json& body, const upstream::Upstream& upstream) {
    body["model"] = upstream.model;
    if (!upstream.max_output_tokens) return;
    uint64_t limit = *upstream.max_output_tokens;
    for (const char* key : {"max_tokens", "max_completion_tokens"}) {
        if (auto requested = get_u64(body, key)) {
            if (*requested > limit) body[key] = limit;
        }
    }
}

}  // namespace s1::openai

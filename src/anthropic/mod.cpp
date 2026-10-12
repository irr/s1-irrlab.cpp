#include "mod.hpp"

#include "request.hpp"
#include "utf8.hpp"

namespace s1::anthropic {

decider::Conversation conversation(const Json& body) {
    decider::Conversation conv;
    conv.system = request::system_text(
        body.contains("system") ? &body.at("system") : nullptr);
    if (body.contains("tools") && body.at("tools").is_array()) {
        conv.tools = body.at("tools").size();
    }
    if (!body.contains("messages") || !body.at("messages").is_array()) return conv;

    for (const auto& message : body.at("messages")) {
        decider::Role role = get_str(message, "role") == "assistant"
                                 ? decider::Role::Assistant
                                 : decider::Role::User;
        const Json* content =
            message.contains("content") ? &message.at("content") : nullptr;
        if (!content) continue;
        if (content->is_string()) {
            conv.push(role, content->get<std::string>());
            continue;
        }
        if (!content->is_array()) continue;

        std::vector<std::string> text;
        for (const auto& block : *content) {
            auto type = get_str(block, "type");
            if (!type) continue;
            if (*type == "text") {
                if (auto t = get_str(block, "text")) text.push_back(*t);
            } else if (*type == "image" || *type == "document") {
                conv.has_images = true;
            } else if (*type == "tool_use") {
                text.push_back("[tool call: " + get_str(block, "name").value_or("?") +
                               "]");
            } else if (*type == "tool_result") {
                auto [result, images] = request::tool_result_content(
                    block.contains("content") ? &block.at("content") : nullptr);
                conv.has_images = conv.has_images || !images.empty();
                conv.push(decider::Role::Tool, std::move(result));
            }
        }
        if (!text.empty()) {
            std::string joined;
            for (size_t i = 0; i < text.size(); ++i) {
                if (i) joined += "\n";
                joined += text[i];
            }
            conv.push(role, std::move(joined));
        }
    }
    return conv;
}

uint64_t estimate_tokens(const Json& body) {
    uint64_t chars = 0;
    for (const char* key : {"system", "messages", "tools"}) {
        if (body.contains(key)) {
            chars += static_cast<uint64_t>(utf8::length(body.at(key).dump()));
        }
    }
    uint64_t tokens = (chars + 3) / 4;
    return tokens > 0 ? tokens : 1;
}

}  // namespace s1::anthropic

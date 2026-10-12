#include "request.hpp"

namespace s1::anthropic::request {

namespace {

const Json& null_value() {
    static const Json kNull = nullptr;
    return kNull;
}

// An image block as a chat-completions part: base64 sources become data URLs,
// url sources pass through, anything else is dropped.
std::optional<Json> image_part(const Json& block) {
    if (!block.contains("source") || !block.at("source").is_object()) return std::nullopt;
    const Json& source = block.at("source");
    auto kind = get_str(source, "type");
    std::optional<std::string> url;
    if (kind == "base64") {
        auto media_type = get_str(source, "media_type");
        auto data = get_str(source, "data");
        if (media_type && data) url = "data:" + *media_type + ";base64," + *data;
    } else if (kind == "url") {
        url = get_str(source, "url");
    }
    if (!url) return std::nullopt;
    return Json{{"type", "image_url"}, {"image_url", Json{{"url", *url}}}};
}

// A user message becomes one `tool` message per tool result, followed by a
// `user` message with whatever text and images remain. Tool messages go first
// because chat completions requires them directly after the assistant's calls.
void push_user(std::vector<Json>& out, const Json& content) {
    if (content.is_string()) {
        out.push_back(Json{{"role", "user"}, {"content", content.get<std::string>()}});
        return;
    }
    if (!content.is_array()) return;

    std::vector<Json> parts;
    for (const auto& block : content) {
        auto type = get_str(block, "type");
        if (!type) continue;
        if (*type == "text") {
            if (auto text = get_str(block, "text")) {
                parts.push_back(Json{{"type", "text"}, {"text", *text}});
            }
        } else if (*type == "image") {
            if (auto part = image_part(block)) parts.push_back(*part);
        } else if (*type == "document") {
            parts.push_back(Json{{"type", "text"},
                                 {"text",
                                  "[document omitted: not supported by this model]"}});
        } else if (*type == "tool_result") {
            auto [text, images] =
                tool_result_content(block.contains("content") ? &block.at("content")
                                                              : nullptr);
            Json tool_call_id =
                block.contains("tool_use_id") ? block.at("tool_use_id") : Json(nullptr);
            out.push_back(Json{{"role", "tool"},
                               {"tool_call_id", tool_call_id},
                               {"content", text}});
            for (const auto& image : images) parts.push_back(image);
        }
    }
    if (parts.empty()) return;

    bool all_text = true;
    for (const auto& part : parts) {
        if (get_str(part, "type") != "text") all_text = false;
    }
    Json content_value;
    if (all_text) {
        std::string joined;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i) joined += "\n\n";
            joined += parts[i].at("text").get<std::string>();
        }
        content_value = joined;
    } else {
        content_value = parts;
    }
    out.push_back(Json{{"role", "user"}, {"content", content_value}});
}

// Assistant text blocks become `content`, `tool_use` blocks become
// `tool_calls`; thinking blocks have no chat-completions equivalent.
Json assistant(const Json& content) {
    std::string text;
    std::vector<Json> calls;
    if (content.is_string()) {
        text = content.get<std::string>();
    } else if (content.is_array()) {
        for (const auto& block : content) {
            auto type = get_str(block, "type");
            if (!type) continue;
            if (*type == "text") {
                text += get_str(block, "text").value_or("");
            } else if (*type == "tool_use") {
                Json input = block.contains("input") ? block.at("input") : Json::object();
                calls.push_back(Json{
                    {"id", block.contains("id") ? block.at("id") : Json(nullptr)},
                    {"type", "function"},
                    {"function",
                     Json{{"name", block.contains("name") ? block.at("name")
                                                          : Json(nullptr)},
                          {"arguments", input.dump()}}}});
            }
        }
    }

    Json message;
    message["role"] = "assistant";
    if (text.empty() && !calls.empty()) {
        message["content"] = nullptr;
    } else {
        message["content"] = text;
    }
    if (!calls.empty()) message["tool_calls"] = calls;
    return message;
}

// Client tools only: server tools (web search, etc.) carry no `input_schema`
// and cannot run on a chat-completions upstream.
std::vector<Json> tools(const Json* value) {
    std::vector<Json> out;
    if (!value || !value->is_array()) return out;
    for (const auto& tool : *value) {
        if (!tool.is_object() || !tool.contains("input_schema")) continue;
        Json function;
        function["name"] = tool.at("name");
        if (tool.contains("description") && tool.at("description").is_string()) {
            function["description"] = tool.at("description");
        }
        function["parameters"] = tool.at("input_schema");
        out.push_back(Json{{"type", "function"}, {"function", function}});
    }
    return out;
}

std::optional<Json> tool_choice(const Json& choice) {
    auto type = get_str(choice, "type");
    if (!type) return std::nullopt;
    if (*type == "auto") return Json("auto");
    if (*type == "any") return Json("required");
    if (*type == "none") return Json("none");
    if (*type == "tool") {
        if (!choice.contains("name")) return std::nullopt;
        return Json{{"type", "function"}, {"function", Json{{"name", choice.at("name")}}}};
    }
    return std::nullopt;
}

}  // namespace

ToChatResult to_chat(const Json& req, const std::string& model,
                     std::optional<uint64_t> max_output) {
    ToChatResult result;
    if (!req.contains("messages") || !req.at("messages").is_array()) {
        result.error = "`messages` must be an array";
        return result;
    }

    std::vector<Json> out;
    std::string system =
        system_text(req.contains("system") ? &req.at("system") : nullptr);
    if (!system.empty()) {
        out.push_back(Json{{"role", "system"}, {"content", system}});
    }
    const Json& messages = req.at("messages");
    for (size_t i = 0; i < messages.size(); ++i) {
        const Json& message = messages[i];
        const Json& content = message.contains("content") ? message.at("content")
                                                          : null_value();
        auto role = get_str(message, "role");
        if (role == "user") {
            push_user(out, content);
        } else if (role == "assistant") {
            out.push_back(assistant(content));
        } else if (role == "system") {
            out.push_back(Json{{"role", "system"}, {"content", system_text(&content)}});
        } else if (role) {
            result.error = "messages[" + std::to_string(i) + "].role `" + *role +
                           "` is not supported";
            return result;
        } else {
            result.error = "messages[" + std::to_string(i) + "].role is required";
            return result;
        }
    }

    Json chat;
    chat["model"] = model;
    chat["messages"] = out;

    if (auto max = get_u64(req, "max_tokens")) {
        chat["max_tokens"] = max_output ? std::min(*max, *max_output) : *max;
    }
    for (const char* key : {"temperature", "top_p"}) {
        if (req.contains(key) && req.at(key).is_number()) {
            chat[key] = req.at(key);
        }
    }
    if (req.contains("stop_sequences") && req.at("stop_sequences").is_array() &&
        !req.at("stop_sequences").empty()) {
        chat["stop"] = req.at("stop_sequences");
    }
    if (get_bool(req, "stream") == true) {
        chat["stream"] = true;
        chat["stream_options"] = Json{{"include_usage", true}};
    }

    std::vector<Json> translated_tools =
        tools(req.contains("tools") ? &req.at("tools") : nullptr);
    if (!translated_tools.empty()) {
        chat["tools"] = translated_tools;
        if (req.contains("tool_choice")) {
            const Json& choice = req.at("tool_choice");
            if (auto mapped = tool_choice(choice)) {
                chat["tool_choice"] = *mapped;
            }
            if (get_bool(choice, "disable_parallel_tool_use") == true) {
                chat["parallel_tool_calls"] = false;
            }
        }
    }

    result.ok = true;
    result.value = std::move(chat);
    return result;
}

std::string system_text(const Json* system) {
    if (!system) return "";
    if (system->is_string()) return system->get<std::string>();
    if (system->is_array()) {
        std::vector<std::string> texts;
        for (const auto& block : *system) {
            if (auto text = get_str(block, "text")) texts.push_back(*text);
        }
        std::string joined;
        for (size_t i = 0; i < texts.size(); ++i) {
            if (i) joined += "\n\n";
            joined += texts[i];
        }
        return joined;
    }
    return "";
}

std::pair<std::string, std::vector<Json>> tool_result_content(const Json* content) {
    if (!content) return {"", {}};
    if (content->is_string()) return {content->get<std::string>(), {}};
    if (!content->is_array()) return {"", {}};

    std::vector<std::string> text;
    std::vector<Json> images;
    for (const auto& block : *content) {
        auto type = get_str(block, "type");
        if (!type) continue;
        if (*type == "text") {
            if (auto t = get_str(block, "text")) text.push_back(*t);
        } else if (*type == "image") {
            if (auto part = image_part(block)) images.push_back(*part);
        }
    }
    std::string joined;
    for (size_t i = 0; i < text.size(); ++i) {
        if (i) joined += "\n";
        joined += text[i];
    }
    return {joined, images};
}

}  // namespace s1::anthropic::request

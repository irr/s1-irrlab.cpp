#include "stream.hpp"

#include "log.hpp"
#include "response.hpp"
#include "utf8.hpp"

namespace s1::anthropic::stream {

namespace {

SseEvent new_event(const char* event, Json data) {
    data["type"] = event;
    return SseEvent{event, std::move(data)};
}

}  // namespace

std::string SseEvent::encode() const {
    return std::string("event: ") + event + "\ndata: " + data.dump() + "\n\n";
}

// --- SseParser ---

void SseParser::line(const std::string& text) {
    if (text.empty()) {
        dispatch();
        return;
    }
    if (text[0] == ':') return;  // comment / keep-alive
    size_t colon = text.find(':');
    std::string field = colon == std::string::npos ? text : text.substr(0, colon);
    if (field != "data") return;
    if (colon == std::string::npos) {
        data_lines_.push_back("");
        return;
    }
    std::string value = text.substr(colon + 1);
    if (!value.empty() && value[0] == ' ') value.erase(0, 1);
    data_lines_.push_back(value);
}

void SseParser::dispatch() {
    if (!data_lines_.empty()) {
        std::string data;
        for (size_t i = 0; i < data_lines_.size(); ++i) {
            if (i) data += "\n";
            data += data_lines_[i];
        }
        ready_.push_back(std::move(data));
    }
    data_lines_.clear();
}

std::vector<std::string> SseParser::drain() {
    // Extract complete lines; a trailing lone \r may be half of a \r\n.
    size_t pos = 0;
    while (pos < buffer_.size()) {
        size_t end = std::string::npos;
        size_t skip = 0;
        for (size_t i = pos; i < buffer_.size(); ++i) {
            if (buffer_[i] == '\n') {
                end = i;
                skip = 1;
                break;
            }
            if (buffer_[i] == '\r') {
                if (i + 1 == buffer_.size()) break;  // wait for a possible \n
                end = i;
                skip = (buffer_[i + 1] == '\n') ? 2 : 1;
                break;
            }
        }
        if (end == std::string::npos) break;  // incomplete line
        line(buffer_.substr(pos, end - pos));
        pos = end + skip;
    }
    buffer_.erase(0, pos);
    std::vector<std::string> out = std::move(ready_);
    ready_.clear();
    return out;
}

// --- StreamTranslator ---

StreamTranslator::StreamTranslator(std::string model) : model_(std::move(model)) {}

std::vector<SseEvent> StreamTranslator::start() const {
    return {new_event(
        "message_start",
        Json{{"message",
              Json{{"id", response::message_id()},
                   {"type", "message"},
                   {"role", "assistant"},
                   {"model", model_},
                   {"content", Json::array()},
                   {"stop_reason", nullptr},
                   {"stop_sequence", nullptr},
                   {"usage", response::usage(nullptr)}}}})};
}

std::vector<SseEvent> StreamTranslator::on_chunk(const Json& chunk) {
    std::vector<SseEvent> out;
    if (chunk.contains("usage") && chunk.at("usage").is_object()) {
        usage_ = chunk.at("usage");
    }
    const Json* choice = json_ptr(chunk, "/choices/0");
    if (!choice) return out;
    if (auto reason = get_str(*choice, "finish_reason")) {
        finish_reason_ = reason;
    }
    if (!choice->contains("delta")) return out;
    const Json& delta_value = choice->at("delta");

    if (auto thinking = response::reasoning(delta_value)) {
        if (open_.kind != Open::Kind::Thinking) {
            open_block(out, Open{Open::Kind::Thinking, 0, ""},
                       Json{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}});
        }
        delta(out, Json{{"type", "thinking_delta"}, {"thinking", *thinking}});
    }
    if (auto text = get_str(delta_value, "content")) {
        if (!text->empty()) {
            if (open_.kind != Open::Kind::Text) {
                open_block(out, Open{Open::Kind::Text, 0, ""},
                           Json{{"type", "text"}, {"text", ""}});
            }
            delta(out, Json{{"type", "text_delta"}, {"text", *text}});
        }
    }
    if (delta_value.contains("tool_calls") && delta_value.at("tool_calls").is_array()) {
        for (const auto& call : delta_value.at("tool_calls")) {
            tool_delta(out, call);
        }
    }
    return out;
}

void StreamTranslator::tool_delta(std::vector<SseEvent>& out, const Json& call) {
    uint64_t index = get_u64(call, "index").value_or(0);
    std::optional<std::string> id;
    if (auto given = get_str(call, "id")) {
        if (!given->empty()) id = given;
    }
    // Continues the open block unless the index changes, or a new id shows up
    // on the same index (some gateways number every parallel call 0).
    bool continues = open_.kind == Open::Kind::Tool && open_.index == index &&
                     (!id || *id == open_.id);
    if (!continues) {
        if (!id && tools_seen_.count(index)) {
            S1_LOG(log::Level::Warn,
                   "dropping tool-call delta for an already closed block index=" +
                       std::to_string(index));
            return;
        }
        std::string block_id = id.value_or(response::tool_use_id());
        std::string name = get_str_at(call, "/function/name").value_or("");
        Json block = Json{{"type", "tool_use"}, {"id", block_id}, {"name", name},
                          {"input", Json::object()}};
        tools_seen_.insert(index);
        open_block(out, Open{Open::Kind::Tool, index, block_id}, std::move(block));
    }
    if (auto arguments = get_str_at(call, "/function/arguments")) {
        if (!arguments->empty()) {
            delta(out, Json{{"type", "input_json_delta"}, {"partial_json", *arguments}});
        }
    }
}

void StreamTranslator::open_block(std::vector<SseEvent>& out, Open open, Json block) {
    close_block(out);
    open_ = open;
    out.push_back(new_event(
        "content_block_start",
        Json{{"index", blocks_}, {"content_block", std::move(block)}}));
    blocks_ += 1;
}

void StreamTranslator::close_block(std::vector<SseEvent>& out) {
    if (open_.kind != Open::Kind::None) {
        open_ = Open{};
        out.push_back(new_event("content_block_stop", Json{{"index", blocks_ - 1}}));
    }
}

void StreamTranslator::delta(std::vector<SseEvent>& out, Json delta_value) {
    out.push_back(new_event(
        "content_block_delta",
        Json{{"index", blocks_ - 1}, {"delta", std::move(delta_value)}}));
}

std::vector<SseEvent> StreamTranslator::finish() {
    std::vector<SseEvent> out;
    if (blocks_ == 0) {
        open_block(out, Open{Open::Kind::Text, 0, ""},
                   Json{{"type", "text"}, {"text", ""}});
    }
    close_block(out);
    const char* reason =
        response::stop_reason(finish_reason_, !tools_seen_.empty());
    out.push_back(new_event(
        "message_delta",
        Json{{"delta", Json{{"stop_reason", reason}, {"stop_sequence", nullptr}}},
         {"usage", response::usage(usage_ ? &*usage_ : nullptr)}}));
    out.push_back(new_event("message_stop", Json::object()));
    return out;
}

SseEvent StreamTranslator::error_event(const std::string& message) {
    return new_event("error",
                     Json{{"error", Json{{"type", "api_error"}, {"message", message}}}});
}

}  // namespace s1::anthropic::stream

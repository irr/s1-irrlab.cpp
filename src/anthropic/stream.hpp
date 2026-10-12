// OpenAI `chat.completion.chunk` stream → Anthropic Messages event stream.
// Port of src/anthropic/stream.rs; the network loop that feeds the translator
// lives in server.cpp, this module is the pure state machine plus the SSE
// frame parser.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "json.hpp"

namespace s1::anthropic::stream {

/// One Anthropic SSE event: the frame name and its JSON payload.
struct SseEvent {
    const char* event = "";
    Json data = Json::object();

    /// The wire frame: `event: <name>\ndata: <json>\n\n`.
    std::string encode() const;
};

/// Incremental parser for an upstream SSE body (eventsource-stream's part):
/// feed raw bytes, take the `data` payloads of every completed frame.
class SseParser {
public:
    void feed(const char* data, size_t len) { buffer_.append(data, len); }
    void feed(const std::string& data) { buffer_ += data; }
    /// Data fields of all frames completed so far, in order.
    std::vector<std::string> drain();

private:
    std::string buffer_;
    std::vector<std::string> ready_;
    std::vector<std::string> data_lines_;

    void line(const std::string& text);
    void dispatch();
};

/// Turns chunks into Anthropic events. Anthropic content blocks are strictly
/// sequential, so a block is closed as soon as output moves on to another one.
class StreamTranslator {
public:
    explicit StreamTranslator(std::string model);

    std::vector<SseEvent> start() const;
    std::vector<SseEvent> on_chunk(const Json& chunk);
    /// Close the message once the upstream stream has ended.
    std::vector<SseEvent> finish();
    static SseEvent error_event(const std::string& message);

private:
    /// The content block currently receiving deltas.
    struct Open {
        enum class Kind { None, Thinking, Text, Tool };
        Kind kind = Kind::None;
        uint64_t index = 0;
        std::string id;
    };

    void tool_delta(std::vector<SseEvent>& out, const Json& call);
    void open_block(std::vector<SseEvent>& out, Open open, Json block);
    void close_block(std::vector<SseEvent>& out);
    void delta(std::vector<SseEvent>& out, Json delta);

    std::string model_;
    Open open_;
    /// Blocks started so far; the open block's index is `blocks - 1`.
    size_t blocks_ = 0;
    /// Upstream tool-call indices that already have a block.
    std::unordered_set<uint64_t> tools_seen_;
    std::optional<std::string> finish_reason_;
    std::optional<Json> usage_;
};

}  // namespace s1::anthropic::stream

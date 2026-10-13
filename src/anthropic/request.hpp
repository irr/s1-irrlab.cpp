// Anthropic Messages request → OpenAI chat-completions request.
// Port of src/anthropic/request.rs.

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

namespace s1::anthropic::request {

struct ToChatResult {
    bool ok = false;
    Json value{};
    /// Client-facing message when !ok.
    std::string error;
};

/// Translate a Messages request for `model`. !ok is a client-facing message.
ToChatResult to_chat(const Json& req, const std::string& model, std::optional<uint64_t> max_output);

/// `system` is either a string or a list of text blocks.
std::string system_text(const Json* system);

/// Text of a `tool_result` plus any images it carries (as chat-completions parts).
std::pair<std::string, std::vector<Json>> tool_result_content(const Json* content);

}  // namespace s1::anthropic::request

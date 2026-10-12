// OpenAI chat-completions response → Anthropic Messages response.
// Port of src/anthropic/response.rs.

#pragma once

#include <optional>
#include <string>

#include "json.hpp"

namespace s1::anthropic::response {

std::string message_id();
std::string tool_use_id();

/// `finish_reason` → `stop_reason`. Some servers report `stop` even when the
/// turn ended in tool calls, so their presence wins.
const char* stop_reason(std::optional<std::string> finish_reason, bool has_tool_use);

/// Reasoning text, under either name vLLM has used for it.
std::optional<std::string> reasoning(const Json& message);

/// Anthropic `usage` from a chat-completions `usage` object.
Json usage(const Json* usage);

/// Translate a complete (non-streamed) response; `model` is the model that answered.
Json from_chat(const Json& resp, const std::string& model);

}  // namespace s1::anthropic::response

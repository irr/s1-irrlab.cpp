// Anthropic Messages front end. Both upstreams only speak chat completions,
// so requests and responses (plain and streamed) are translated here.
// Port of src/anthropic/mod.rs.

#pragma once

#include <cstdint>

#include "decider.hpp"
#include "json.hpp"

namespace s1::anthropic {

/// Build the decider's view of a Messages request.
decider::Conversation conversation(const Json& body);

/// Rough token count for /v1/messages/count_tokens; neither upstream offers one.
uint64_t estimate_tokens(const Json& body);

}  // namespace s1::anthropic

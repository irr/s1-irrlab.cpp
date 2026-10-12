// OpenAI chat-completions front end. Requests are forwarded as-is apart from
// the `model` rewrite, so this module only reads the body for the decider and
// patches the two fields the proxy owns. Port of src/openai.rs.

#pragma once

#include <string>

#include "decider.hpp"
#include "json.hpp"
#include "upstream.hpp"

namespace s1::openai {

/// Build the decider's view of a chat-completions request.
decider::Conversation conversation(const Json& body);

/// Point the request at the chosen tier: its model name and output-token limit.
void prepare(Json& body, const upstream::Upstream& upstream);

}  // namespace s1::openai

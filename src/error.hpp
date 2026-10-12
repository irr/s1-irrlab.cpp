// Errors shaped for whichever API the caller is speaking. Port of src/error.rs.

#pragma once

#include <string>

#include "json.hpp"

namespace s1 {

enum class Protocol { OpenAi, Anthropic };

inline const char* protocol_str(Protocol protocol) {
    return protocol == Protocol::OpenAi ? "openai" : "anthropic";
}

// The JSON body rendered for an error in the caller's dialect.
Json api_error_body(Protocol protocol, int status, const std::string& message);

// The Anthropic error body: {"type":"error","error":{"type":..., "message":...}}.
Json anthropic_error(int status, const std::string& message);

// Pull a human-readable message out of an upstream error body.
std::string upstream_message(const std::string& body);

}  // namespace s1

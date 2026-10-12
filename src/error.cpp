#include "error.hpp"

#include "utf8.hpp"

namespace s1 {

namespace {

bool is_client_error(int status) { return status >= 400 && status < 500; }

const char* anthropic_kind(int status) {
    switch (status) {
        case 400:
        case 422: return "invalid_request_error";
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        case 413: return "request_too_large";
        case 429: return "rate_limit_error";
        case 503:
        case 529: return "overloaded_error";
        default: return "api_error";
    }
}

}  // namespace

Json api_error_body(Protocol protocol, int status, const std::string& message) {
    if (protocol == Protocol::Anthropic) return anthropic_error(status, message);
    Json body = Json{{"error", Json{{"message", message},
                                    {"type", is_client_error(status)
                                                 ? "invalid_request_error"
                                                 : "api_error"},
                                    {"param", nullptr},
                                    {"code", nullptr}}}};
    return body;
}

Json anthropic_error(int status, const std::string& message) {
    return Json{{"type", "error"},
                {"error", Json{{"type", anthropic_kind(status)}, {"message", message}}}};
}

std::string upstream_message(const std::string& body) {
    Json parsed = Json::parse(body, nullptr, false);
    // The first candidate that EXISTS is taken, whatever its type; only a
    // string is used, otherwise the whole body becomes the message (the
    // or_else chain in the Rust original behaves exactly this way).
    const Json* found = nullptr;
    if (!parsed.is_discarded()) {
        const Json* candidates[] = {
            json_ptr(parsed, "/error/message"),
            parsed.is_object() && parsed.contains("error") ? &parsed.at("error") : nullptr,
            parsed.is_object() && parsed.contains("message") ? &parsed.at("message")
                                                             : nullptr,
            parsed.is_object() && parsed.contains("detail") ? &parsed.at("detail")
                                                            : nullptr,
        };
        for (const Json* candidate : candidates) {
            if (candidate) {
                found = candidate;
                break;
            }
        }
    }
    std::string text = found && found->is_string() ? found->get<std::string>() : body;
    text = utf8::trim_copy(text);
    if (text.empty()) return "upstream returned an error";
    return utf8::take(text, 2000);
}

}  // namespace s1

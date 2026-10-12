// The two model tiers and the HTTP client that talks to each. Port of
// src/upstream.rs; streaming requests are driven by server.cpp through the
// URLs and headers exposed here.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <httplib.h>

#include "config.hpp"
#include "http.hpp"
#include "json.hpp"

namespace s1::upstream {

enum class Tier { Top, Flash };

inline const char* tier_str(Tier tier) {
    switch (tier) {
        case Tier::Top:
            return "top";
        case Tier::Flash:
            return "flash";
    }
    return "top";
}
inline Tier other_tier(Tier tier) {
    switch (tier) {
        case Tier::Top:
            return Tier::Flash;
        case Tier::Flash:
            return Tier::Top;
    }
    return Tier::Flash;
}

/// Client headers that must reach a Messages-capable upstream. Nothing else,
/// in particular no credentials, is forwarded.
struct MessagesHeaders {
    std::optional<std::string> version;
    std::optional<std::string> beta;

    static MessagesHeaders extract(const httplib::Headers& client);

    // Dialect headers for the upstream request (always a version, defaulting
    // like the Rust code, plus the beta when the client sent one).
    void apply(std::vector<http::HeaderEntry>& headers) const;
};

/// One upstream, reachable both as a chat-completions server and, for servers
/// that serve it, as an Anthropic Messages server (both paths hang off the
/// configured base_url).
class Upstream {
public:
    explicit Upstream(const config::TierConfig& cfg);

    /// POST a chat-completions body, reading the whole response. The caller's
    /// own credentials are never forwarded; only this tier's configured key
    /// (if any) is sent.
    http::Reply send(const Json& body) const;

    /// The client's Messages body retargeted at this tier: its model name and
    /// output-token cap; everything else travels untouched.
    Json native_body(const Json& request) const;

    const std::string& chat_url() const { return chat_url_; }
    const std::string& messages_url() const { return messages_url_; }

    /// Authorization (if configured) for an upstream request.
    std::vector<http::HeaderEntry> auth_headers() const;

    /// Authorization plus the dialect headers for a native Messages request.
    std::vector<http::HeaderEntry> messages_headers(const MessagesHeaders& dialect) const;

    long timeout_ms() const { return timeout_ms_; }
    static constexpr long kConnectTimeoutMs = 15000;

    std::string model;
    std::optional<uint64_t> max_output_tokens;

private:
    std::string chat_url_;
    std::string messages_url_;
    std::optional<std::string> api_key_;    long timeout_ms_ = 0;  // whole-request limit, 0 = wait forever
};

}  // namespace s1::upstream

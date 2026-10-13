// `s1.toml` loading, `{env:VAR}` expansion and validation.
//
// Both tiers and the decider must be fully configured; the proxy refuses to
// start otherwise. Port of src/config.rs.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace s1::config {

/// What to do with a request when the decider cannot produce a decision.
enum class OnError { Top, Flash, Fail, Ha };

inline const char* on_error_str(OnError value) {
    switch (value) {
        case OnError::Top:
            return "top";
        case OnError::Flash:
            return "flash";
        case OnError::Fail:
            return "fail";
        case OnError::Ha:
            return "ha";
    }
    return "top";
}

struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 1970;
};

struct TierConfig {
    /// OpenAI-compatible base URL, without a trailing slash (`http://host:8000/v1`).
    std::string base_url;
    std::optional<std::string> api_key;
    std::string model;
    /// Whole-request timeout in seconds; nullopt means wait forever.
    std::optional<uint64_t> timeout_secs;
    /// Clamp for the client's requested output tokens.
    std::optional<uint64_t> max_output_tokens;
};

struct DeciderConfig {
    std::string url;
    std::optional<std::string> api_key;
    std::string model = "clef-flash";
    uint64_t timeout_ms = 3000;
    /// Route to the top tier when p(top) >= threshold.
    double threshold = 0.5;
    OnError on_error = OnError::Top;
    /// How long a decision is reused for follow-up steps of the same user
    /// turn; 0 disables.
    uint64_t sticky_ttl_secs = 900;
};

struct Config {
    ServerConfig server;
    TierConfig top;
    TierConfig flash;
    DeciderConfig decider;
};

struct ConfigError {
    enum class Kind { Read, Parse, Invalid };
    Kind kind = Kind::Invalid;
    /// Rendered message, the Display text of the Rust ConfigError.
    std::string message;
    /// The individual problems, filled for Kind::Invalid.
    std::vector<std::string> problems;
};

struct ConfigResult {
    bool ok = false;
    Config value{};
    ConfigError error{};

    explicit operator bool() const { return ok; }
};

/// Resolves `{env:VAR}` references; returns nullopt for unset variables.
using EnvFn = std::function<std::optional<std::string>(const std::string&)>;

/// Read a file and parse it.
ConfigResult load(const std::string& path);
/// Parse and validate; `env` resolves `{env:VAR}` references.
ConfigResult parse(const std::string& text, const EnvFn& env);

/// Replace every `{env:VAR}` in `value`; nullopt carries the first unset
/// variable's name.
std::optional<std::string> expand_env(const std::string& value, const EnvFn& env,
                                      std::string& missing_var);

}  // namespace s1::config

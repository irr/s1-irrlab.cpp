// The routing decision: build a compact description of the conversation, ask
// the System One decider (POST /v1/decisions) which tier should answer, and
// remember the answer for follow-up steps of the same user turn.
// Port of src/decider.rs.

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.hpp"
#include "json.hpp"
#include "upstream.hpp"

namespace s1::decider {

enum class Role { User, Assistant, Tool };

struct Turn {
    Role role;
    std::string text;
};

/// Protocol-neutral view of a request, built by the OpenAI and Anthropic front ends.
struct Conversation {
    std::string system;
    std::vector<Turn> turns;
    size_t tools = 0;
    bool has_images = false;

    void push(Role role, std::string text) { turns.push_back({role, std::move(text)}); }

    /// The `state` handed to the decider.
    Json state() const;

    /// Identifies one user turn of one conversation, so every step of an
    /// agent's tool loop maps to the same decision.
    uint64_t sticky_key() const;

private:
    // Index of the last turn the user actually wrote (not a tool result).
    std::optional<size_t> latest_user() const;
};

enum class Source { Decider, Cache, Fallback };

inline const char* source_str(Source source) {
    switch (source) {
        case Source::Decider: return "decider";
        case Source::Cache: return "cache";
        case Source::Fallback: return "fallback";
    }
    return "decider";
}

struct Decision {
    upstream::Tier tier = upstream::Tier::Top;
    /// Probability the decider gave the top tier; absent on fallback.
    std::optional<double> p_top;
    Source source = Source::Decider;
    std::chrono::milliseconds elapsed{0};
    /// Set when the decided tier's model failed and the other tier answered.
    std::optional<upstream::Tier> failed_over_from;

    /// Value of the `x-s1-decision` response header.
    std::string header() const;
};

struct DeciderError {
    enum class Kind { Http, Status, Answer };
    Kind kind = Kind::Http;
    std::string message;  // the Display text

    explicit operator bool() const { return true; }
};

struct DecideResult {
    bool ok = false;
    Decision decision{};
    DeciderError error{};
};

/// p(top) from a /v1/decisions response, falling back to the hard `choice`.
std::optional<double> p_top(const Json& answer);

class Decider {
public:
    explicit Decider(config::DeciderConfig cfg);

    /// Pick a tier. Only fails when on_error = "fail" and the decider is
    /// unusable.
    DecideResult decide(const Conversation& conv);

private:
    std::optional<double> ask(const Conversation& conv, DeciderError& error);

    std::optional<std::pair<upstream::Tier, double>> cached(uint64_t key) const;
    void store(uint64_t key, upstream::Tier tier, double p);

    config::DeciderConfig cfg_;
    mutable std::mutex mu_;

    struct Entry {
        upstream::Tier tier;
        double p;
        std::chrono::steady_clock::time_point at;
    };
    std::unordered_map<uint64_t, Entry> cache_;
};

}  // namespace s1::decider

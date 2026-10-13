#include "decider.hpp"

#include <cstdio>

#include "log.hpp"
#include "utf8.hpp"

namespace s1::decider {

namespace {

const char* const kQuestionId = "tier";
const char* const kInstructions =
    "A coding agent is working in a software project. Which model tier "
    "should write its next turn for the latest user request?";
const char* const kFlashCriteria =
    "Routine coding work a small fast model handles reliably: reading, "
    "searching or listing files, running a command and reporting its output, a single-file or "
    "few-line edit with clear instructions, renames, formatting, lint or typo fixes, "
    "boilerplate, simple tests, commit messages, short explanations of code, summaries, "
    "titles, acknowledgements and other mechanical follow-ups";
const char* const kTopCriteria =
    "Demanding software engineering that needs the strongest model: planning or "
    "implementing a feature across several files, architecture or API design, debugging a "
    "failure whose root cause is unknown, large refactors or migrations, concurrency, "
    "performance or security work, code review, long autonomous multi-step tasks with many "
    "tool calls, and vague or underspecified requirements";

// Character budgets keeping the state far below the decider's 16384-token cap.
constexpr size_t kRequestHead = 4000;
constexpr size_t kRequestTail = 2000;
constexpr size_t kContextTurns = 6;
constexpr size_t kContextChars = 500;
constexpr size_t kSystemChars = 800;
constexpr size_t kCachePruneAt = 1024;

const char* role_name(Role role) {
    switch (role) {
        case Role::User:
            return "user";
        case Role::Assistant:
            return "assistant";
        case Role::Tool:
            return "tool";
    }
    return "user";
}

/// Keep `head` leading and `tail` trailing characters, eliding the middle.
std::string clip(const std::string& text, size_t head, size_t tail) {
    size_t total = utf8::length(text);
    if (total <= head + tail) return text;
    std::string start = utf8::take(text, head);
    std::string end = utf8::skip(text, total - tail);
    return start + " \xe2\x80\xa6 " + end;
}

/// Agents inject <system-reminder> blocks into user turns; they are not part
/// of what the user asked, so they should not drive the decision.
std::string strip_reminders(const std::string& text) {
    const std::string open = "<system-reminder>";
    const std::string close = "</system-reminder>";
    std::string out;
    std::string rest = text;
    while (true) {
        size_t start = rest.find(open);
        if (start == std::string::npos) break;
        size_t len = rest.find(close, start);
        if (len == std::string::npos) break;
        out += rest.substr(0, start);
        rest = rest.substr(len + close.size());
    }
    out += rest;
    if (utf8::trimmed_empty(out)) return text;
    return out;
}

std::string format_p(double value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", value);
    return buf;
}

}  // namespace

std::optional<size_t> Conversation::latest_user() const {
    for (size_t i = turns.size(); i-- > 0;) {
        if (turns[i].role == Role::User && !utf8::trimmed_empty(turns[i].text)) return i;
    }
    return std::nullopt;
}

Json Conversation::state() const {
    auto latest = latest_user();
    std::string request = latest ? strip_reminders(turns[*latest].text) : "";

    Json context = Json::array();
    for (size_t i = 0; i < turns.size(); ++i) {
        if (latest && i == *latest) continue;
        if (utf8::trimmed_empty(turns[i].text)) continue;
        context.push_back(std::string(role_name(turns[i].role)) + ": " +
                          clip(utf8::trim_copy(strip_reminders(turns[i].text)), kContextChars, 0));
    }
    if (context.size() > kContextTurns) {
        Json last = Json::array();
        for (size_t i = context.size() - kContextTurns; i < context.size(); ++i) {
            last.push_back(context[i]);
        }
        context = std::move(last);
    }

    Json state =
        Json{{"latest_user_request", clip(utf8::trim_copy(request), kRequestHead, kRequestTail)},
             {"recent_context", context},
             {"system_prompt_excerpt", clip(utf8::trim_copy(system), kSystemChars, 0)},
             {"turns", turns.size()},
             {"tools_available", tools},
             {"has_images", has_images},
             {"last_turn_is_tool_result", !turns.empty() && turns.back().role == Role::Tool}};
    return state;
}

uint64_t Conversation::sticky_key() const {
    // FNV-1a over a canonical encoding; only within-process stability matters
    // (the Rust version used the std DefaultHasher the same way).
    const std::string* first_user = nullptr;
    for (const auto& turn : turns) {
        if (turn.role == Role::User) {
            first_user = &turn.text;
            break;
        }
    }
    auto latest = latest_user();
    std::string encoded;
    encoded += system;
    encoded += '\x1f';
    encoded += first_user ? *first_user : std::string(1, '\x01');
    encoded += '\x1f';
    encoded += latest ? std::to_string(*latest) : std::string(1, '\x01');
    encoded += '\x1f';
    encoded += latest ? turns[*latest].text : std::string(1, '\x01');

    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : encoded) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string Decision::header() const {
    std::string header;
    if (p_top) {
        header = "p=" + format_p(*p_top) + ";src=" + source_str(source);
    } else {
        header = std::string("src=") + source_str(source);
    }
    if (failed_over_from) {
        header += std::string(";failover_from=") + upstream::tier_str(*failed_over_from);
    }
    return header;
}

std::optional<double> p_top(const Json& answer) {
    const Json* tier = json_ptr(answer, "/answers/" + std::string(kQuestionId));
    if (!tier) return std::nullopt;
    if (auto p = get_f64_at(*tier, "/probabilities/top")) return p;
    auto choice = get_str(*tier, "choice");
    if (!choice) return std::nullopt;
    if (*choice == "top") return 1.0;
    if (*choice == "flash") return 0.0;
    return std::nullopt;
}

Decider::Decider(config::DeciderConfig cfg) : cfg_(std::move(cfg)) {}

DecideResult Decider::decide(const Conversation& conv) {
    auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started);
    };

    DecideResult result;
    uint64_t key = conv.sticky_key();
    if (auto hit = cached(key)) {
        result.ok = true;
        result.decision = Decision{hit->first, hit->second, Source::Cache, elapsed(), std::nullopt};
        return result;
    }

    DeciderError error;
    if (auto p = ask(conv, error)) {
        upstream::Tier tier = *p >= cfg_.threshold ? upstream::Tier::Top : upstream::Tier::Flash;
        store(key, tier, *p);
        result.ok = true;
        result.decision = Decision{tier, *p, Source::Decider, elapsed(), std::nullopt};
        return result;
    }

    upstream::Tier tier = upstream::Tier::Top;
    switch (cfg_.on_error) {
        case config::OnError::Fail:
            result.error = std::move(error);
            return result;
        case config::OnError::Top:
        case config::OnError::Ha:
            tier = upstream::Tier::Top;
            break;
        case config::OnError::Flash:
            tier = upstream::Tier::Flash;
            break;
    }
    S1_LOG(log::Level::Warn,
           "decider unavailable error=" + error.message + " fallback=" + upstream::tier_str(tier));
    result.ok = true;
    result.decision = Decision{tier, std::nullopt, Source::Fallback, elapsed(), std::nullopt};
    return result;
}

std::optional<double> Decider::ask(const Conversation& conv, DeciderError& error) {
    Json body =
        Json{{"model", cfg_.model},
             {"state", conv.state()},
             {"questions", Json{{kQuestionId, Json{{"type", "choice"},
                                                   {"instructions", kInstructions},
                                                   {"criteria", Json{{"flash", kFlashCriteria},
                                                                     {"top", kTopCriteria}}}}}}}};
    if (log::enabled(log::Level::Debug)) {
        S1_LOG(log::Level::Debug, "asking decider state=" + body["state"].dump());
    }

    std::vector<http::HeaderEntry> headers;
    if (cfg_.api_key) {
        headers.push_back({"Authorization", "Bearer " + *cfg_.api_key});
    }
    http::Reply reply =
        http::post(cfg_.url, body.dump(), headers, static_cast<long>(cfg_.timeout_ms), 0);

    if (!reply.ok) {
        error = DeciderError{DeciderError::Kind::Http, "decider request failed: " + reply.error};
        return std::nullopt;
    }
    if (reply.status < 200 || reply.status > 299) {
        error = DeciderError{DeciderError::Kind::Status, "decider returned HTTP " +
                                                             std::to_string(reply.status) + ": " +
                                                             utf8::take(reply.body, 500)};
        return std::nullopt;
    }
    Json answer = Json::parse(reply.body, nullptr, false);
    if (answer.is_discarded()) {
        error = DeciderError{DeciderError::Kind::Http,
                             "decider request failed: error decoding response body"};
        return std::nullopt;
    }
    if (log::enabled(log::Level::Debug)) {
        S1_LOG(log::Level::Debug, "decider answered answers=" + answer["answers"].dump());
    }
    auto p = p_top(answer);
    if (!p) {
        error = DeciderError{DeciderError::Kind::Answer,
                             "decider response has no usable `tier` answer: " + answer.dump()};
    }
    return p;
}

std::optional<std::pair<upstream::Tier, double>> Decider::cached(uint64_t key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return std::nullopt;
    auto ttl = std::chrono::seconds(cfg_.sticky_ttl_secs);
    if (std::chrono::steady_clock::now() - it->second.at >= ttl) return std::nullopt;
    return std::make_pair(it->second.tier, it->second.p);
}

void Decider::store(uint64_t key, upstream::Tier tier, double p) {
    if (cfg_.sticky_ttl_secs == 0) return;
    auto ttl = std::chrono::seconds(cfg_.sticky_ttl_secs);
    std::lock_guard<std::mutex> lock(mu_);
    if (cache_.size() >= kCachePruneAt) {
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (std::chrono::steady_clock::now() - it->second.at >= ttl) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }
    cache_[key] = Entry{tier, p, std::chrono::steady_clock::now()};
}

}  // namespace s1::decider

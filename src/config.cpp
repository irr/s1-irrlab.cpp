#include "config.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>
#include <toml.hpp>

#include "utf8.hpp"

namespace s1::config {

namespace {

const char* const kPlaceholder = "REPLACE_ME";

// Abort parsing with a serde-style message (the Rust version surfaced these
// as toml::de::Error, i.e. Kind::Parse).
struct ParseAbort {
    std::string message;
};

[[noreturn]] void fail(std::string message) { throw ParseAbort{std::move(message)}; }

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// http(s) URL with a host, the check reqwest::Url::parse applied.
bool valid_http_url(const std::string& value) {
    size_t sep = value.find("://");
    if (sep == std::string::npos) return false;
    std::string scheme = value.substr(0, sep);
    for (char& c : scheme) {
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    if (scheme != "http" && scheme != "https") return false;
    std::string rest = value.substr(sep + 3);
    size_t end = rest.find_first_of("/?#");
    std::string authority = rest.substr(0, end == std::string::npos ? rest.size() : end);
    size_t at = authority.rfind('@');
    std::string hostport = at == std::string::npos ? authority : authority.substr(at + 1);
    std::string host = hostport;
    if (!host.empty() && host.front() == '[') {
        size_t close = host.find(']');
        if (close == std::string::npos) return false;
        host = host.substr(1, close - 1);
    } else {
        size_t colon = host.find(':');
        if (colon != std::string::npos) host = host.substr(0, colon);
    }
    return !host.empty();
}

std::string strip_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}

std::string quote(const std::string& value) { return "\"" + value + "\""; }

std::string format_double(double value) {
    // Rust `{}` for f64: shortest round-trip form, no trailing zeros.
    char buf[40];
    auto result = std::to_chars(buf, buf + sizeof(buf), value);
    return std::string(buf, result.ptr);
}

// --- toml++ field extraction with serde-style type errors ---

const toml::table* section_of(const toml::table& root, const char* name) {
    auto it = root.find(name);
    if (it == root.end()) return nullptr;
    if (const auto* table = it->second.as_table()) return table;
    fail("invalid type for key `" + std::string(name) + "`, expected a table");
}

void check_keys(const toml::table& table, const std::vector<std::string>& known,
                const std::string& where) {
    for (const auto& [key, node] : table) {
        if (std::find(known.begin(), known.end(), std::string(key.str())) == known.end()) {
            std::string expected;
            for (size_t i = 0; i < known.size(); ++i) {
                expected += i > 0 ? ", " : "";
                expected += "`" + known[i] + "`";
            }
            std::string what = where.empty() ? "config" : "[" + where + "]";
            fail("unknown field `" + std::string(key.str()) + "` in " + what +
                 ", expected one of " + expected);
        }
    }
}

std::optional<std::string> str_field(const toml::table& table, const char* key,
                                     const std::string& where) {
    auto it = table.find(key);
    if (it == table.end()) return std::nullopt;
    if (const auto* s = it->second.as_string()) return s->get();
    fail("invalid type for key `" + where + "." + key + "`, expected a string");
}

std::optional<uint64_t> uint_field(const toml::table& table, const char* key,
                                   const std::string& where) {
    auto it = table.find(key);
    if (it == table.end()) return std::nullopt;
    if (const auto* i = it->second.as_integer()) {
        int64_t value = i->get();
        if (value < 0) {
            fail("invalid type for key `" + where + "." + key + "`, expected an unsigned integer");
        }
        return static_cast<uint64_t>(value);
    }
    fail("invalid type for key `" + where + "." + key + "`, expected an unsigned integer");
}

std::optional<double> double_field(const toml::table& table, const char* key,
                                   const std::string& where) {
    auto it = table.find(key);
    if (it == table.end()) return std::nullopt;
    if (const auto* f = it->second.as_floating_point()) return f->get();
    if (const auto* i = it->second.as_integer()) return static_cast<double>(i->get());
    fail("invalid type for key `" + where + "." + key + "`, expected a number");
}

// --- validation, mirroring the Rust Validator ---

struct Validator {
    const EnvFn& env;
    std::vector<std::string> problems;

    // A required string: present, env-expanded, non-empty and not a placeholder.
    std::optional<std::string> required(const std::string& key, std::optional<std::string> value) {
        if (!value) {
            problems.push_back(key + " is required");
            return std::nullopt;
        }
        auto expanded = expand(key, *value);
        if (!expanded) return std::nullopt;
        if (utf8::trimmed_empty(*expanded) || contains(*expanded, kPlaceholder)) {
            problems.push_back(key + " is not set (found " + quote(*expanded) + ")");
            return std::nullopt;
        }
        return expanded;
    }

    std::optional<std::string> optional(const std::string& key, std::optional<std::string> value) {
        if (!value) return std::nullopt;
        auto expanded = expand(key, *value);
        if (!expanded) return std::nullopt;
        if (utf8::trimmed_empty(*expanded)) return std::nullopt;
        return expanded;
    }

    std::optional<std::string> expand(const std::string& key, const std::string& value) {
        std::string missing;
        auto expanded = expand_env(value, env, missing);
        if (expanded) return expanded;
        problems.push_back(key + " references environment variable " + missing +
                           ", which is not set");
        return std::nullopt;
    }

    std::optional<std::string> url(const std::string& key, std::optional<std::string> value) {
        auto resolved = required(key, std::move(value));
        if (!resolved) return std::nullopt;
        if (!valid_http_url(*resolved)) {
            problems.push_back(key + " is not a valid http(s) URL: " + quote(*resolved));
            return std::nullopt;
        }
        return strip_trailing_slashes(*resolved);
    }

    std::optional<TierConfig> tier(const std::string& name, const toml::table* raw) {
        if (!raw) {
            problems.push_back("[" + name + "] section is required (base_url, model)");
            return std::nullopt;
        }
        check_keys(*raw, {"base_url", "api_key", "model", "timeout_secs", "max_output_tokens"},
                   name);
        auto base_url = url(name + ".base_url", str_field(*raw, "base_url", name));
        auto model = required(name + ".model", str_field(*raw, "model", name));
        auto api_key = optional(name + ".api_key", str_field(*raw, "api_key", name));
        auto timeout_secs = uint_field(*raw, "timeout_secs", name);
        auto max_output_tokens = uint_field(*raw, "max_output_tokens", name);
        if (max_output_tokens && *max_output_tokens == 0) {
            problems.push_back(name + ".max_output_tokens must be greater than 0");
        }
        if (!base_url || !model) return std::nullopt;
        TierConfig cfg;
        cfg.base_url = *base_url;
        cfg.api_key = std::move(api_key);
        cfg.model = *model;
        if (timeout_secs && *timeout_secs > 0) cfg.timeout_secs = timeout_secs;
        cfg.max_output_tokens = max_output_tokens;
        return cfg;
    }

    std::optional<DeciderConfig> decider(const toml::table* raw) {
        if (!raw) {
            problems.push_back("[decider] section is required (url)");
            return std::nullopt;
        }
        check_keys(
            *raw,
            {"url", "api_key", "model", "timeout_ms", "threshold", "on_error", "sticky_ttl_secs"},
            "decider");
        auto url = this->url("decider.url", str_field(*raw, "url", "decider"));
        auto api_key = optional("decider.api_key", str_field(*raw, "api_key", "decider"));
        std::optional<std::string> model;
        if (auto provided = str_field(*raw, "model", "decider")) {
            model = required("decider.model", std::move(provided));
        } else {
            model = std::string("clef-flash");
        }
        double threshold = 0.5;
        if (auto value = double_field(*raw, "threshold", "decider")) threshold = *value;
        if (!(threshold >= 0.0 && threshold <= 1.0)) {
            problems.push_back("decider.threshold must be within 0..=1, got " +
                               format_double(threshold));
        }
        uint64_t timeout_ms = 3000;
        if (auto value = uint_field(*raw, "timeout_ms", "decider")) timeout_ms = *value;
        if (timeout_ms == 0) {
            problems.push_back("decider.timeout_ms must be greater than 0");
        }
        OnError on_error = OnError::Top;
        if (auto value = str_field(*raw, "on_error", "decider")) {
            if (*value == "top")
                on_error = OnError::Top;
            else if (*value == "flash")
                on_error = OnError::Flash;
            else if (*value == "fail")
                on_error = OnError::Fail;
            else if (*value == "ha")
                on_error = OnError::Ha;
            else
                fail("unknown variant `" + *value + "`, expected one of `top`, " +
                     "`flash`, `fail`, `ha` for key `decider.on_error`");
        }
        uint64_t sticky_ttl_secs = 900;
        if (auto value = uint_field(*raw, "sticky_ttl_secs", "decider")) {
            sticky_ttl_secs = *value;
        }

        if (!url || !model) return std::nullopt;
        DeciderConfig cfg;
        cfg.url = *url;
        cfg.api_key = std::move(api_key);
        cfg.model = *model;
        cfg.timeout_ms = timeout_ms;
        cfg.threshold = threshold;
        cfg.on_error = on_error;
        cfg.sticky_ttl_secs = sticky_ttl_secs;
        return cfg;
    }
};

}  // namespace

std::optional<std::string> expand_env(const std::string& value, const EnvFn& env,
                                      std::string& missing_var) {
    const std::string open = "{env:";
    std::string out;
    out.reserve(value.size());
    std::string rest = value;
    while (true) {
        size_t start = rest.find(open);
        if (start == std::string::npos) break;
        std::string after = rest.substr(start + open.size());
        size_t end = after.find('}');
        if (end == std::string::npos) break;
        std::string name = utf8::trim_copy(after.substr(0, end));
        auto resolved = env(name);
        if (!resolved) {
            missing_var = name;
            return std::nullopt;
        }
        out += rest.substr(0, start);
        out += *resolved;
        rest = after.substr(end + 1);
    }
    out += rest;
    return out;
}

ConfigResult load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        int err = errno;
        ConfigResult result;
        result.error.kind = ConfigError::Kind::Read;
        result.error.message = "cannot read config " + path + ": " +
                               std::system_category().message(err) + " (os error " +
                               std::to_string(err) + ")";
        return result;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    EnvFn env = [](const std::string& name) -> std::optional<std::string> {
        if (const char* value = std::getenv(name.c_str())) return std::string(value);
        return std::nullopt;
    };
    return parse(buffer.str(), env);
}

ConfigResult parse(const std::string& text, const EnvFn& env) {
    ConfigResult result;
    try {
        toml::table root;
        try {
            root = toml::parse(text);
        } catch (const toml::parse_error& e) {
            fail(std::string(e.description()));
        }
        check_keys(root, {"server", "top", "flash", "decider"}, "");

        Validator validator{env, {}};
        Config cfg;

        if (const auto* server = section_of(root, "server")) {
            check_keys(*server, {"host", "port"}, "server");
            if (auto host = str_field(*server, "host", "server")) cfg.server.host = *host;
            if (auto port = uint_field(*server, "port", "server")) {
                if (*port > 65535) {
                    fail("invalid value for key `server.port`, expected u16");
                }
                cfg.server.port = static_cast<uint16_t>(*port);
            }
        }

        auto top = validator.tier("top", section_of(root, "top"));
        auto flash = validator.tier("flash", section_of(root, "flash"));
        auto decider = validator.decider(section_of(root, "decider"));

        if (top && flash && decider && validator.problems.empty()) {
            cfg.top = *top;
            cfg.flash = *flash;
            cfg.decider = *decider;
            result.ok = true;
            result.value = std::move(cfg);
        } else {
            result.error.kind = ConfigError::Kind::Invalid;
            result.error.problems = std::move(validator.problems);
            std::string message = "invalid config:";
            for (const auto& problem : result.error.problems) {
                message += "\n  - " + problem;
            }
            result.error.message = std::move(message);
        }
        return result;
    } catch (const ParseAbort& abort) {
        result.error.kind = ConfigError::Kind::Parse;
        result.error.message = "invalid TOML in config: " + abort.message;
        return result;
    }
}

}  // namespace s1::config

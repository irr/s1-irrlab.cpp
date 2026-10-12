// JSON helpers. `nlohmann::ordered_json` stands in for serde_json with the
// `preserve_order` feature: object keys keep insertion order, which the
// passthrough paths rely on when they re-serialize client bodies.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace s1 {

using Json = nlohmann::ordered_json;

// serde_json's `Value::pointer("/a/b")`: the addressed value, or null when the
// path does not exist. The pointer string may be empty for the value itself.
const Json* json_ptr(const Json& value, const std::string& pointer);
Json* json_ptr(Json& value, const std::string& pointer);

// Accessors that only succeed when the JSON type matches, mirroring serde's
// `as_str` / `as_u64` / `as_f64` (integers are not strings, floats are not
// u64s, and so on).
std::optional<std::string> as_str(const Json& value);
std::optional<std::string> get_str(const Json& value, const char* key);
std::optional<std::string> get_str_at(const Json& value, const std::string& pointer);
std::optional<uint64_t> as_u64(const Json& value);
std::optional<uint64_t> get_u64(const Json& value, const char* key);
std::optional<bool> get_bool(const Json& value, const char* key);
std::optional<double> as_f64(const Json& value);
std::optional<double> get_f64_at(const Json& value, const std::string& pointer);

inline const Json* json_ptr(const Json& value, const std::string& pointer) {
    try {
        return &value.at(Json::json_pointer(pointer));
    } catch (const std::exception&) {
        return nullptr;
    }
}

inline Json* json_ptr(Json& value, const std::string& pointer) {
    try {
        return &value.at(Json::json_pointer(pointer));
    } catch (const std::exception&) {
        return nullptr;
    }
}

inline std::optional<std::string> as_str(const Json& value) {
    if (value.is_string()) return value.get<std::string>();
    return std::nullopt;
}

inline std::optional<std::string> get_str(const Json& value, const char* key) {
    if (!value.is_object() || !value.contains(key)) return std::nullopt;
    return as_str(value.at(key));
}

inline std::optional<std::string> get_str_at(const Json& value, const std::string& pointer) {
    const Json* found = json_ptr(value, pointer);
    if (!found) return std::nullopt;
    return as_str(*found);
}

inline std::optional<uint64_t> as_u64(const Json& value) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    return std::nullopt;
}

inline std::optional<uint64_t> get_u64(const Json& value, const char* key) {
    if (!value.is_object() || !value.contains(key)) return std::nullopt;
    return as_u64(value.at(key));
}

inline std::optional<bool> get_bool(const Json& value, const char* key) {
    if (!value.is_object() || !value.contains(key) || !value.at(key).is_boolean()) {
        return std::nullopt;
    }
    return value.at(key).get<bool>();
}

inline std::optional<double> as_f64(const Json& value) {
    if (value.is_number()) return value.get<double>();
    return std::nullopt;
}

inline std::optional<double> get_f64_at(const Json& value, const std::string& pointer) {
    const Json* found = json_ptr(value, pointer);
    if (!found) return std::nullopt;
    return as_f64(*found);
}

}  // namespace s1

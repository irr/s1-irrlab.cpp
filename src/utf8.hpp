// UTF-8 helpers. The Rust code counts and clips `char`s (Unicode code points),
// not bytes, so the same semantics are needed wherever lengths matter: the
// decider state clipping, error message caps and the token estimate.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace s1::utf8 {

// Number of code points (Rust `str::chars().count()`); invalid bytes count as
// one code point each.
size_t length(const std::string& text);

// First `n` code points (Rust `chars().take(n)`).
std::string take(const std::string& text, size_t n);

// All but the first `n` code points (Rust `chars().skip(n)`).
std::string skip(const std::string& text, size_t n);

// Rust `str::trim()`: strip ASCII whitespace from both ends.
std::string_view trim(std::string_view text);
bool trimmed_empty(std::string_view text);
std::string trim_copy(std::string_view text);

inline bool is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Length in bytes of the code point starting with `lead`.
inline size_t cp_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;  // stray continuation byte
}

}  // namespace s1::utf8

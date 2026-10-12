#include "utf8.hpp"

namespace s1::utf8 {

size_t length(const std::string& text) {
    size_t count = 0;
    for (size_t i = 0; i < text.size();) {
        i += cp_len(static_cast<unsigned char>(text[i]));
        ++count;
    }
    return count;
}

std::string take(const std::string& text, size_t n) {
    size_t i = 0;
    while (i < text.size() && n > 0) {
        i += cp_len(static_cast<unsigned char>(text[i]));
        --n;
    }
    if (i > text.size()) i = text.size();  // truncated multi-byte sequence
    return text.substr(0, i);
}

std::string skip(const std::string& text, size_t n) {
    size_t i = 0;
    while (i < text.size() && n > 0) {
        i += cp_len(static_cast<unsigned char>(text[i]));
        --n;
    }
    if (i > text.size()) i = text.size();
    return text.substr(i);
}

std::string_view trim(std::string_view text) {
    size_t begin = 0, end = text.size();
    while (begin < end && is_space(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && is_space(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

bool trimmed_empty(std::string_view text) { return trim(text).empty(); }

std::string trim_copy(std::string_view text) { return std::string(trim(text)); }

}  // namespace s1::utf8

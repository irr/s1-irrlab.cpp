#include "log.hpp"

#include <sys/time.h>

#include <atomic>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace s1::log {

namespace {

std::atomic<int> g_level{static_cast<int>(Level::Info)};
std::mutex g_mutex;

const char* kLevelNames[] = {"DEBUG", " INFO", " WARN", "ERROR"};

}  // namespace

void init_from_env() {
    const char* env = std::getenv("S1_LOG_LEVEL");
    if (!env || !*env) return;
    std::string value = env;
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    if (value == "debug" || value == "trace") {
        g_level = static_cast<int>(Level::Debug);
    } else if (value == "info") {
        g_level = static_cast<int>(Level::Info);
    } else if (value == "warn" || value == "warning") {
        g_level = static_cast<int>(Level::Warn);
    } else if (value == "error") {
        g_level = static_cast<int>(Level::Error);
    }
}

bool enabled(Level level) { return static_cast<int>(level) >= g_level.load(); }

const char* level_name(Level level) { return kLevelNames[static_cast<int>(level)]; }

void write(Level level, const std::string& message) {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    std::tm tm{};
    gmtime_r(&tv.tv_sec, &tm);
    char stamp[64];
    std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02dT%02d:%02d:%02d.%06dZ", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(tv.tv_usec));
    std::lock_guard<std::mutex> lock(g_mutex);
    std::fprintf(stdout, "%s %s s1: %s\n", stamp, level_name(level), message.c_str());
    std::fflush(stdout);
}

}  // namespace s1::log

// Logging: one line per event on stdout, in the shape tracing-subscriber
// produces for the Rust original. The level comes from S1_LOG_LEVEL
// (debug | info | warn | error); the default is info, which was the Rust
// default for RUST_LOG.

#pragma once

#include <string>

namespace s1::log {

enum class Level : int { Debug = 0, Info = 1, Warn = 2, Error = 3 };

// Reads S1_LOG_LEVEL once; safe to call repeatedly.
void init_from_env();

bool enabled(Level level);
void write(Level level, const std::string& message);
const char* level_name(Level level);

}  // namespace s1::log

#define S1_LOG(lvl, msg)                                   \
    do {                                                   \
        if (::s1::log::enabled(lvl)) {                     \
            ::s1::log::write(lvl, msg);                    \
        }                                                  \
    } while (0)

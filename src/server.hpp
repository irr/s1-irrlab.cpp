// HTTP surface: decide first, then forward to the chosen tier. Port of
// src/server.rs; the routes are installed on a cpp-httplib server that the
// caller binds and listens (main, tests).

#pragma once

#include <memory>

#include <httplib.h>

#include "config.hpp"

namespace s1::server {

struct State;

/// Routes + shared state for one proxy instance. Copyable handle around the
/// shared state; install() may be called once per httplib::Server.
class Router {
public:
    explicit Router(config::Config cfg);
    ~Router();
    Router(const Router&) = delete;
    Router& operator=(const Router&) = delete;

    void install(httplib::Server& svr);

private:
    std::shared_ptr<State> state_;
};

}  // namespace s1::server

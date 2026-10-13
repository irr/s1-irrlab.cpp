// s1 entry point: CLI, config load, logging, listen. Port of src/main.rs.

#include <httplib.h>

#include <charconv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cxxopts.hpp>
#include <string>

#include "config.hpp"
#include "log.hpp"
#include "server.hpp"

namespace {

httplib::Server* g_server = nullptr;

void on_signal(int) {
    if (g_server) g_server->stop();
}

std::string fmt_double(double value) {
    char buf[64];
    auto result = std::to_chars(buf, buf + sizeof(buf), value);
    return std::string(buf, result.ptr);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace s1;

    std::string config_path = "s1.toml";
    try {
        cxxopts::Options options(
            "s1", "Complexity-routing proxy for OpenAI- and Anthropic-compatible agents.");
        options.add_options()("c,config", "Path to the TOML configuration.",
                              cxxopts::value<std::string>())("h,help", "Show help")(
            "v,version", "Print version");
        auto parsed = options.parse(argc, argv);
        if (parsed.count("help") > 0) {
            std::printf("%s\n", options.help().c_str());
            return 0;
        }
        if (parsed.count("version") > 0) {
            std::printf("s1 0.1.0\n");
            return 0;
        }
        if (parsed.count("config") > 0) {
            config_path = parsed["config"].as<std::string>();
        } else if (const char* env = std::getenv("S1_CONFIG")) {
            config_path = env;
        }
    } catch (const std::exception& err) {
        std::fprintf(stderr, "Error: %s\n", err.what());
        return 1;
    }

    log::init_from_env();

    auto config = config::load(config_path);
    if (!config.ok) {
        std::fprintf(stderr, "Error: %s\n", config.error.message.c_str());
        return 1;
    }
    const auto& cfg = config.value;

    std::string addr = cfg.server.host + ":" + std::to_string(cfg.server.port);
    S1_LOG(log::Level::Info, "tier=top url=" + cfg.top.base_url + " model=" + cfg.top.model +
                                 " api_key=" + (cfg.top.api_key ? "true" : "false"));
    S1_LOG(log::Level::Info, "tier=flash url=" + cfg.flash.base_url + " model=" + cfg.flash.model +
                                 " api_key=" + (cfg.flash.api_key ? "true" : "false"));
    S1_LOG(log::Level::Info, "decider=" + cfg.decider.url +
                                 " threshold=" + fmt_double(cfg.decider.threshold) +
                                 " on_error=" + config::on_error_str(cfg.decider.on_error));

    httplib::Server svr;
    if (!svr.bind_to_port(cfg.server.host, cfg.server.port)) {
        std::fprintf(stderr, "Error: cannot listen on %s\n", addr.c_str());
        return 1;
    }
    // Constructed after the bind: the move empties config.value, so nothing
    // may read `cfg` (which aliases it) past this point.
    server::Router router(std::move(config.value));
    router.install(svr);
    S1_LOG(log::Level::Info, "listening on http://" + addr);

    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    svr.listen_after_bind();
    return 0;
}

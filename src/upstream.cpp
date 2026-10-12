#include "upstream.hpp"

#include <httplib.h>

namespace s1::upstream {

namespace {
const char* const kDefaultVersion = "2023-06-01";
}  // namespace

MessagesHeaders MessagesHeaders::extract(const httplib::Headers& client) {
    MessagesHeaders dialect;
    if (client.contains("anthropic-version")) {
        dialect.version = client.find("anthropic-version")->second;
    }
    if (client.contains("anthropic-beta")) {
        dialect.beta = client.find("anthropic-beta")->second;
    }
    return dialect;
}

void MessagesHeaders::apply(std::vector<http::HeaderEntry>& headers) const {
    headers.push_back({"anthropic-version", version.value_or(kDefaultVersion)});
    if (beta) headers.push_back({"anthropic-beta", *beta});
}

Upstream::Upstream(const config::TierConfig& cfg)
    : model(cfg.model),
      max_output_tokens(cfg.max_output_tokens),
      chat_url_(cfg.base_url + "/chat/completions"),
      messages_url_(cfg.base_url + "/messages"),
      api_key_(cfg.api_key) {
    if (cfg.timeout_secs) timeout_ms_ = static_cast<long>(*cfg.timeout_secs) * 1000;
}

std::vector<http::HeaderEntry> Upstream::auth_headers() const {
    std::vector<http::HeaderEntry> headers;
    if (api_key_) headers.push_back({"Authorization", "Bearer " + *api_key_});
    return headers;
}

std::vector<http::HeaderEntry> Upstream::messages_headers(
    const MessagesHeaders& dialect) const {
    auto headers = auth_headers();
    dialect.apply(headers);
    return headers;
}

http::Reply Upstream::send(const Json& body) const {
    return http::post(chat_url_, body.dump(), auth_headers(), timeout_ms_,
                      kConnectTimeoutMs);
}

Json Upstream::native_body(const Json& request) const {
    Json body = request;
    body["model"] = model;
    if (max_output_tokens) {
        if (auto requested = get_u64(body, "max_tokens")) {
            body["max_tokens"] = std::min(*requested, *max_output_tokens);
        }
    }
    return body;
}

}  // namespace s1::upstream

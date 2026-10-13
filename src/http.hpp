// The libcurl wrapper every outbound request goes through: blocking POST/GET
// helpers (decider, buffered tier replies) and StreamRequest, a streamed POST
// whose response head and body chunks are delivered while the transfer runs.
// Timeouts map directly onto the ones reqwest applied: a whole-request
// timeout and a connect timeout.

#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace s1::http {

struct HeaderEntry {
    std::string name;
    std::string value;
};

struct Reply {
    // A response arrived from the server (any status). False means the request
    // could not be completed: refused, timed out, broken mid-transfer.
    bool ok = false;
    long status = 0;
    std::string body;
    std::vector<HeaderEntry> headers;  // in arrival order, final response only
    std::string error;                 // curl's error text when !ok

    std::string header_of(const std::string& name) const;  // first value, case-insensitive
    bool has_header(const std::string& name) const;
};

// Blocking POST with a JSON body; reads the whole response.
// `timeout_ms` / `connect_timeout_ms` of 0 mean "no limit".
Reply post(const std::string& url, const std::string& body, const std::vector<HeaderEntry>& headers,
           long timeout_ms, long connect_timeout_ms);

// Blocking GET (health probes, tests).
Reply get(const std::string& url, long timeout_ms);

// --- streamed POST ---

// A POST transfer decoupled into a head phase (status + headers, used for
// routing and failover decisions before any byte is forwarded) and a body
// phase (chunks consumed by the response provider while the transfer runs).
// The transfer runs on its own thread; cancel() aborts it at any point, also
// unblocking a socket stuck mid-transfer.
class StreamRequest {
public:
    struct Head {
        bool ok = false;  // transport reached the server
        long status = 0;
        std::vector<HeaderEntry> headers;
        std::string error;  // when !ok
    };

    StreamRequest(const std::string& url, const std::string& body,
                  const std::vector<HeaderEntry>& headers, long timeout_ms,
                  long connect_timeout_ms);
    ~StreamRequest();
    StreamRequest(const StreamRequest&) = delete;
    StreamRequest& operator=(const StreamRequest&) = delete;

    // Blocks until the response head arrived or the request failed.
    Head wait_head();

    // Blocks up to `timeout` for the next body chunk. Returns true and fills
    // `chunk`; returns false when the body ended (`timed_out` = false) or the
    // wait elapsed (`timed_out` = true).
    bool next(std::string& chunk, std::chrono::milliseconds timeout, bool& timed_out);

    // Whether the body finished and every chunk was consumed.
    bool finished();

    // Whether the transfer completed without error or cancellation.
    bool body_ok();
    std::string error();

    // Abort the transfer and discard the rest of the body.
    void cancel();

    // Read the remaining body to completion and return it.
    std::string drain();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace s1::http

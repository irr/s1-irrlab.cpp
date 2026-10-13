#include "http.hpp"

#include <curl/curl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace s1::http {

namespace {

void global_init() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_ALL); });
}

struct Slist {
    struct curl_slist* raw = nullptr;
    ~Slist() { curl_slist_free_all(raw); }
    void append(const std::string& entry) { raw = curl_slist_append(raw, entry.c_str()); }
};

std::string ieq_lower(const std::string& name) {
    std::string lower = name;
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower;
}

// Parses one raw header line. Returns false for a blank line (end of a header
// block); manages the pending status/headers of the current response.
struct HeaderBlock {
    long status = 0;
    bool got_status = false;
    std::vector<HeaderEntry> headers;

    // One header line (without the CRLF). True when it belongs to the block
    // (i.e. not the terminating blank line).
    bool line(const char* data, size_t len) {
        std::string text(data, len);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        if (text.empty()) return false;
        if (text.rfind("HTTP/", 0) == 0) {
            // A status line: another header block begins (redirect or 1xx).
            // Only real (2xx+) statuses become the response's status.
            headers.clear();
            size_t space = text.find(' ');
            if (space != std::string::npos) {
                long code = std::strtol(text.c_str() + space + 1, nullptr, 10);
                if (code >= 200) {
                    status = code;
                    got_status = true;
                }
            }
            return true;
        }
        size_t colon = text.find(':');
        if (colon == std::string::npos) return true;
        auto trim = [](std::string& s) {
            size_t b = s.find_first_not_of(" \t");
            size_t e = s.find_last_not_of(" \t");
            s = b == std::string::npos ? "" : s.substr(b, e - b + 1);
        };
        std::string name = text.substr(0, colon);
        std::string value = text.substr(colon + 1);
        trim(name);
        trim(value);
        if (!name.empty()) headers.push_back({name, value});
        return true;
    }
};

// --- blocking helpers ---

struct BufferedTransfer {
    HeaderBlock head;
    std::string body;

    static size_t on_header(char* data, size_t size, size_t count, void* self_) {
        auto* self = static_cast<BufferedTransfer*>(self_);
        self->head.line(data, size * count);
        return size * count;
    }

    static size_t on_write(char* data, size_t size, size_t count, void* self_) {
        auto* self = static_cast<BufferedTransfer*>(self_);
        self->body.append(data, size * count);
        return size * count;
    }
};

bool setup_common(CURL* easy, const std::string& url, const std::string* body,
                  const std::vector<HeaderEntry>& headers, long timeout_ms, long connect_timeout_ms,
                  Slist& list, char* error_buffer) {
    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(easy, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(easy, CURLOPT_ERRORBUFFER, error_buffer);
    if (timeout_ms > 0) curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, timeout_ms);
    if (connect_timeout_ms > 0) {
        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
    }
    // reqwest does not send "Expect: 100-continue"; suppress curl's.
    list.append("Expect:");
    for (const auto& h : headers) list.append(h.name + ": " + h.value);
    if (body) {
        list.append("Content-Type: application/json");
        curl_easy_setopt(easy, CURLOPT_POST, 1L);
        curl_easy_setopt(easy, CURLOPT_POSTFIELDS, body->c_str());
        curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<long long>(body->size()));
    }
    curl_easy_setopt(easy, CURLOPT_HTTPHEADER, list.raw);
    return true;
}

}  // namespace

std::string Reply::header_of(const std::string& name) const {
    std::string want = ieq_lower(name);
    for (const auto& h : headers) {
        if (ieq_lower(h.name) == want) return h.value;
    }
    return "";
}

bool Reply::has_header(const std::string& name) const {
    std::string want = ieq_lower(name);
    for (const auto& h : headers) {
        if (ieq_lower(h.name) == want) return true;
    }
    return false;
}

Reply post(const std::string& url, const std::string& body, const std::vector<HeaderEntry>& headers,
           long timeout_ms, long connect_timeout_ms) {
    global_init();
    CURL* easy = curl_easy_init();
    if (!easy) return Reply{false, 0, "", {}, "cannot start the request"};
    Slist list;
    char error_buffer[CURL_ERROR_SIZE] = {0};
    BufferedTransfer transfer;
    setup_common(easy, url, &body, headers, timeout_ms, connect_timeout_ms, list, error_buffer);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &BufferedTransfer::on_write);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, &BufferedTransfer::on_header);
    curl_easy_setopt(easy, CURLOPT_HEADERDATA, &transfer);

    Reply reply;
    CURLcode rc = curl_easy_perform(easy);
    reply.ok = rc == CURLE_OK;
    reply.status = transfer.head.status;
    reply.body = std::move(transfer.body);
    reply.headers = std::move(transfer.head.headers);
    if (!reply.ok) {
        reply.error = error_buffer[0] ? error_buffer : curl_easy_strerror(rc);
    }
    curl_easy_cleanup(easy);
    return reply;
}

Reply get(const std::string& url, long timeout_ms) {
    global_init();
    CURL* easy = curl_easy_init();
    if (!easy) return Reply{false, 0, "", {}, "cannot start the request"};
    Slist list;
    char error_buffer[CURL_ERROR_SIZE] = {0};
    BufferedTransfer transfer;
    setup_common(easy, url, nullptr, {}, timeout_ms, 0, list, error_buffer);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &BufferedTransfer::on_write);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, &BufferedTransfer::on_header);
    curl_easy_setopt(easy, CURLOPT_HEADERDATA, &transfer);

    Reply reply;
    CURLcode rc = curl_easy_perform(easy);
    reply.ok = rc == CURLE_OK;
    reply.status = transfer.head.status;
    reply.body = std::move(transfer.body);
    reply.headers = std::move(transfer.head.headers);
    if (!reply.ok) {
        reply.error = error_buffer[0] ? error_buffer : curl_easy_strerror(rc);
    }
    curl_easy_cleanup(easy);
    return reply;
}

// --- StreamRequest ---

struct StreamRequest::Impl {
    // Fixed at construction (curl reads them during the transfer).
    std::string url;
    std::string body;
    std::vector<HeaderEntry> headers;
    long timeout_ms = 0;
    long connect_timeout_ms = 0;

    // Worker-thread-owned curl state.
    CURL* easy = nullptr;
    Slist list;
    char error_buffer[CURL_ERROR_SIZE] = {0};
    HeaderBlock head;

    // Shared with the consumer. `fd` is written from the sockopt callback.
    std::atomic<int> fd{-1};

    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    bool head_ready = false;
    StreamRequest::Head published;
    std::deque<std::string> chunks;
    size_t buffered = 0;
    bool body_done = false;
    bool body_ok = false;
    std::string error;
    bool cancel = false;

    static constexpr size_t kMaxBuffered = 8 * 1024 * 1024;

    ~Impl() {
        if (easy) curl_easy_cleanup(easy);
    }

    static size_t on_header(char* data, size_t size, size_t count, void* self_) {
        auto* self = static_cast<Impl*>(self_);
        self->head.line(data, size * count);
        return size * count;
    }

    void publish_head(bool ok, const std::string& err) {
        std::lock_guard<std::mutex> lock(mu);
        if (!head_ready) {
            published.ok = ok;
            published.status = head.status;
            published.headers = head.headers;
            published.error = err;
            head_ready = true;
        }
        cv.notify_all();
    }

    static size_t on_write(char* data, size_t size, size_t count, void* self_) {
        auto* self = static_cast<Impl*>(self_);
        size_t len = size * count;
        if (!self->head_ready && self->head.got_status) {
            self->publish_head(true, "");
        }
        std::unique_lock<std::mutex> lock(self->mu);
        if (self->cancel) return 0;
        while (self->buffered >= kMaxBuffered && !self->cancel) {
            self->cv.wait(lock);
        }
        if (self->cancel) return 0;
        self->chunks.emplace_back(data, len);
        self->buffered += len;
        self->cv.notify_all();
        return len;
    }

    static int on_sockopt(void* self_, curl_socket_t sockfd, curlsocktype type) {
        if (type != CURLSOCKTYPE_IPCXN) return 0;
        auto* self = static_cast<Impl*>(self_);
        self->fd.store(static_cast<int>(sockfd));
        return 0;  // CURL_SOCKOPT_OK
    }

    void run() {
        global_init();
        easy = curl_easy_init();
        if (!easy) {
            std::lock_guard<std::mutex> lock(mu);
            body_done = true;
            body_ok = false;
            error = "cannot start the request";
            if (!head_ready) {
                published.ok = false;
                published.error = error;
                head_ready = true;
            }
            cv.notify_all();
            return;
        }
        setup_common(easy, url, &body, headers, timeout_ms, connect_timeout_ms, list, error_buffer);
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &Impl::on_write);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, &Impl::on_header);
        curl_easy_setopt(easy, CURLOPT_HEADERDATA, this);
        curl_easy_setopt(easy, CURLOPT_SOCKOPTFUNCTION, &Impl::on_sockopt);
        curl_easy_setopt(easy, CURLOPT_SOCKOPTDATA, this);

        CURLcode rc = curl_easy_perform(easy);

        std::lock_guard<std::mutex> lock(mu);
        body_done = true;
        body_ok = rc == CURLE_OK && !cancel;
        if (rc != CURLE_OK) {
            error = error_buffer[0] ? error_buffer : curl_easy_strerror(rc);
        } else if (cancel) {
            error = "cancelled";
        }
        if (!head_ready) {
            // Responses without a body never reach on_write, so the head is
            // published here from what the header callback collected.
            if (head.got_status) {
                published.ok = true;
                published.status = head.status;
                published.headers = head.headers;
            } else {
                published.ok = false;
                published.status = 0;
                published.error = error.empty() ? "no response received" : error;
            }
            head_ready = true;
        }
        cv.notify_all();
    }
};

StreamRequest::StreamRequest(const std::string& url, const std::string& body,
                             const std::vector<HeaderEntry>& headers, long timeout_ms,
                             long connect_timeout_ms)
    : impl_(std::make_unique<Impl>()) {
    impl_->url = url;
    impl_->body = body;
    impl_->headers = headers;
    impl_->timeout_ms = timeout_ms;
    impl_->connect_timeout_ms = connect_timeout_ms;
    impl_->worker = std::thread([impl = impl_.get()] { impl->run(); });
}

StreamRequest::~StreamRequest() {
    cancel();
    if (impl_->worker.joinable()) impl_->worker.join();
}

StreamRequest::Head StreamRequest::wait_head() {
    std::unique_lock<std::mutex> lock(impl_->mu);
    impl_->cv.wait(lock, [&] { return impl_->head_ready; });
    return impl_->published;
}

bool StreamRequest::next(std::string& chunk, std::chrono::milliseconds timeout, bool& timed_out) {
    std::unique_lock<std::mutex> lock(impl_->mu);
    bool signalled = impl_->cv.wait_for(
        lock, timeout, [&] { return !impl_->chunks.empty() || impl_->body_done || impl_->cancel; });
    timed_out = false;
    if (!signalled) {
        timed_out = true;
        return false;
    }
    if (impl_->chunks.empty()) return false;  // body ended (or cancelled)
    chunk = std::move(impl_->chunks.front());
    impl_->buffered -= chunk.size();
    impl_->chunks.pop_front();
    impl_->cv.notify_all();
    return true;
}

bool StreamRequest::finished() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->body_done && impl_->chunks.empty();
}

bool StreamRequest::body_ok() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->body_done && impl_->body_ok && impl_->chunks.empty();
}

std::string StreamRequest::error() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->error;
}

void StreamRequest::cancel() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->cancel && impl_->body_done) return;
    impl_->cancel = true;
    int fd = impl_->fd.load();
    if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
    impl_->cv.notify_all();
}

std::string StreamRequest::drain() {
    std::string out;
    std::string chunk;
    for (;;) {
        bool timed_out = false;
        if (!next(chunk, std::chrono::hours(24), timed_out)) break;
        out += chunk;
    }
    return out;
}

}  // namespace s1::http

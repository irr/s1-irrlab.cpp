# s1 — complexity-routing LLM proxy (C++).
# Default: release build → dist/s1.

BINARY   := s1
DIST     := dist
BUILD    := build
BUILD_DEBUG := build-debug
PREFIX   ?= $(HOME)/.local/bin
CONFIG   ?= s1.toml
URL      ?= http://127.0.0.1:1970

.PHONY: help configure build debug run run-debug sample test vet fmt install clean

.DEFAULT_GOAL := build

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

configure: ## Configure the CMake build (build/)
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release

build: configure ## Build release binary → dist/s1
	cmake --build $(BUILD) --parallel
	@mkdir -p $(DIST)
	cp -f $(BUILD)/$(BINARY) $(DIST)/$(BINARY)
	@echo "built $(DIST)/$(BINARY)"

debug: ## Configure + build the debug tree (build-debug/, -g -O0) for gdb/lldb
	cmake -S . -B $(BUILD_DEBUG) -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DEBUG) --parallel
	@echo "built $(BUILD_DEBUG)/$(BINARY) (plus s1_unit / s1_e2e)"

run: build ## Run the release proxy (CONFIG=s1.toml)
	./$(DIST)/$(BINARY) --config $(CONFIG)

run-debug: debug ## Run the debug binary (build-debug/s1) with the same CONFIG
	./$(BUILD_DEBUG)/$(BINARY) --config $(CONFIG)

# jq filter for `make sample`: slurps the reply body and a trailer curl writes
# with the routing headers, then prints a summary and the answer.
define SAMPLE_JQ
.[-1] as $$meta
| (if length > 1 then .[0] else {} end) as $$body
| ($$body.choices[0].message.content // $$body.error.message
    // "(no answer; finish_reason=\($$body.choices[0].finish_reason // "none"))") as $$answer
| "────────────────────────────────────────────────────────────",
  "prompt    \($$prompt)",
  "tier      \($$meta.tier)",
  "decision  \($$meta.decision)",
  "model     \($$body.model // "-")",
  "http      \($$meta.http)",
  "tokens    \($$body.usage.prompt_tokens // "-") in / \($$body.usage.completion_tokens // "-") out",
  "",
  ($$answer | sub("^\\s+"; "")),
  ""
endef
export SAMPLE_JQ

SAMPLE = jq -n --arg prompt "$$PROMPT" \
		'{model: "s1-auto", max_tokens: 4096, messages: [{role: "user", content: $$prompt}]}' \
	| curl -sS $(URL)/v1/chat/completions -H 'Content-Type: application/json' -d @- \
		-w '\n{"tier":"%header{x-s1-tier}","decision":"%header{x-s1-decision}","http":"%{http_code}"}' \
	| jq -rs --arg prompt "$$PROMPT" "$$SAMPLE_JQ"

sample: ## Send one routine and one demanding coding request to a running proxy (needs curl + jq; URL=...)
	@PROMPT='Write a one-line Python function that returns the square of a number.'; $(SAMPLE)
	@PROMPT='Our Rust HTTP service intermittently deadlocks under load since we added a connection pool shared across tokio tasks. Explain how to find the root cause, the likely lock-ordering and async-mutex mistakes, and outline a refactor across the pool, handler and shutdown modules. Keep the answer under 200 words.'; $(SAMPLE)

test: configure ## Unit + integration tests (mock decider and upstreams)
	cmake --build $(BUILD) --parallel
	ctest --test-dir $(BUILD) --output-on-failure

vet: configure ## strict -Werror build + format check
	cmake --build $(BUILD) --parallel --clean-first
	@command -v clang-format >/dev/null 2>&1 && clang-format --dry-run -Werror src/*.cpp src/*.hpp src/anthropic/*.cpp src/anthropic/*.hpp tests/*.cpp || echo "clang-format not found; skipping format check"

fmt: ## Format the code
	clang-format -i src/*.cpp src/*.hpp src/anthropic/*.cpp src/anthropic/*.hpp tests/*.cpp

install: build ## Install to PREFIX (default ~/.local/bin)
	install -m 0755 $(DIST)/$(BINARY) $(PREFIX)/$(BINARY)

clean: ## Remove build artifacts
	rm -rf $(BUILD) $(BUILD_DEBUG) $(DIST)

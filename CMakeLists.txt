cmake_minimum_required(VERSION 3.20)
project(s1-irrlab CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.json for IDE IntelliSense (VS Code C/C++, clangd, ...).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

add_library(s1_core STATIC
    src/anthropic/mod.cpp
    src/anthropic/request.cpp
    src/anthropic/response.cpp
    src/anthropic/stream.cpp
    src/config.cpp
    src/decider.cpp
    src/error.cpp
    src/http.cpp
    src/log.cpp
    src/openai.cpp
    src/server.cpp
    src/upstream.cpp
    src/utf8.cpp
)
target_include_directories(s1_core PUBLIC src)
target_include_directories(s1_core SYSTEM PUBLIC
    third_party
    third_party/nlohmann
    third_party/doctest
)
target_compile_definitions(s1_core PUBLIC CPPHTTPLIB_THREAD_POOL_COUNT=32)
target_link_libraries(s1_core PUBLIC CURL::libcurl Threads::Threads)
target_compile_options(s1_core PRIVATE -Wall -Wextra -Werror)

add_executable(s1 src/main.cpp)
target_link_libraries(s1 PRIVATE s1_core)
target_compile_options(s1 PRIVATE -Wall -Wextra -Werror)

enable_testing()

add_executable(s1_unit tests/unit.cpp)
target_link_libraries(s1_unit PRIVATE s1_core)
target_compile_options(s1_unit PRIVATE -Wall -Wextra -Werror)

add_executable(s1_e2e tests/e2e.cpp)
target_link_libraries(s1_e2e PRIVATE s1_core)
target_compile_options(s1_e2e PRIVATE -Wall -Wextra -Werror)

add_test(NAME unit COMMAND s1_unit)
add_test(NAME e2e COMMAND s1_e2e)

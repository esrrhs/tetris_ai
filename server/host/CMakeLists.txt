# Tetris AI 网页服务的 fakelua 宿主。
#
# 分层照 doudizhu_ai/server: 规则与推理留在 C++ (src/tetris + src/server/lite_actor),
# HTTP / 会话 / 路由交给 fakelua 与 server/lua/*.lua。宿主本身不写一行 socket 代码。
#
# 独立构建 (不需要顶层 CMakeLists, 不引入 googletest / LibTorch):
#   cmake -S server/host -B server/host/build -DCMAKE_BUILD_TYPE=Release \
#         -DFAKELUA_PREFIX=/usr/local
#   cmake --build server/host/build -j2
cmake_minimum_required(VERSION 3.16)
project(tetris_server CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# 只认指定前缀, 避免一台机器上 /usr/local 与 /opt/homebrew 多个 fakelua 混用。
set(FAKELUA_PREFIX "/opt/homebrew" CACHE PATH "已安装 fakelua 的前缀")
find_package(fakelua CONFIG QUIET HINTS "${FAKELUA_PREFIX}"
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu NO_DEFAULT_PATH)
if(NOT fakelua_FOUND)
    if(NOT FAKELUA_DIR)
        message(FATAL_ERROR
            "在 ${FAKELUA_PREFIX} 找不到 fakeluaConfig.cmake。二选一:\n"
            "  1) cmake --install <fakelua build> --prefix ${FAKELUA_PREFIX}\n"
            "  2) 配置时加 -DFAKELUA_PREFIX=/other/prefix 或 -DFAKELUA_DIR=/path/to/fakelua")
    endif()
    add_subdirectory(${FAKELUA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/fakelua)
endif()

if(TARGET fakelua::fakelua)
    set(FAKELUA_LINK fakelua::fakelua)
else()
    set(FAKELUA_LINK fakelua)
endif()

find_package(Threads REQUIRED)

add_executable(tetris_server
    main.cpp
    ${ROOT}/src/tetris/shapes.cpp
    ${ROOT}/src/tetris/bag.cpp
    ${ROOT}/src/tetris/board.cpp
    ${ROOT}/src/tetris/game.cpp
    ${ROOT}/src/tetris/render.cpp
    ${ROOT}/src/tetris/features.cpp
    ${ROOT}/src/tetris/env.cpp
    ${ROOT}/src/tetris/heuristic.cpp
    ${ROOT}/src/server/lite_actor.cpp
    ${ROOT}/src/server/policy.cpp
    ${ROOT}/src/server/session.cpp
)
target_include_directories(tetris_server PRIVATE ${ROOT}/src)
target_link_libraries(tetris_server PRIVATE ${FAKELUA_LINK} Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    target_compile_options(tetris_server PRIVATE -O3 -march=native)
else()
    target_compile_options(tetris_server PRIVATE -O3)
endif()

install(TARGETS tetris_server RUNTIME DESTINATION bin)

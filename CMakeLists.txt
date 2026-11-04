cmake_minimum_required(VERSION 3.14)
project(tetris_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/utf-8)
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

find_package(Threads REQUIRED)

# ---- Core (no Torch) ----
add_library(tetris_ai_core STATIC
  src/tetris/shapes.cpp
  src/tetris/bag.cpp
  src/tetris/board.cpp
  src/tetris/game.cpp
  src/tetris/render.cpp
  src/tetris/features.cpp
  src/tetris/env.cpp
)
target_include_directories(tetris_ai_core PUBLIC src)

add_executable(tetris_ai_demo src/demo/main.cpp)
target_link_libraries(tetris_ai_demo PRIVATE tetris_ai_core)

# ---- Optional LibTorch RL ----
option(TETRIS_AI_WITH_TORCH "Build PPO training with LibTorch" ON)
set(TETRIS_AI_TORCH_FOUND FALSE)
if(TETRIS_AI_WITH_TORCH)
  find_package(Torch QUIET)
  if(Torch_FOUND)
    set(TETRIS_AI_TORCH_FOUND TRUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "LibTorch found: building tetris_ai_rl + tetris_ai_train + tetris_ai_eval")
  else()
    message(STATUS "LibTorch not found: skipping RL targets (core + tests only)")
  endif()
endif()

if(TETRIS_AI_TORCH_FOUND)
  add_library(tetris_ai_rl STATIC
    src/algo/ppo.cpp
    src/algo/rollout.cpp
    src/algo/eval.cpp
    src/torch_nn/models.cpp
    src/torch_nn/ppo_learner.cpp
    src/torch_nn/tensor_bridge.cpp
  )
  target_include_directories(tetris_ai_rl PUBLIC src ${TORCH_INCLUDE_DIRS})
  target_link_libraries(tetris_ai_rl PUBLIC tetris_ai_core Threads::Threads ${TORCH_LIBRARIES})

  add_executable(tetris_ai_train src/train.cpp)
  target_link_libraries(tetris_ai_train PRIVATE tetris_ai_rl)

  add_executable(tetris_ai_eval src/eval.cpp)
  target_link_libraries(tetris_ai_eval PRIVATE tetris_ai_rl)
endif()

# ---- Tests ----
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(test_game tests/test_game.cpp)
target_link_libraries(test_game PRIVATE tetris_ai_core GTest::gtest_main)
gtest_discover_tests(test_game)

add_executable(test_rl_env tests/test_rl_env.cpp)
target_link_libraries(test_rl_env PRIVATE tetris_ai_core GTest::gtest_main)
gtest_discover_tests(test_rl_env)

if(TETRIS_AI_TORCH_FOUND)
  add_executable(test_rl tests/test_rl.cpp)
  target_link_libraries(test_rl PRIVATE tetris_ai_rl GTest::gtest_main)
  gtest_discover_tests(test_rl)
endif()

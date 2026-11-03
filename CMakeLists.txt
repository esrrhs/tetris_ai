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

add_library(tetris_ai_core STATIC
  src/tetris/shapes.cpp
  src/tetris/bag.cpp
  src/tetris/board.cpp
  src/tetris/game.cpp
  src/tetris/render.cpp
)
target_include_directories(tetris_ai_core PUBLIC src)

add_executable(tetris_ai_demo src/demo/main.cpp)
target_link_libraries(tetris_ai_demo PRIVATE tetris_ai_core)

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

# C++ contenders (tlx::btree_map, absl::btree_map) for the perf.md
# scoreboard. See main.cc for methodology.
#
#   cmake -S benches/cpp -B benches/cpp/build -DCMAKE_PREFIX_PATH=/opt/homebrew
#   cmake --build benches/cpp/build
#   ./benches/cpp/build/bench_cpp
#
# abseil comes from Homebrew (`brew install abseil`); tlx is fetched
# from GitHub at configure time.

cmake_minimum_required(VERSION 3.24)
project(beets_cpp_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(absl CONFIG REQUIRED)

# tlx's own CMakeLists predates CMake 4's minimum-version floor.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(FetchContent)
FetchContent_Declare(
  tlx
  GIT_REPOSITORY https://github.com/tlx/tlx.git
  GIT_TAG v0.6.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(tlx)

add_executable(bench_cpp main.cc)
target_link_libraries(bench_cpp PRIVATE absl::btree tlx)

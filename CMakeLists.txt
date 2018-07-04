cmake_minimum_required(VERSION 3.10)
project(Homework4)

set(CMAKE_CXX_STANDARD 17)
set(Sources UnitTests.cpp)
set(Headers Generators.hpp BucketKNN.hpp KDTree.hpp Point.hpp Stopwatch.hpp catch.hpp UnitTests.cpp Naive.hpp Benchmark.hpp Spinner.hpp QuadTree.hpp VPTree.hpp)
add_executable(superlib ${Sources} ${Headers})

add_executable(Homework4 main.cpp)
cmake_minimum_required(VERSION 3.20)
project(WiDec)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/cpp)

add_executable(WiDec src/cpp/models.cpp)

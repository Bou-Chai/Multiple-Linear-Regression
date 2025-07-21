cmake_minimum_required(VERSION 3.28)
project(Multiple-Linear-Regression)

add_subdirectory(./libs/Tables)

add_library(libMLRModel ./src/MLRModel.cpp)
target_include_directories(libMLRModel PUBLIC ./include PRIVATE ./src)
target_link_libraries(libMLRModel libTables)

add_executable(testMLRModel ./tests/Test.cpp)
target_link_libraries(testMLRModel libMLRModel)

cmake_minimum_required(VERSION 3.20)
project(qsengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(qsengine src/bindings.cpp)
target_include_directories(qsengine PRIVATE include)

if(MSVC)
    target_compile_options(qsengine PRIVATE /O2 /EHsc /permissive- /W3 $<$<CONFIG:Release>:/GL>)
    target_link_options(qsengine PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_compile_options(qsengine PRIVATE -O3 -Wall -Wextra)
endif()

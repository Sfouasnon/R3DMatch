cmake_minimum_required(VERSION 3.20)
project(r3d_sdk_wrapper LANGUAGES CXX)

option(R3DSPLAT_ENABLE_RED_SDK "Build the optional RED SDK wrapper" OFF)
set(RED_SDK_ROOT "" CACHE PATH "Path to a local RED SDK installation")
set(RED_SDK_INCLUDE_DIR "" CACHE PATH "Path to RED SDK headers")
set(RED_SDK_LIBRARY_DIR "" CACHE PATH "Path to RED SDK redistributable libraries")
set(RED_SDK_REDISTRIBUTABLE_DIR "" CACHE PATH "Path to RED SDK runtime shared libraries")

if(NOT R3DSPLAT_ENABLE_RED_SDK)
  message(STATUS "RED SDK wrapper disabled. Enable with -DR3DSPLAT_ENABLE_RED_SDK=ON and provide SDK paths.")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

if(NOT RED_SDK_INCLUDE_DIR)
  if(RED_SDK_ROOT)
    set(RED_SDK_INCLUDE_DIR "${RED_SDK_ROOT}/Include")
  endif()
endif()

if(NOT RED_SDK_LIBRARY_DIR)
  if(RED_SDK_ROOT)
    set(RED_SDK_LIBRARY_DIR "${RED_SDK_ROOT}/Lib/linux64")
  endif()
endif()

if(NOT RED_SDK_REDISTRIBUTABLE_DIR)
  if(RED_SDK_ROOT)
    set(RED_SDK_REDISTRIBUTABLE_DIR "${RED_SDK_ROOT}/Redistributable/linux")
  endif()
endif()

pybind11_add_module(_r3d_native src/module.cpp)
target_compile_features(_r3d_native PRIVATE cxx_std_17)
target_include_directories(_r3d_native PRIVATE include)

if(RED_SDK_INCLUDE_DIR)
  target_include_directories(_r3d_native PRIVATE "${RED_SDK_INCLUDE_DIR}")
endif()

if(RED_SDK_ROOT)
  target_compile_definitions(_r3d_native PRIVATE R3DSPLAT_RED_SDK_ROOT="${RED_SDK_ROOT}")
endif()

if(RED_SDK_LIBRARY_DIR)
  target_link_directories(_r3d_native PRIVATE "${RED_SDK_LIBRARY_DIR}")
endif()

if(RED_SDK_REDISTRIBUTABLE_DIR)
  target_compile_definitions(_r3d_native PRIVATE R3DSPLAT_RED_SDK_REDIS_DIR="${RED_SDK_REDISTRIBUTABLE_DIR}")
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(_r3d_native PRIVATE R3DSDKPIC-cpp11 dl)
endif()

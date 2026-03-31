cmake_minimum_required(VERSION 3.20)
project(r3dmatch_red_sdk_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED RED_SDK_ROOT)
  set(RED_SDK_ROOT "$ENV{RED_SDK_ROOT}" CACHE PATH "Path to a local RED SDK installation")
endif()
if(NOT DEFINED RED_SDK_INCLUDE_DIR)
  set(RED_SDK_INCLUDE_DIR "$ENV{RED_SDK_INCLUDE_DIR}" CACHE PATH "Path to RED SDK headers")
endif()
if(NOT DEFINED RED_SDK_LIBRARY_DIR)
  set(RED_SDK_LIBRARY_DIR "$ENV{RED_SDK_LIBRARY_DIR}" CACHE PATH "Path to RED SDK libraries")
endif()
if(NOT DEFINED RED_SDK_REDISTRIBUTABLE_DIR)
  set(RED_SDK_REDISTRIBUTABLE_DIR "$ENV{RED_SDK_REDISTRIBUTABLE_DIR}" CACHE PATH "Path to RED SDK redistributable runtime libraries")
endif()

if(NOT RED_SDK_INCLUDE_DIR AND RED_SDK_ROOT)
  if(EXISTS "${RED_SDK_ROOT}/Include")
    set(RED_SDK_INCLUDE_DIR "${RED_SDK_ROOT}/Include")
  elseif(EXISTS "${RED_SDK_ROOT}/include")
    set(RED_SDK_INCLUDE_DIR "${RED_SDK_ROOT}/include")
  endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(NOT RED_SDK_INCLUDE_DIR AND RED_SDK_ROOT)
  set(RED_SDK_INCLUDE_DIR "${RED_SDK_ROOT}/Include")
endif()

if(NOT RED_SDK_LIBRARY_DIR AND RED_SDK_ROOT)
  if(APPLE)
    set(RED_SDK_LIBRARY_DIR "${RED_SDK_ROOT}/Lib/mac64")
  elseif(WIN32)
    set(RED_SDK_LIBRARY_DIR "${RED_SDK_ROOT}/Lib/win64")
  else()
    set(RED_SDK_LIBRARY_DIR "${RED_SDK_ROOT}/Lib/linux64")
  endif()
endif()

if(NOT RED_SDK_REDISTRIBUTABLE_DIR AND RED_SDK_ROOT)
  if(APPLE)
    set(RED_SDK_REDISTRIBUTABLE_DIR "${RED_SDK_ROOT}/Redistributable/mac")
  elseif(WIN32)
    set(RED_SDK_REDISTRIBUTABLE_DIR "${RED_SDK_ROOT}/Redistributable/win")
  else()
    set(RED_SDK_REDISTRIBUTABLE_DIR "${RED_SDK_ROOT}/Redistributable/linux")
  endif()
endif()

add_library(_red_sdk_bridge MODULE red_sdk_bridge.cpp)
target_link_libraries(_red_sdk_bridge PRIVATE pybind11::module Python3::Module)
target_include_directories(_red_sdk_bridge PRIVATE ${Python3_INCLUDE_DIRS})

if(RED_SDK_ROOT)
  target_compile_definitions(_red_sdk_bridge PRIVATE R3DMATCH_RED_SDK_ENABLED=1)
  target_compile_definitions(_red_sdk_bridge PRIVATE R3DMATCH_RED_SDK_ROOT="${RED_SDK_ROOT}")
endif()

if(RED_SDK_INCLUDE_DIR)
  target_include_directories(_red_sdk_bridge PRIVATE "${RED_SDK_INCLUDE_DIR}")
  target_compile_definitions(_red_sdk_bridge PRIVATE R3DMATCH_RED_SDK_INCLUDE_DIR="${RED_SDK_INCLUDE_DIR}")
endif()

if(RED_SDK_LIBRARY_DIR)
  target_link_directories(_red_sdk_bridge PRIVATE "${RED_SDK_LIBRARY_DIR}")
  target_compile_definitions(_red_sdk_bridge PRIVATE R3DMATCH_RED_SDK_LIBRARY_DIR="${RED_SDK_LIBRARY_DIR}")
endif()

if(RED_SDK_REDISTRIBUTABLE_DIR)
  target_compile_definitions(_red_sdk_bridge PRIVATE R3DMATCH_RED_SDK_REDIS_DIR="${RED_SDK_REDISTRIBUTABLE_DIR}")
endif()

if(RED_SDK_ROOT)
  if(APPLE)
    target_link_libraries(_red_sdk_bridge PRIVATE R3DSDK-libcpp)
  elseif(WIN32)
    target_link_libraries(_red_sdk_bridge PRIVATE R3DSDK)
  else()
    target_link_libraries(_red_sdk_bridge PRIVATE R3DSDKPIC-cpp11)
  endif()

  if(UNIX AND NOT APPLE)
    target_link_libraries(_red_sdk_bridge PRIVATE dl)
  endif()
endif()

set_target_properties(
  _red_sdk_bridge
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_red_sdk_bridge"
)

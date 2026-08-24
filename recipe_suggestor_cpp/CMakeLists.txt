cmake_minimum_required(VERSION 3.16)
project(RecipeSuggestorCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimised build. With CMAKE_BUILD_TYPE empty -- which is what
# `cmake -S . -B build` gives you -- there is NO -O flag at all, and everything
# here is compute-bound: detector pre/post-processing, the pipeline, and above
# all the 2^32 seed scan in tools/find_start_seed, which runs tens of times
# slower unoptimised.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE was empty; defaulting to Release")
endif()

# Debug
# add_compile_definitions(DEBUG)

option(RS_SANITIZE "Build with the address and undefined-behaviour sanitizers" OFF)

find_package(OpenCV REQUIRED)
find_package(X11 QUIET)

# ONNX Runtime Setup
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.16.3")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ORT_LIB_DIR} REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ORT_INCLUDE_DIR})

add_compile_options(-Wall -Wextra)
if(RS_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Core library: every source except main(). Each test links this instead of
# repeating its own list of .cpp files, which is what let floor_test drift into
# an undefined-symbol link failure.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE RS_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM RS_CORE_SRC ${CMAKE_SOURCE_DIR}/src/main.cpp)

if(NOT X11_FOUND)
    message(WARNING "X11 not found: screen capture and the full pipeline are excluded.")
    list(REMOVE_ITEM RS_CORE_SRC
        ${CMAKE_SOURCE_DIR}/src/pipeline/nodes/frame_capturer.cpp
        ${CMAKE_SOURCE_DIR}/src/pipeline/pipeline.cpp)
endif()

add_library(rs_core STATIC ${RS_CORE_SRC})
target_link_libraries(rs_core PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} crypto)
if(X11_FOUND)
    target_link_libraries(rs_core PUBLIC ${X11_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# Runtime assets. Every binary is expected to run from its own directory, so
# resources/ is staged next to them and all paths stay relative to the binary.
# ---------------------------------------------------------------------------
add_custom_target(stage_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources ${CMAKE_CURRENT_BINARY_DIR}/resources
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_LIB_DIR}/libonnxruntime.so.1.16.3 ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Staging resources/ and libonnxruntime.so next to the binaries")

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
if(X11_FOUND)
    add_executable(${PROJECT_NAME} src/main.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE rs_core)
    add_dependencies(${PROJECT_NAME} stage_resources)
endif()

# ROI calibration helper (not a test -- human looks at the overlay it writes)
add_executable(roi_preview tools/roi_preview.cpp)
target_link_libraries(roi_preview PRIVATE rs_core)
add_dependencies(roi_preview stage_resources)

# Records crafts from the live game into an observations file
if(X11_FOUND)
    add_executable(craft_recorder tools/craft_recorder.cpp)
    target_link_libraries(craft_recorder PRIVATE rs_core)
    add_dependencies(craft_recorder stage_resources)
endif()

# Reads the bag out of captured frames, for building observations
add_executable(bag_log tools/bag_log.cpp)
target_link_libraries(bag_log PRIVATE rs_core)
add_dependencies(bag_log stage_resources)

# Recovers the run's 32-bit crafting seed from observed crafts
add_executable(find_start_seed tools/find_start_seed.cpp)
target_link_libraries(find_start_seed PRIVATE rs_core)
add_dependencies(find_start_seed stage_resources)

set(RS_TESTS detector_test boc_test boc_geometry_test screen_regression_test floor_test router_test bag_of_crafting_test trie_test data_test suggestor_test)
if(X11_FOUND)
    list(APPEND RS_TESTS capturer_test pipeline_test)
endif()

enable_testing()
foreach(TEST_NAME ${RS_TESTS})
    add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE rs_core)
    add_dependencies(${TEST_NAME} stage_resources)
    add_test(NAME ${TEST_NAME}
             COMMAND ${TEST_NAME}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

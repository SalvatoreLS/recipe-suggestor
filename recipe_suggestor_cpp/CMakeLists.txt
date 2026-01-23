cmake_minimum_required(VERSION 3.10)
project(RecipeSuggestorCPP LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(X11 QUIET)

# ONNX Runtime Setup
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.16.3")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ORT_LIB_DIR} REQUIRED)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ORT_INCLUDE_DIR})

# Collect all source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

if(X11_FOUND)


# Remove detector_torch.cpp if it's still picked up (should be deleted)
list(REMOVE_ITEM SRC_FILES ${CMAKE_SOURCE_DIR}/src/pipeline/detector_torch.cpp)

# Define executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${X11_LIBRARIES})

    # Copy ONNX Runtime lib to build directory for runtime finding
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_LIB_DIR}/libonnxruntime.so.1.16.3 $<TARGET_FILE_DIR:${PROJECT_NAME}>)
else()
    message(WARNING "X11 not found. Skipping RecipeSuggestorCPP build.")
endif()

# Test Executable
add_executable(detector_test tests/detector_test.cpp src/detector.cpp)
target_link_libraries(detector_test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

add_executable(boc_test tests/boc_test.cpp src/detector.cpp)
target_link_libraries(boc_test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} stdc++fs)

add_executable(floor_test tests/floor_test.cpp src/detector.cpp)
target_link_libraries(floor_test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

add_executable(capturer_test tests/capturer_test.cpp src/pipeline/nodes/frame_capturer.cpp src/utils.cpp)
target_link_libraries(capturer_test PRIVATE ${OpenCV_LIBS} ${X11_LIBRARIES})

add_executable(router_test tests/router_test.cpp src/pipeline/nodes/router.cpp src/utils.cpp)
target_link_libraries(router_test PRIVATE ${OpenCV_LIBS})

# For pipeline test
add_executable(pipeline_test tests/pipeline_test.cpp 
    src/pipeline/pipeline.cpp 
    src/pipeline/nodes/frame_capturer.cpp 
    src/pipeline/nodes/router.cpp 
    src/detector.cpp 
    src/pipeline/nodes/recipe_suggestor.cpp
    src/utils.cpp)
target_link_libraries(pipeline_test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} ${X11_LIBRARIES})
add_executable(trie_test tests/trie_test.cpp src/data_structures/trie.cpp)
target_link_libraries(trie_test PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

# Copy ONNX Runtime lib to build directory for runtime finding
foreach(TARGET_NAME detector_test boc_test floor_test capturer_test router_test pipeline_test trie_test)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_LIB_DIR}/libonnxruntime.so.1.16.3 $<TARGET_FILE_DIR:${TARGET_NAME}>)
endforeach()

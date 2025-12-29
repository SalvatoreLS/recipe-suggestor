cmake_minimum_required(VERSION 3.10)
project(RecipeSuggestorCPP LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use LibTorch
option(USE_LIBTORCH "Use LibTorch for inference instead of OpenCV DNN" OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

if(USE_LIBTORCH)
    add_definitions(-DUSE_LIBTORCH)
    include_directories(${CMAKE_SOURCE_DIR}/libtorch/include)
    include_directories(${CMAKE_SOURCE_DIR}/libtorch/include/torch/csrc/api/include)
endif()

# Collect all source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Exclude detector.cpp if using LibTorch, include detector_torch.cpp
if(USE_LIBTORCH)
    list(REMOVE_ITEM SRC_FILES ${CMAKE_SOURCE_DIR}/src/pipeline/detector.cpp)
    list(APPEND SRC_FILES ${CMAKE_SOURCE_DIR}/src/pipeline/detector_torch.cpp)
else()
    list(REMOVE_ITEM SRC_FILES ${CMAKE_SOURCE_DIR}/src/pipeline/detector_torch.cpp)
endif()

# Define executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

if(USE_LIBTORCH)
    # Link LibTorch
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch.so)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/libtorch/lib/libtorch_cpu.so)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/libtorch/lib/libc10.so)
endif()

# Test Executable
if(NOT USE_LIBTORCH)
    add_executable(detector_test tests/detector_test.cpp src/pipeline/detector.cpp)
    target_link_libraries(detector_test PRIVATE ${OpenCV_LIBS})
endif()

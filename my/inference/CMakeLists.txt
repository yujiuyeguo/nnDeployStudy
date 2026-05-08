cmake_minimum_required(VERSION 3.16)
project(isaaclab_onnx_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX Runtime SDK root")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT to your ONNX Runtime SDK path.")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
    NO_DEFAULT_PATH
)



if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "libonnxruntime not found under ${ONNXRUNTIME_ROOT}/lib or lib64")
endif()



add_executable(onnx_infer onnx_infer.cpp)
target_include_directories(onnx_infer PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(onnx_infer PRIVATE "${ONNXRUNTIME_LIBRARY}")

add_executable(imu_change_infer imu_change_infer.cpp)
target_include_directories(imu_change_infer PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(imu_change_infer PRIVATE "${ONNXRUNTIME_LIBRARY}")

if(UNIX AND NOT APPLE)
    get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    target_link_options(onnx_infer PRIVATE "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
    target_link_options(imu_change_infer PRIVATE "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
endif()

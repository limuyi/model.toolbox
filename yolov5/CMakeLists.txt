cmake_minimum_required(VERSION 3.10)
project(model.toolbox.yolov5)

set(CMAKE_CXX_FLAGS "-std=c++11" )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)
set(RUN_FILE_NAME "detect")

# message("PROJECT_SOURCE_DIR: "${PROJECT_SOURCE_DIR})
# message("PROJECT_NAME: "${PROJECT_NAME})

# 头文件路径
include_directories(${PROJECT_SOURCE_DIR}/../third_libs/install/onnxruntime/include)
include_directories(${PROJECT_SOURCE_DIR}/../common)
include_directories(${PROJECT_SOURCE_DIR}/include)

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB YOLOV5_SRCS (${PROJECT_SOURCE_DIR}/src/*.cpp))
file(GLOB UTILS_SRCS (${PROJECT_SOURCE_DIR}/../common/*.cpp))

# 生成可执行文件
add_executable(${RUN_FILE_NAME} main.cpp ${UTILS_SRCS} ${YOLOV5_SRCS})

# 链接OpenCV库
target_link_libraries(${RUN_FILE_NAME} ${OpenCV_LIBS})

target_link_libraries(${RUN_FILE_NAME} ${PROJECT_SOURCE_DIR}/../third_libs/install/onnxruntime/lib/libonnxruntime.so.1.4.0)

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/limuyi/code/model.toolbox/third_libs/install/onnxruntime/lib
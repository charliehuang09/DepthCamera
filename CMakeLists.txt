cmake_minimum_required(VERSION 3.8)
project(DepthCamera)


find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

add_executable(main main.cc)
target_link_libraries(main ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(main nvinfer)
target_link_libraries(main nvinfer_plugin)
target_link_libraries(main cudart)

add_executable(frame_logger frame_logger.cc)
target_link_libraries(frame_logger ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(frame_logger nvinfer)
target_link_libraries(frame_logger nvinfer_plugin)
target_link_libraries(frame_logger cudart)

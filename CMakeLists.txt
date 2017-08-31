cmake_minimum_required(VERSION 2.8.3)
project(cvTest)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
)

add_executable(cuda_test
	src/cuda_test.cpp
)

target_link_libraries(cuda_test
	${OpenCV_LIBS}
)

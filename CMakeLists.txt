cmake_minimum_required(VERSION 3.13.0)
project(neural_network VERSION 0.0.1)

set(CMAKE_BUILD_TYPE Debug)


find_package(xtensor REQUIRED)
include_directories(${XTENSOR_INCLUDE_DIRS})
find_package(xtensor-blas REQUIRED)
include_directories(${XTENSOR_BLAS_INCLUDE_DIRS})

include_directories(src)

add_executable(neural_network src/main.cpp

	src/neural_network/dense.cpp
	src/neural_network/sequential.cpp

	src/neural_network/relu.cpp
	src/neural_network/sigmoid.cpp

	src/neural_network/softmax_loss.cpp
	src/neural_network/simple_loss.cpp

	# src/neural_network/
	# src/neural_network/
	# src/neural_network/
	# src/neural_network/
)

# find_package(OpenBLAS REQUIRED)

target_link_libraries(neural_network cblas)
# target_link_libraries(neural_network OpenBLAS::OpenBLAS)


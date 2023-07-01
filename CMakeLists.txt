cmake_minimum_required(VERSION 3.17)

project(Voronoi LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

set(SOURCES 
	src/main.cu
	)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES 
	CUDA_SEPERABLE_COMPILATION ON)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--keep # Keep PTX code 						
	>)

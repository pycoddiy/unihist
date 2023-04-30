cmake_minimum_required(VERSION 3.23)

project(sycl_histogram LANGUAGES CXX)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Add the executable target
pybind11_add_module(${PROJECT_NAME} cpp/hist_sycl.cpp)

SET(CONDA_PREFIX $ENV{CONDA_PREFIX})

find_package(IntelSYCL REQUIRED)

add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES cpp/hist_sycl.cpp)

list(APPEND CMAKE_MODULE_PATH "${DPCTL_MODULE_PATH}")

find_package(Dpctl REQUIRED)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${CONDA_PREFIX}/include ${Dpctl_INCLUDE_DIR})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
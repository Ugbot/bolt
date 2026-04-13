cmake_minimum_required(VERSION 3.20)
project(bolt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(BOLT_IS_TOPLEVEL ON)
else()
    set(BOLT_IS_TOPLEVEL OFF)
endif()

option(BOLT_BUILD_TESTS "Build bolt unit tests" ${BOLT_IS_TOPLEVEL})
option(BOLT_BUILD_BENCHMARKS "Build bolt benchmarks" ${BOLT_IS_TOPLEVEL})

add_library(bolt INTERFACE)
add_library(bolt::bolt ALIAS bolt)

target_include_directories(bolt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(bolt INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bolt INTERFACE -fno-exceptions -fno-rtti)
elseif(MSVC)
    target_compile_options(bolt INTERFACE /GR- /EHs-c-)
endif()

if(BOLT_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        find_package(GTest QUIET)
    endif()
    if(GTest_FOUND)
        add_executable(test_bolt_primitives tests/test_bolt_primitives.cpp)
        target_link_libraries(test_bolt_primitives PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
        add_test(NAME test_bolt_primitives COMMAND test_bolt_primitives)
    else()
        message(STATUS "GTest not found; skipping bolt tests")
    endif()
endif()

if(BOLT_BUILD_BENCHMARKS)
    add_executable(bench_bolt benchmarks/bench_bolt.cpp)
    target_link_libraries(bench_bolt PRIVATE bolt::bolt)
    if(NOT MSVC)
        target_compile_options(bench_bolt PRIVATE -O3 -march=native)
        target_link_libraries(bench_bolt PRIVATE pthread)
    endif()
endif()

include(GNUInstallDirs)
install(DIRECTORY include/bolt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bolt EXPORT bolt-targets)
install(EXPORT bolt-targets
    FILE bolt-config.cmake
    NAMESPACE bolt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bolt
)

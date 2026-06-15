# bolt::ybolt — bolt-arena-backed binding for the ycpp CRDT runtime.
#
# Header-only INTERFACE library; pulls bolt::core (for Arena) and ycpp::headers
# (for the algorithm + wire format). Builds only when both targets are
# present — Gestalt2's top-level CMake wires this by adding extern/ycpp/
# BEFORE extern/bolt/ so ycpp::ycpp is defined by the time we get here.

if(NOT TARGET ycpp::headers AND NOT TARGET ycpp::ycpp)
    message(STATUS "bolt::ybolt — ycpp target not present, skipping")
    return()
endif()

add_library(bolt_ybolt INTERFACE)
add_library(bolt::ybolt ALIAS bolt_ybolt)

target_include_directories(bolt_ybolt INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(bolt_ybolt INTERFACE
    bolt::core
    ycpp::headers
)

# NOTE: ybolt is NOT folded into the bolt umbrella INTERFACE target — its
# install set is independent. Consumers that want the binding link
# `bolt::ybolt` explicitly. This keeps bolt's existing `bolt-targets`
# install export clean.

if(BOLT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

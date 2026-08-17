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

# ybolt is NOT folded into the bolt umbrella INTERFACE target — consumers that
# want the binding link `bolt::ybolt` explicitly.
#
# The previous note here claimed ybolt's "install set is independent". No such
# install set existed anywhere in the repo, so bolt::ybolt could not be
# installed at all while `$<INSTALL_INTERFACE:include>` above promised a header
# location nothing ever populated. bolt_ybolt now joins bolt's own export set
# (see the parent CMakeLists), and the header it promises is installed here.
#
# Note this target links ycpp::headers INTERFACE, so exporting it requires ycpp
# to be an exported package too — bolt-config.cmake.in find_dependency's ycpp
# when ybolt was built.
if(BOLT_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/bolt
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")
endif()

if(BOLT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Yjs interop CLI — exercises ycpp's algorithm through bolt::Arena.
# Driven by extern/ycpp/tools/yjs_interop/test.js with `node test.js path/to/ybolt_cli`.
#
# Gated: this was the one unguarded add_executable in bolt. It was built
# whenever a ycpp target happened to be present, regardless of
# BOLT_BUILD_TESTS/BENCHMARKS — so an embedding build that asked for no tests
# and no benchmarks still compiled and linked a CLI it never uses.
option(BOLT_BUILD_YBOLT_CLI "Build the ybolt Yjs-interop CLI" ${BOLT_BUILD_TESTS})
if(BOLT_BUILD_YBOLT_CLI)
    add_executable(ybolt_cli tools/ybolt_cli.cpp)
    target_link_libraries(ybolt_cli PRIVATE bolt::ybolt)
    if(COMMAND bolt_apply_hardening)
        bolt_apply_hardening(ybolt_cli)
    endif()
endif()

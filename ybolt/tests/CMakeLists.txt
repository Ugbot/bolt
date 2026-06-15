# bolt::ybolt tests — exercise the BoltArenaAllocator through ycpp's
# primitives + verify the adapter compiles + links + behaves.

add_executable(test_bolt_ybolt_smoke test_bolt_ybolt_smoke.cpp)
target_link_libraries(test_bolt_ybolt_smoke
    PRIVATE
        bolt::ybolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_ybolt_smoke)
add_test(NAME test_bolt_ybolt_smoke COMMAND test_bolt_ybolt_smoke)

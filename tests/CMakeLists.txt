add_executable(test_bolt_primitives test_bolt_primitives.cpp)

target_link_libraries(test_bolt_primitives
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_primitives)

add_test(NAME test_bolt_primitives COMMAND test_bolt_primitives)

add_executable(test_bolt_scheduler test_bolt_scheduler.cpp)

target_link_libraries(test_bolt_scheduler
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_scheduler)

add_test(NAME test_bolt_scheduler COMMAND test_bolt_scheduler)

add_executable(test_bolt_strtemp test_bolt_strtemp.cpp)

target_link_libraries(test_bolt_strtemp
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_strtemp)

add_test(NAME test_bolt_strtemp COMMAND test_bolt_strtemp)

add_executable(test_bolt_join test_bolt_join.cpp)

target_link_libraries(test_bolt_join
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_join)

add_test(NAME test_bolt_join COMMAND test_bolt_join)

add_executable(test_bolt_wire test_bolt_wire.cpp)

target_link_libraries(test_bolt_wire
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wire)

add_test(NAME test_bolt_wire COMMAND test_bolt_wire)

add_executable(test_bolt_kernels test_bolt_kernels.cpp)

target_link_libraries(test_bolt_kernels
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_kernels)

add_test(NAME test_bolt_kernels COMMAND test_bolt_kernels)

add_executable(test_bolt_simd test_bolt_simd.cpp)

target_link_libraries(test_bolt_simd
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_simd)

add_test(NAME test_bolt_simd COMMAND test_bolt_simd)

add_executable(test_bolt_topology test_bolt_topology.cpp)

target_link_libraries(test_bolt_topology
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_topology)

add_test(NAME test_bolt_topology COMMAND test_bolt_topology)

add_executable(test_bolt_parallel test_bolt_parallel.cpp)

target_link_libraries(test_bolt_parallel
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_parallel)

add_test(NAME test_bolt_parallel COMMAND test_bolt_parallel)

add_executable(test_bolt_config test_bolt_config.cpp)

target_link_libraries(test_bolt_config
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_config)

add_test(NAME test_bolt_config COMMAND test_bolt_config)

add_executable(test_bolt_arrow test_bolt_arrow.cpp)

target_link_libraries(test_bolt_arrow
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_arrow)

add_test(NAME test_bolt_arrow COMMAND test_bolt_arrow)

add_executable(test_bolt_stream test_bolt_stream.cpp)

target_link_libraries(test_bolt_stream
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_stream)

add_test(NAME test_bolt_stream COMMAND test_bolt_stream)

add_executable(test_bolt_branchless test_bolt_branchless.cpp)

target_link_libraries(test_bolt_branchless
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_branchless)

add_test(NAME test_bolt_branchless COMMAND test_bolt_branchless)

if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_ascii test_bolt_ascii.cpp)
    target_link_libraries(test_bolt_ascii
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_ascii)
    add_test(NAME test_bolt_ascii COMMAND test_bolt_ascii)
endif()

if(BOLT_BUILD_INGEST)
    add_executable(test_bolt_csv test_bolt_csv.cpp)
    target_link_libraries(test_bolt_csv
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_csv)
    add_test(NAME test_bolt_csv COMMAND test_bolt_csv)
endif()

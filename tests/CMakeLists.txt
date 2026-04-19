add_executable(test_bolt_primitives test_bolt_primitives.cpp)

target_link_libraries(test_bolt_primitives
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_primitives)

add_test(NAME test_bolt_primitives COMMAND test_bolt_primitives)

# ZoneMap primitive — MarbleDB Wave 10.1 consolidation.
add_executable(test_bolt_zonemap test_bolt_zonemap.cpp)
target_link_libraries(test_bolt_zonemap
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_zonemap)
add_test(NAME test_bolt_zonemap COMMAND test_bolt_zonemap)

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

add_executable(test_bolt_mergejoin test_bolt_mergejoin.cpp)

target_link_libraries(test_bolt_mergejoin
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_mergejoin)

add_test(NAME test_bolt_mergejoin COMMAND test_bolt_mergejoin)

add_executable(test_bolt_fintech_phase1 test_bolt_fintech_phase1.cpp)

target_link_libraries(test_bolt_fintech_phase1
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase1)

add_test(NAME test_bolt_fintech_phase1 COMMAND test_bolt_fintech_phase1)

add_executable(test_bolt_fintech_phase2 test_bolt_fintech_phase2.cpp)

target_link_libraries(test_bolt_fintech_phase2
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase2)

add_test(NAME test_bolt_fintech_phase2 COMMAND test_bolt_fintech_phase2)

add_executable(test_bolt_fintech_phase4a test_bolt_fintech_phase4a.cpp)

target_link_libraries(test_bolt_fintech_phase4a
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase4a)

add_test(NAME test_bolt_fintech_phase4a COMMAND test_bolt_fintech_phase4a)

add_executable(test_bolt_fintech_phase4b test_bolt_fintech_phase4b.cpp)

target_link_libraries(test_bolt_fintech_phase4b
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase4b)

add_test(NAME test_bolt_fintech_phase4b COMMAND test_bolt_fintech_phase4b)

add_executable(test_bolt_fintech_phase5r test_bolt_fintech_phase5r.cpp)

target_link_libraries(test_bolt_fintech_phase5r
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5r)

add_test(NAME test_bolt_fintech_phase5r COMMAND test_bolt_fintech_phase5r)

add_executable(test_bolt_fintech_phase3 test_bolt_fintech_phase3.cpp)

target_link_libraries(test_bolt_fintech_phase3
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase3)

add_test(NAME test_bolt_fintech_phase3 COMMAND test_bolt_fintech_phase3)

add_executable(test_bolt_fintech_phase5e test_bolt_fintech_phase5e.cpp)

target_link_libraries(test_bolt_fintech_phase5e
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5e)

add_test(NAME test_bolt_fintech_phase5e COMMAND test_bolt_fintech_phase5e)

add_executable(test_bolt_fintech_phase5w test_bolt_fintech_phase5w.cpp)

target_link_libraries(test_bolt_fintech_phase5w
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5w)

add_test(NAME test_bolt_fintech_phase5w COMMAND test_bolt_fintech_phase5w)

add_executable(test_bolt_fintech_phase5s_sorted test_bolt_fintech_phase5s_sorted.cpp)

target_link_libraries(test_bolt_fintech_phase5s_sorted
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_sorted)

add_test(NAME test_bolt_fintech_phase5s_sorted COMMAND test_bolt_fintech_phase5s_sorted)

add_executable(test_bolt_fintech_phase5s_core test_bolt_fintech_phase5s_core.cpp)

target_link_libraries(test_bolt_fintech_phase5s_core
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_core)

add_test(NAME test_bolt_fintech_phase5s_core COMMAND test_bolt_fintech_phase5s_core)

add_executable(test_bolt_fintech_phase5s_kalman test_bolt_fintech_phase5s_kalman.cpp)

target_link_libraries(test_bolt_fintech_phase5s_kalman
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_kalman)

add_test(NAME test_bolt_fintech_phase5s_kalman COMMAND test_bolt_fintech_phase5s_kalman)

add_executable(test_bolt_wave2 test_bolt_wave2.cpp)

target_link_libraries(test_bolt_wave2
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wave2)

add_test(NAME test_bolt_wave2 COMMAND test_bolt_wave2)

add_executable(test_bolt_wire test_bolt_wire.cpp)

target_link_libraries(test_bolt_wire
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wire)

add_test(NAME test_bolt_wire COMMAND test_bolt_wire)

add_executable(test_bolt_wire_stream test_bolt_wire_stream.cpp)

target_link_libraries(test_bolt_wire_stream
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wire_stream)

add_test(NAME test_bolt_wire_stream COMMAND test_bolt_wire_stream)

add_executable(test_bolt_kmerge test_bolt_kmerge.cpp)

target_link_libraries(test_bolt_kmerge
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_kmerge)

add_test(NAME test_bolt_kmerge COMMAND test_bolt_kmerge)

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

add_executable(test_bolt_ebr test_bolt_ebr.cpp)

target_link_libraries(test_bolt_ebr
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_ebr)

add_test(NAME test_bolt_ebr COMMAND test_bolt_ebr)

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

if(BOLT_BUILD_IO)
    add_executable(test_bolt_binsearch_crc32c test_bolt_binsearch_crc32c.cpp)
    target_link_libraries(test_bolt_binsearch_crc32c
        PRIVATE bolt::bolt bolt::io GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_binsearch_crc32c)
    add_test(NAME test_bolt_binsearch_crc32c COMMAND test_bolt_binsearch_crc32c)
endif()

# Layout settings must travel to installed consumers, unlike tuning flags.
set(BOLT_CACHE_ISOLATION_BYTES "AUTO" CACHE STRING "Shared-state isolation: AUTO, 64 or 128 bytes")
set_property(CACHE BOLT_CACHE_ISOLATION_BYTES PROPERTY STRINGS AUTO 64 128)
set(BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES "64" CACHE STRING "Channel slot alignment: 64 or 128 bytes")
set_property(CACHE BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES PROPERTY STRINGS 64 128)
if(NOT BOLT_CACHE_ISOLATION_BYTES MATCHES "^(AUTO|64|128)$")
    message(FATAL_ERROR "BOLT_CACHE_ISOLATION_BYTES must be AUTO, 64 or 128")
endif()
if(NOT BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES MATCHES "^(64|128)$")
    message(FATAL_ERROR "BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES must be 64 or 128")
endif()
if(BOLT_CACHE_ISOLATION_BYTES STREQUAL "AUTO")
    set(_bolt_isolation 0) # Compiler target, never configure-host detection.
else()
    set(_bolt_isolation ${BOLT_CACHE_ISOLATION_BYTES})
endif()
target_compile_definitions(bolt_core INTERFACE
    BOLT_CONFIGURED_CACHE_ISOLATION_BYTES=${_bolt_isolation}
    BOLT_CONFIGURED_CHANNEL_SLOT_ALIGNMENT_BYTES=${BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES})
message(STATUS "Bolt layout: isolation=${BOLT_CACHE_ISOLATION_BYTES}, channel slots=${BOLT_CHANNEL_SLOT_ALIGNMENT_BYTES}")
unset(_bolt_isolation)

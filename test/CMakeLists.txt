project (n8tigall_tests)

add_executable (
   gas_consumption
   gas_consumption.cpp
)

TARGET_LINK_LIBRARIES(gas_consumption n8tigall)
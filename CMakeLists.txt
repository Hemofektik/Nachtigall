cmake_minimum_required(VERSION 3.3)

project (N8igall)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/bin")

add_definitions( -DLITTLE_ENDIAN=1 )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/3rdparty/")

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /openmp")

	add_definitions( -DNOMINMAX )
else()

    #set(CMAKE_BUILD_TYPE Debug)

    # Set C++11 feature set for both GCC and Clang 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Debug symbols for GCC
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

endif()


include_directories(
	${CMAKE_SOURCE_DIR}/3rdparty/zlib
	${CMAKE_SOURCE_DIR}/3rdparty/zlib/contrib/minizip
	${CMAKE_BINARY_DIR}/3rdparty/zlib
	${CMAKE_SOURCE_DIR}/3rdparty/fast-cpp-csv-parser
	${CMAKE_SOURCE_DIR}/3rdparty/date
)


add_subdirectory(3rdparty/zlib EXCLUDE_FROM_ALL)

set  (THIRDPARTY
		zlib)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
	set_property(TARGET ${THIRDPARTY} PROPERTY FOLDER "3rdparty")
endif()

link_libraries (
	${THIRDPARTY}
)


if(MSVC)
else()
link_libraries (
	stdc++fs
)
endif()

add_subdirectory(src)
add_subdirectory(test)


if(CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
	set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
		"Reset the configurations to what we need"
		FORCE)
endif()

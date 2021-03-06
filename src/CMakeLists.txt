project (n8igall)
file(GLOB_RECURSE SOURCES *.cpp *.cc *.h *.hpp *.c)
SET(SOURCES ${SOURCES} ${CMAKE_SOURCE_DIR}/3rdparty/zlib/contrib/minizip/unzip.c)
SET(SOURCES ${SOURCES} ${CMAKE_SOURCE_DIR}/3rdparty/zlib/contrib/minizip/ioapi.c)
SET(SOURCES ${SOURCES} ${CMAKE_SOURCE_DIR}/3rdparty/zlib/contrib/minizip/iowin32.c)

foreach(f ${SOURCES})
    # Get the path of the file relative to ${DIRECTORY},
    # then alter it (not compulsory)
    file(RELATIVE_PATH SRCGR ${CMAKE_CURRENT_SOURCE_DIR} ${f})
    set(SRCGR "Sources/${SRCGR}")

    # Extract the folder, ie remove the filename part
    string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

    # Source_group expects \\ (double antislash), not / (slash)
    string(REPLACE / \\ SRCGR ${SRCGR})
    source_group("${SRCGR}" FILES ${f})
endforeach()

add_library (
   n8igall SHARED
#   WIN32 # Only if you don't want the DOS prompt to appear in the background in Windows
#   MACOSX_BUNDLE
   ${SOURCES} # We could've listed the source files here directly instead of using a variable to store them
   ${ADDITIONAL_SOURCES}  
   #${INCLUDES}
)

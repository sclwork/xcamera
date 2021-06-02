# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("camera")

SET(CMAKE_BUILD_TYPE "Release")
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wno-unused-variable -pthread")
elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -pthread")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -pthread")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(opencv4 SHARED IMPORTED)
set_target_properties(opencv4 PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libopencv_java4.so)

add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libMNN.so)
add_library(mnn_expr SHARED IMPORTED)
set_target_properties(mnn_expr PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libMNN_Express.so)
add_library(mnn_cl SHARED IMPORTED)
set_target_properties(mnn_cl PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libMNN_CL.so)
add_library(mnn_gl SHARED IMPORTED)
set_target_properties(mnn_gl PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libMNN_GL.so)
add_library(mnn_vulkan SHARED IMPORTED)
set_target_properties(mnn_vulkan PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libMNN_Vulkan.so)

add_library(avcodec SHARED IMPORTED)
set_target_properties(avcodec PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libavcodec.so)
add_library(avdevice SHARED IMPORTED)
set_target_properties(avdevice PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libavdevice.so)
add_library(avfilter SHARED IMPORTED)
set_target_properties(avfilter PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libavfilter.so)
add_library(avformat SHARED IMPORTED)
set_target_properties(avformat PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libavformat.so)
add_library(avutil SHARED IMPORTED)
set_target_properties(avutil PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libavutil.so)
add_library(swresample SHARED IMPORTED)
set_target_properties(swresample PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libswresample.so)
add_library(swscale SHARED IMPORTED)
set_target_properties(swscale PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libswscale.so)
add_library(postproc SHARED IMPORTED)
set_target_properties(postproc PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libpostproc.so)

include_directories(
        src/main/cpp/include )

# yuv
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/jniLibs/${ANDROID_ABI})
aux_source_directory(src/main/cpp/yuv YUV_SRC_LIST)

add_library( # Sets the name of the library.
        libyuv

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${YUV_SRC_LIST})

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

target_link_libraries( # Specifies the target library.
        libyuv

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

aux_source_directory(src/main/cpp SRC_LIST)

add_library( # Sets the name of the library.
        xcamera-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${SRC_LIST} )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

find_library( # Sets the name of the path variable.
        opensl-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        OpenSLES )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        xcamera-lib

        # ffmpeg
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale
        postproc
        # opencv
        opencv4
        # mnn
        mnn
        mnn_expr
        mnn_cl
        mnn_gl
        mnn_vulkan
        # yuv
        libyuv
        # android ndk library
        android
        camera2ndk
        mediandk
        GLESv3
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${opensl-lib} )

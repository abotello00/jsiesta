cmake_minimum_required(VERSION 3.20)
project(jsiesta_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(JSIESTA_USE_LIBXC "Link against external libxc if available" OFF)
option(JSIESTA_ENABLE_CUDA "Build CUDA FFI kernels" OFF)

if(NOT XLA_FFI_INCLUDE_DIR)
    foreach(_py_cand
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python"
        "$ENV{VIRTUAL_ENV}/bin/python"
        "python3"
    )
        if(_py_cand AND NOT XLA_FFI_INCLUDE_DIR)
            execute_process(
                COMMAND "${_py_cand}" -c "import jaxlib, pathlib; print(pathlib.Path(jaxlib.__file__).parent / 'include')"
                OUTPUT_VARIABLE _probe_inc
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(EXISTS "${_probe_inc}/xla/ffi/api/ffi.h")
                set(XLA_FFI_INCLUDE_DIR "${_probe_inc}")
            endif()
        endif()
    endforeach()
endif()

if(NOT EXISTS "${XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "Could not locate xla/ffi/api/ffi.h in ${XLA_FFI_INCLUDE_DIR}")
endif()

add_library(jsiesta_ffi SHARED
    src/ffi_registry.cpp
    src/two_center.cpp
    src/grid_ffi.cpp
    src/xc_ffi.cpp
    src/kb_three_center.cpp
    src/eri.cpp
    src/hfx.cpp
)

target_include_directories(jsiesta_ffi
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(jsiesta_ffi SYSTEM
    PUBLIC
        "${XLA_FFI_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)
target_compile_options(jsiesta_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(jsiesta_ffi PRIVATE Threads::Threads)

if(JSIESTA_USE_LIBXC)
    # LIBXC_ROOT (variable or environment) takes precedence over a Libxc found elsewhere. Releases 5.1 to 7.x work.
    find_package(Libxc CONFIG QUIET HINTS "${LIBXC_ROOT}" "$ENV{LIBXC_ROOT}")
    if(Libxc_FOUND)
        target_compile_definitions(jsiesta_ffi PRIVATE JSIESTA_HAS_LIBXC=1)
        target_link_libraries(jsiesta_ffi PRIVATE Libxc::xc)
        message(STATUS "Linked Libxc ${Libxc_VERSION} from ${Libxc_DIR}")
    else()
        find_path(LIBXC_INCLUDE_DIR
            NAMES xc.h
            HINTS "$ENV{LIBXC_ROOT}/include" "${LIBXC_ROOT}/include" "/opt/etsf/include" "/usr/local/include"
        )
        find_library(LIBXC_LIBRARY
            NAMES libxc.a xc
            HINTS "$ENV{LIBXC_ROOT}/lib" "${LIBXC_ROOT}/lib" "/opt/etsf/lib" "/usr/local/lib"
        )
        if(LIBXC_INCLUDE_DIR AND LIBXC_LIBRARY)
            target_compile_definitions(jsiesta_ffi PRIVATE JSIESTA_HAS_LIBXC=1)
            target_include_directories(jsiesta_ffi SYSTEM PRIVATE "${LIBXC_INCLUDE_DIR}")
            target_link_libraries(jsiesta_ffi PRIVATE "${LIBXC_LIBRARY}")
            message(STATUS "Linked external libxc: include=${LIBXC_INCLUDE_DIR} lib=${LIBXC_LIBRARY}")
        else()
            message(WARNING "JSIESTA_USE_LIBXC=ON requested, but Libxc was not found; using built-in C++ XC kernels.")
        endif()
    endif()
endif()

if(JSIESTA_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(jsiesta_ffi PRIVATE cuda/rhoofd_vmat.cu)
    target_compile_definitions(jsiesta_ffi PRIVATE JSIESTA_HAS_CUDA=1)
    target_link_libraries(jsiesta_ffi PRIVATE CUDA::cudart)
    set_target_properties(jsiesta_ffi PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

set_target_properties(jsiesta_ffi PROPERTIES
    OUTPUT_NAME "jsiesta_ffi"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

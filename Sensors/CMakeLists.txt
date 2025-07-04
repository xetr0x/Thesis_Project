# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.0)
set(toolchainVersion 13_3_Rel1)
set(picotoolVersion 2.1.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(test1 C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1
if (TARGET tinyusb_device)
    add_executable(test1
            test1.c
            sht31.c
            sgp40.c
            tsl.c
            scd30.c
            sensirion.c
            sensirion_gas_index_algorithm.c
            )

    # pull in common dependencies
    # enable usb output, disable uart output
    pico_enable_stdio_usb(test1 1)
    pico_enable_stdio_uart(test1 0)

    # create map/bin/hex/uf2 file etc.
    pico_add_extra_outputs(test1)

    # add url via pico_set_program_url
elseif(PICO_ON_DEVICE)
    pico_enable_stdio_uart(test1 0)
    pico_enable_stdio_usb(test1 1)
    add_executable(test1 test1.c sht31.c sgp40.c tsl.c scd30.c sensirion.c sensirion_gas_index_algorithm.c)
endif()

pico_set_program_name(test1 "test1")
pico_set_program_version(test1 "0.1")

# Modify the below lines to enable/disable output over UART/USB

# Add the standard library to the build
target_link_libraries(test1
        hardware_i2c
        pico_stdlib)

# Add the standard include files to the build
target_include_directories(test1 PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(test1)


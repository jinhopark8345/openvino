# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ie_generate_dev_package_config)
    # dummy check that OpenCV is here
    find_package(OpenCV QUIET)

    set(all_dev_targets gflags ov_runtime_libraries)
    foreach(component IN LISTS openvino_export_components)
        # export all targets with prefix and use them during extra modules build
        export(TARGETS ${${component}} NAMESPACE IE::
            APPEND FILE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
        list(APPEND all_dev_targets ${${component}})
    endforeach()
    add_custom_target(ie_dev_targets DEPENDS ${all_dev_targets})

    configure_package_config_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineDeveloperPackageConfig.cmake.in"
                                  "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
                                  INSTALL_DESTINATION share # not used
                                  PATH_VARS "OpenVINO_SOURCE_DIR;gflags_BINARY_DIR"
                                  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    configure_file("${OpenVINO_SOURCE_DIR}/cmake/templates/InferenceEngineConfig-version.cmake.in"
                   "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
                   @ONLY)
endfunction()

#
# Add extra modules
#

function(register_extra_modules)
    # post export
    openvino_developer_export_targets(COMPONENT core TARGETS inference_engine)
    openvino_developer_export_targets(COMPONENT core TARGETS ngraph)

    set(InferenceEngineDeveloperPackage_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime")

    function(generate_fake_dev_package)
        set(iedevconfig_file "${InferenceEngineDeveloperPackage_DIR}/InferenceEngineDeveloperPackageConfig.cmake")
        file(REMOVE "${iedevconfig_file}")

        file(WRITE "${iedevconfig_file}" "\# !! AUTOGENERATED: DON'T EDIT !!\n\n")
        file(APPEND "${iedevconfig_file}" "ie_deprecated_no_errors()\n")

        foreach(target IN LISTS ${openvino_export_components})
            if(target)
                file(APPEND "${iedevconfig_file}" "add_library(IE::${target} ALIAS ${target})\n")
            endif()
        endforeach()
    endfunction()

    generate_fake_dev_package()

    # automatically import plugins from the 'runtime/plugins' folder
    file(GLOB local_extra_modules "runtime/plugins/*")
    # add template plugin
    if(ENABLE_TEMPLATE)
        list(APPEND local_extra_modules "${OpenVINO_SOURCE_DIR}/docs/template_plugin")
    endif()

    # detect where IE_EXTRA_MODULES contains folders with CMakeLists.txt
    # other folders are supposed to have sub-folders with CMakeLists.txt
    foreach(module_path IN LISTS IE_EXTRA_MODULES)
        if(EXISTS "${module_path}/CMakeLists.txt")
            list(APPEND extra_modules "${module_path}")
        elseif(module_path)
            file(GLOB extra_modules ${extra_modules} "${module_path}/*")
        endif()
    endforeach()

    # add each extra module
    foreach(module_path IN LISTS extra_modules local_extra_modules)
        if(module_path)
            get_filename_component(module_name "${module_path}" NAME)
            set(build_module ON)
            if(NOT EXISTS "${module_path}/CMakeLists.txt") # if module is built not using cmake
                set(build_module OFF)
            endif()
            if(NOT DEFINED BUILD_${module_name})
                set(BUILD_${module_name} ${build_module} CACHE BOOL "Build ${module_name} extra module" FORCE)
            endif()
            if(BUILD_${module_name})
                message(STATUS "Register ${module_name} to be built in build-modules/${module_name}")
                add_subdirectory("${module_path}" "build-modules/${module_name}")
            endif()
        endif()
    endforeach()
endfunction()

#
# Extra modules support
#

# this InferenceEngineDeveloperPackageConfig.cmake is not used
# during extra modules build since it's generated after modules
# are configured
ie_generate_dev_package_config()

# extra modules must be registered after inference_engine library
# and all other IE common libraries (ov_runtime_libraries) are creared
# because 'register_extra_modules' creates fake InferenceEngineDeveloperPackageConfig.cmake
# with all imported developer targets
register_extra_modules()

# for static libraries case we need to generate final ie_plugins.hpp
# with all the information about plugins
ie_generate_plugins_hpp()

# used for static build
ov_generate_frontends_hpp()

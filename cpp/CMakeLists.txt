cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_pose_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 设置RKNN API路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/Linux/librknn_api)

# 检测编译器架构
if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

# 设置LIBRKNNRT变量
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(LIBRKNNRT ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  # 如果在系统路径中找不到librknnrt.so，则使用项目根目录下的库
  find_library(LIBRKNNRT_SYSTEM rknnrt)
  if(LIBRKNNRT_SYSTEM)
    set(LIBRKNNRT ${LIBRKNNRT_SYSTEM})
  else()
    set(LIBRKNNRT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../librknnrt.so)
  endif()
endif()

# 设置LIBRKNNRT_INCLUDES变量
set(LIBRKNNRT_INCLUDES ${RKNN_API_PATH}/include)
include_directories(${LIBRKNNRT_INCLUDES})

# 添加fp16目录的包含路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/fp16)

# 使用系统的OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(rknpu_yolov8-pose_file rknpu2/yolov8-pose.cc)
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
    set(rknpu_yolov8-pose_file rknpu2/yolov8-pose_rv1106_1103.cc)
    #dma
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
elseif(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    add_definitions(-DRKNPU1)
    set(rknpu_yolov8-pose_file rknpu1/yolov8-pose.cc)
endif()

# 禁用音频处理库，避免编译错误
set(DISABLE_AUDIO_UTILS ON)
# 禁用RGA，使用OpenCV进行图像处理
set(DISABLE_RGA ON)

# 修改为相对于rknpu2/examples的路径
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# 原始的图像处理demo
add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    ${rknpu_yolov8-pose_file}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

# 添加摄像头实时处理demo
add_executable(rknn_yolov8_pose_camera_demo
    main_camera.cc
    postprocess.cc
    ${rknpu_yolov8-pose_file}
)

target_link_libraries(rknn_yolov8_pose_camera_demo
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    dl
)

# 添加零拷贝优化版本demo
add_executable(rknn_yolov8_pose_camera_optimized
    main_camera_optimized.cc
    postprocess.cc
    ${rknpu_yolov8-pose_file}
)

target_link_libraries(rknn_yolov8_pose_camera_optimized
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    dl
)

# 添加纯性能测试版本demo
add_executable(rknn_yolov8_pose_benchmark
    main_camera_benchmark.cc
    postprocess.cc
    ${rknpu_yolov8-pose_file}
)

target_link_libraries(rknn_yolov8_pose_benchmark
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
    target_link_libraries(rknn_yolov8_pose_camera_demo
    log
)
    target_link_libraries(rknn_yolov8_pose_camera_optimized
    log
)
    target_link_libraries(rknn_yolov8_pose_benchmark
    log
)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # 直接链接pthread库
    target_link_libraries(${PROJECT_NAME} pthread)
    target_link_libraries(rknn_yolov8_pose_camera_demo pthread)
    target_link_libraries(rknn_yolov8_pose_camera_optimized pthread)
    target_link_libraries(rknn_yolov8_pose_benchmark pthread)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

target_include_directories(rknn_yolov8_pose_camera_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(rknn_yolov8_pose_camera_optimized PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(rknn_yolov8_pose_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

install(TARGETS ${PROJECT_NAME} rknn_yolov8_pose_camera_demo rknn_yolov8_pose_camera_optimized rknn_yolov8_pose_benchmark DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION ./model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/yolov8_pose_labels_list.txt DESTINATION ./model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)

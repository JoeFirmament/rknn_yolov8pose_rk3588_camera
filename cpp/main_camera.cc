// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <memory>

// OpenCV头文件
#include <opencv2/opencv.hpp>

#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

int skeleton[38] ={16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}; 

// 全局变量，用于信号处理
bool g_running = true;

// 用于坐标变换的扩展信息
typedef struct {
    int x_offset;
    int y_offset;
    int width;
    int height;
    float scale;
} letterbox_ext_t;

// 信号处理函数
void sig_handler(int signo) {
    if (signo == SIGINT) {
        printf("Received SIGINT. Exiting...\n");
        g_running = false;
    }
}

// 计算FPS的辅助函数
static float get_fps(int64_t start_time, int64_t end_time) {
    return 1000000.0 / (end_time - start_time);
}

// 获取当前时间（微秒）
static int64_t getCurrentTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// 将OpenCV Mat转换为image_buffer_t
static int mat_to_image_buffer(cv::Mat& mat, image_buffer_t* img) {
    if (mat.empty() || !img) {
        printf("Error: Empty matrix or null image buffer\n");
        return -1;
    }

    printf("Debug: Input image size: %dx%d, channels: %d, type: %d\n", 
           mat.cols, mat.rows, mat.channels(), mat.type());

    img->width = mat.cols;
    img->height = mat.rows;
    img->width_stride = mat.cols;
    img->height_stride = mat.rows;
    img->format = IMAGE_FORMAT_RGB888;
    img->size = mat.cols * mat.rows * 3;
    
    // 分配内存 - 确保16字节对齐以满足RGA要求
    size_t alignment = 16;
    size_t size = img->size;
    size_t padding = (alignment - (size % alignment)) % alignment;
    img->virt_addr = (unsigned char*)malloc(size + padding);
    
    if (!img->virt_addr) {
        printf("Error: Failed to allocate memory for image buffer\n");
        return -1;
    }

    // 转换颜色空间（如果需要）
    if (mat.channels() == 3 && mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        memcpy(img->virt_addr, rgb.data, img->size);
        printf("Debug: Converted BGR to RGB\n");
    } else if (mat.channels() == 1) {
        // 灰度图转RGB
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
        memcpy(img->virt_addr, rgb.data, img->size);
        printf("Debug: Converted GRAY to RGB\n");
    } else {
        // 不支持的格式
        printf("Error: Unsupported image format: channels=%d, type=%d\n", 
               mat.channels(), mat.type());
        free(img->virt_addr);
        img->virt_addr = nullptr;
        return -1;
    }

    printf("Debug: Successfully created image buffer: %dx%d, format=%d, size=%d\n", 
           img->width, img->height, img->format, img->size);
    return 0;
}

// 手动实现letterbox操作，不依赖RGA
static int manual_letterbox(image_buffer_t* src, image_buffer_t* dst, letterbox_t* letter_box, letterbox_ext_t* ext_info) {
    if (!src || !dst || !letter_box || !ext_info || !src->virt_addr || !dst->virt_addr) {
        printf("Error: Invalid parameters for manual_letterbox\n");
        return -1;
    }
    
    printf("Debug: Manual letterbox: src=%dx%d, dst=%dx%d\n", 
           src->width, src->height, dst->width, dst->height);
    
    // 计算缩放比例和偏移
    float scale_w = (float)dst->width / src->width;
    float scale_h = (float)dst->height / src->height;
    float scale = std::min(scale_w, scale_h);
    
    int new_width = (int)(src->width * scale);
    int new_height = (int)(src->height * scale);
    int offset_x = (dst->width - new_width) / 2;
    int offset_y = (dst->height - new_height) / 2;
    
    // 填充letterbox_t结构体 (官方API需要的)
    letter_box->scale = scale;
    letter_box->x_pad = offset_x;
    letter_box->y_pad = offset_y;
    
    // 填充扩展信息 (我们自己需要的)
    ext_info->scale = scale;
    ext_info->x_offset = offset_x;
    ext_info->y_offset = offset_y;
    ext_info->width = new_width;
    ext_info->height = new_height;
    
    printf("Debug: Letterbox params: scale=%.2f, offset=(%d,%d), new_size=%dx%d\n",
           scale, offset_x, offset_y, new_width, new_height);
    
    // 创建OpenCV矩阵
    cv::Mat src_mat(src->height, src->width, CV_8UC3, src->virt_addr);
    cv::Mat dst_mat(dst->height, dst->width, CV_8UC3, dst->virt_addr);
    
    // 填充背景色 (114,114,114)
    dst_mat.setTo(cv::Scalar(114, 114, 114));
    
    // 调整大小
    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(new_width, new_height));
    
    // 将调整大小后的图像复制到目标的中心位置
    cv::Mat roi = dst_mat(cv::Rect(offset_x, offset_y, new_width, new_height));
    resized.copyTo(roi);
    
    printf("Debug: Manual letterbox completed successfully\n");
    return 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("%s <model_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // 注册信号处理函数
    signal(SIGINT, sig_handler);

    // 初始化后处理
    init_post_process();

    // 初始化模型
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_pose_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    printf("Debug: Model initialized successfully, input size: %dx%d\n", 
           rknn_app_ctx.model_width, rknn_app_ctx.model_height);

    // 打开摄像头
    cv::VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        printf("Failed to open camera\n");
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FPS, 30);

    // 获取实际的摄像头参数
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    int actual_fourcc = cap.get(cv::CAP_PROP_FOURCC);
    
    printf("Camera opened successfully with parameters:\n");
    printf("Width: %d, Height: %d, FPS: %.1f, FourCC: %c%c%c%c\n",
           actual_width, actual_height, actual_fps,
           (actual_fourcc & 0xFF),
           ((actual_fourcc >> 8) & 0xFF),
           ((actual_fourcc >> 16) & 0xFF),
           ((actual_fourcc >> 24) & 0xFF));
    printf("Press Ctrl+C to exit\n");

    // 创建窗口
    cv::namedWindow("YOLOv8 Pose", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLOv8 Pose", 640, 480);

    // 主循环
    while (g_running) {
        cv::Mat frame;
        int64_t start_time = getCurrentTimeUs();
        
        // 捕获一帧
        if (!cap.read(frame)) {
            printf("Failed to read frame\n");
            break;
        }

        // 将OpenCV Mat转换为image_buffer_t
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = mat_to_image_buffer(frame, &src_image);
        if (ret != 0) {
            printf("Failed to convert Mat to image_buffer\n");
            continue;
        }

        // 创建目标图像缓冲区（模型输入大小）
        image_buffer_t dst_image;
        letterbox_t letter_box;
        letterbox_ext_t letter_box_ext;
        memset(&dst_image, 0, sizeof(image_buffer_t));
        memset(&letter_box, 0, sizeof(letterbox_t));
        memset(&letter_box_ext, 0, sizeof(letterbox_ext_t));
        
        dst_image.width = rknn_app_ctx.model_width;
        dst_image.height = rknn_app_ctx.model_height;
        dst_image.format = IMAGE_FORMAT_RGB888;
        dst_image.size = dst_image.width * dst_image.height * 3;
        dst_image.virt_addr = (unsigned char*)malloc(dst_image.size);
        
        if (!dst_image.virt_addr) {
            printf("Failed to allocate memory for destination image\n");
            free(src_image.virt_addr);
            continue;
        }

        // 尝试使用手动letterbox而不是RGA
        ret = manual_letterbox(&src_image, &dst_image, &letter_box, &letter_box_ext);
        if (ret != 0) {
            printf("Failed to perform manual letterbox\n");
            free(src_image.virt_addr);
            free(dst_image.virt_addr);
            continue;
        }

        // 打印letterbox信息
        printf("Debug: letterbox info - scale=%.3f, x_pad=%d, y_pad=%d\n", 
               letter_box.scale, letter_box.x_pad, letter_box.y_pad);
        printf("Debug: letterbox_ext info - scale=%.3f, x_offset=%d, y_offset=%d, width=%d, height=%d\n", 
               letter_box_ext.scale, letter_box_ext.x_offset, letter_box_ext.y_offset, 
               letter_box_ext.width, letter_box_ext.height);

        // 目标检测结果
        object_detect_result_list od_results;

        // 推理 - 使用预处理后的图像
        ret = inference_yolov8_pose_model(&rknn_app_ctx, &dst_image, &od_results);
        if (ret != 0) {
            printf("inference_yolov8_pose_model fail! ret=%d\n", ret);
            free(src_image.virt_addr);
            free(dst_image.virt_addr);
            continue;
        }

        printf("Debug: Detected %d objects\n", od_results.count);

        // 转回OpenCV格式以便显示
        cv::Mat result_frame;
        cv::cvtColor(frame, result_frame, cv::COLOR_BGR2RGB);

        // 画框和概率
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            
            // 将检测框从模型输入坐标转换回原始图像坐标
            float scale = letter_box_ext.scale;
            int x_offset = letter_box_ext.x_offset;
            int y_offset = letter_box_ext.y_offset;
            
            int x1 = (det_result->box.left - x_offset) / scale;
            int y1 = (det_result->box.top - y_offset) / scale;
            int x2 = (det_result->box.right - x_offset) / scale;
            int y2 = (det_result->box.bottom - y_offset) / scale;
            
            // 确保坐标在有效范围内
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));

            printf("Debug: Original box: (%d,%d,%d,%d), Transformed box: (%d,%d,%d,%d)\n",
                   det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                   x1, y1, x2, y2);

            // 画矩形框
            cv::rectangle(result_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);

            // 显示类别和置信度
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            cv::putText(result_frame, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            // 转换关键点坐标并绘制
            float keypoints[17][2];
            for (int j = 0; j < 17; j++) {
                keypoints[j][0] = (det_result->keypoints[j][0] - x_offset) / scale;
                keypoints[j][1] = (det_result->keypoints[j][1] - y_offset) / scale;
                
                // 确保关键点坐标在有效范围内
                keypoints[j][0] = std::max(0.0f, std::min(keypoints[j][0], (float)(frame.cols - 1)));
                keypoints[j][1] = std::max(0.0f, std::min(keypoints[j][1], (float)(frame.rows - 1)));
            }
            
            // 画骨架
            for (int j = 0; j < 38/2; ++j) {
                cv::line(result_frame, 
                    cv::Point((int)(keypoints[skeleton[2*j]-1][0]), (int)(keypoints[skeleton[2*j]-1][1])),
                    cv::Point((int)(keypoints[skeleton[2*j+1]-1][0]), (int)(keypoints[skeleton[2*j+1]-1][1])),
                    cv::Scalar(255, 128, 0), 2);
            }
            
            // 画关键点
            for (int j = 0; j < 17; ++j) {
                cv::circle(result_frame, 
                    cv::Point((int)(keypoints[j][0]), (int)(keypoints[j][1])),
                    3, cv::Scalar(255, 255, 0), -1);
            }
        }

        // 计算FPS
        int64_t end_time = getCurrentTimeUs();
        float fps = get_fps(start_time, end_time);
        
        // 显示FPS
        sprintf(text, "FPS: %.1f", fps);
        cv::putText(result_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // 显示结果
        cv::cvtColor(result_frame, result_frame, cv::COLOR_RGB2BGR);
        cv::imshow("YOLOv8 Pose", result_frame);

        // 释放资源
        free(src_image.virt_addr);
        free(dst_image.virt_addr);

        // 等待按键
        int key = cv::waitKey(1);
        if (key == 27) { // ESC键
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
} 
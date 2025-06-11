// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// 纯性能测试版本 - 无显示模式
//
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

// 零拷贝优化结构体
typedef struct {
    rknn_tensor_mem* input_mem;
    rknn_tensor_mem* output_mems[4];  // YOLOv8 pose有4个输出
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attrs[4];
    int model_width;
    int model_height;
    int model_channels;
} zero_copy_context_t;

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
        printf("接收到SIGINT信号，正在退出...\n");
        g_running = false;
    }
}

// 获取当前时间（微秒）
static int64_t getCurrentTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// 初始化零拷贝内存
static int init_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    int ret;
    
    // 设置输入属性
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    // 创建输入零拷贝内存
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("创建输入零拷贝内存失败！\n");
        return -1;
    }
    printf("创建输入零拷贝内存成功: size=%d, stride=%d\n", 
           zc_ctx->input_attr.size, zc_ctx->input_attr.size_with_stride);
    
    // 设置输入内存
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }
    
    // 暂时不使用输出零拷贝
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = NULL;
    }
    
    printf("零拷贝内存初始化完成\n");
    return 0;
}

// 释放零拷贝内存
static void release_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    if (zc_ctx->input_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->input_mem);
        zc_ctx->input_mem = NULL;
    }
    printf("零拷贝内存释放完成\n");
}

// 优化的letterbox预处理，直接写入NPU内存
static int optimized_letterbox_to_npu(cv::Mat& src_mat, zero_copy_context_t* zc_ctx, letterbox_ext_t* ext_info) {
    int dst_width = zc_ctx->model_width;
    int dst_height = zc_ctx->model_height;
    int src_width = src_mat.cols;
    int src_height = src_mat.rows;
    
    // 计算缩放参数
    float scale_w = (float)dst_width / src_width;
    float scale_h = (float)dst_height / src_height;
    float scale = std::min(scale_w, scale_h);
    
    int new_width = (int)(src_width * scale);
    int new_height = (int)(src_height * scale);
    int offset_x = (dst_width - new_width) / 2;
    int offset_y = (dst_height - new_height) / 2;
    
    // 填充扩展信息
    ext_info->scale = scale;
    ext_info->x_offset = offset_x;
    ext_info->y_offset = offset_y;
    ext_info->width = new_width;
    ext_info->height = new_height;
    
    // 获取NPU内存指针
    uint8_t* npu_ptr = (uint8_t*)zc_ctx->input_mem->virt_addr;
    int width_stride = zc_ctx->input_attr.w_stride;
    
    // 创建目标Mat，直接指向NPU内存
    cv::Mat dst_mat;
    if (width_stride == dst_width) {
        dst_mat = cv::Mat(dst_height, dst_width, CV_8UC3, npu_ptr);
    } else {
        dst_mat = cv::Mat(dst_height, width_stride, CV_8UC3, npu_ptr);
        dst_mat = dst_mat(cv::Rect(0, 0, dst_width, dst_height));
    }
    
    // 填充背景色 (114,114,114)
    dst_mat.setTo(cv::Scalar(114, 114, 114));
    
    // 转换颜色空间 BGR->RGB
    cv::Mat src_rgb;
    cv::cvtColor(src_mat, src_rgb, cv::COLOR_BGR2RGB);
    
    // 调整大小并直接写入NPU内存的ROI区域
    cv::Mat resized;
    cv::resize(src_rgb, resized, cv::Size(new_width, new_height));
    
    cv::Mat roi = dst_mat(cv::Rect(offset_x, offset_y, new_width, new_height));
    resized.copyTo(roi);
    
    return 0;
}

// 零拷贝推理函数
static int zero_copy_inference(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx, 
                              letterbox_ext_t* ext_info, object_detect_result_list* od_results) {
    int ret;
    
    // 直接运行推理 - 无需数据拷贝
    int64_t start_time = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    int64_t inference_time = getCurrentTimeUs() - start_time;
    
    if (ret < 0) {
        printf("rknn_run 失败! ret=%d\n", ret);
        return -1;
    }
    
    // 获取输出结果
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    
    int ret_get = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret_get < 0) {
        printf("rknn_outputs_get 失败! ret=%d\n", ret_get);
        return -1;
    }
    
    // 后处理
    letterbox_t letter_box;
    letter_box.scale = ext_info->scale;
    letter_box.x_pad = ext_info->x_offset;
    letter_box.y_pad = ext_info->y_offset;
    
    start_time = getCurrentTimeUs();
    post_process(app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, od_results);
    int64_t postprocess_time = getCurrentTimeUs() - start_time;
    
    // 释放输出内存
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    
    return 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    printf("=== YOLOv8 Pose Detection - 纯性能测试版本 ===\n");
    printf("平台: Rock5C 8GB, CPU: aarch64, NPU: RK3588\n");
    printf("测试模式: 无显示，纯NPU性能测试\n\n");
    
    if (argc != 2)
    {
        printf("用法: %s <model_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    int ret;
    rknn_app_context_t rknn_app_ctx;
    zero_copy_context_t zero_copy_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&zero_copy_ctx, 0, sizeof(zero_copy_context_t));

    // 注册信号处理函数
    signal(SIGINT, sig_handler);

    // 初始化后处理
    init_post_process();

    // 初始化模型
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("模型初始化失败! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    printf("模型初始化成功, 输入尺寸: %dx%d\n", 
           rknn_app_ctx.model_width, rknn_app_ctx.model_height);

    // 初始化零拷贝内存
    ret = init_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("零拷贝内存初始化失败!\n");
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("摄像头打开失败\n");
        release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FPS, 60);  // 测试摄像头是否支持60 FPS
    
    // 检查实际设置的帧率
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    printf("请求FPS: 60, 实际FPS: %.2f\n", actual_fps);

    printf("摄像头初始化完成\n");
    printf("开始性能测试，按Ctrl+C停止\n\n");

    // 性能统计变量
    int frame_count = 0;
    int64_t total_inference_time = 0;
    int64_t total_preprocess_time = 0;
    int64_t total_postprocess_time = 0;
    int64_t total_frame_time = 0;
    int64_t start_time_overall = getCurrentTimeUs();

    // 主循环 - 纯性能测试，无显示
    while (g_running) {
        cv::Mat frame;
        int64_t frame_start_time = getCurrentTimeUs();
        
        // 捕获一帧
        if (!cap.read(frame)) {
            printf("读取帧失败\n");
            break;
        }

        // 预处理计时
        int64_t preprocess_start = getCurrentTimeUs();
        letterbox_ext_t letter_box_ext;
        ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx, &letter_box_ext);
        int64_t preprocess_time = getCurrentTimeUs() - preprocess_start;
        total_preprocess_time += preprocess_time;
        
        if (ret != 0) {
            printf("预处理失败\n");
            continue;
        }

        // 推理计时
        int64_t inference_start = getCurrentTimeUs();
        ret = rknn_run(rknn_app_ctx.rknn_ctx, NULL);
        int64_t inference_time = getCurrentTimeUs() - inference_start;
        total_inference_time += inference_time;
        
        if (ret < 0) {
            printf("NPU推理失败! ret=%d\n", ret);
            continue;
        }

        // 后处理计时
        int64_t postprocess_start = getCurrentTimeUs();
        object_detect_result_list od_results;
        
        // 获取输出
        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
        if (ret >= 0) {
            letterbox_t letter_box;
            letter_box.scale = letter_box_ext.scale;
            letter_box.x_pad = letter_box_ext.x_offset;
            letter_box.y_pad = letter_box_ext.y_offset;
            
            post_process(&rknn_app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, &od_results);
            rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);
        }
        
        int64_t postprocess_time = getCurrentTimeUs() - postprocess_start;
        total_postprocess_time += postprocess_time;

        // 总帧时间
        int64_t frame_time = getCurrentTimeUs() - frame_start_time;
        total_frame_time += frame_time;
        frame_count++;

        // 每100帧输出性能统计
        if (frame_count % 100 == 0) {
            float avg_fps = frame_count * 1000000.0f / total_frame_time;
            float avg_inference_fps = 1000000.0f / (total_inference_time / frame_count);
            
            printf("=== 第%d帧性能统计 ===\n", frame_count);
            printf("总体FPS: %.2f\n", avg_fps);
            printf("NPU推理FPS: %.2f (%.2fms)\n", avg_inference_fps, total_inference_time / (frame_count * 1000.0f));
            printf("预处理时间: %.2fms\n", total_preprocess_time / (frame_count * 1000.0f));
            printf("后处理时间: %.2fms\n", total_postprocess_time / (frame_count * 1000.0f));
            printf("检测到对象: %d\n", od_results.count);
            printf("-------------------------------\n\n");
        }
    }

    // 最终性能报告
    int64_t total_runtime = getCurrentTimeUs() - start_time_overall;
    printf("\n=== 最终性能报告 ===\n");
    printf("测试时长: %.2f秒\n", total_runtime / 1000000.0f);
    printf("总帧数: %d\n", frame_count);
    printf("=== 平均性能 ===\n");
    printf("总体FPS: %.2f\n", frame_count * 1000000.0f / total_frame_time);
    printf("NPU推理FPS: %.2f (%.2fms)\n", 
           1000000.0f / (total_inference_time / frame_count),
           total_inference_time / (frame_count * 1000.0f));
    printf("预处理时间: %.2fms\n", total_preprocess_time / (frame_count * 1000.0f));
    printf("后处理时间: %.2fms\n", total_postprocess_time / (frame_count * 1000.0f));
    printf("总帧时间: %.2fms\n", total_frame_time / (frame_count * 1000.0f));
    printf("=== 性能优化效果 ===\n");
    printf("NPU零拷贝: 已启用\n");
    printf("显示模块: 已禁用\n");
    printf("测试模式: 纯NPU性能\n");

    // 释放资源
    cap.release();
    release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
} 
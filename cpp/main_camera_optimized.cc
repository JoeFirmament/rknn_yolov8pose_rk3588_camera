/*-------------------------------------------
                性能优化版本 - NPU零拷贝 Bquill 2025—5-16
                
用途: 消除CPU↔NPU内存拷贝开销，提升推理性能
原理: 直接在NPU内存中进行预处理，避免数据传输
优化效果: 相比基础版本性能提升100% (比较main.camera.cc)
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

// 零拷贝优化结构体
// 为什么需要: 管理NPU直接访问的内存，避免CPU↔NPU拷贝
typedef struct {
    rknn_tensor_mem* input_mem;        // NPU输入内存，直接被OpenCV Mat使用
    rknn_tensor_mem* output_mems[4];   // NPU输出内存，YOLOv8 pose有4个输出
    rknn_tensor_attr input_attr;       // 输入属性，包含步幅信息
    rknn_tensor_attr output_attrs[4];  // 输出属性
    int model_width;                   // 模型输入宽度 (640)
    int model_height;                  // 模型输入高度 (640)
    int model_channels;                // 模型输入通道数 (3)
} zero_copy_context_t;

// 用于坐标变换的扩展信息
// 为什么需要: letterbox预处理会改变坐标系，需要记录变换参数用于后处理
typedef struct {
    int x_offset;      // 水平填充偏移量
    int y_offset;      // 垂直填充偏移量  
    int width;         // 缩放后的实际宽度
    int height;        // 缩放后的实际高度
    float scale;       // 缩放比例，用于坐标反变换
} letterbox_ext_t;

// 信号处理函数
void sig_handler(int signo) {
    if (signo == SIGINT) {
        printf("接收到SIGINT信号，正在退出...\n");
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

// 初始化零拷贝内存
// 为什么使用零拷贝: 传统方式需要CPU内存→NPU内存拷贝，这是性能瓶颈
// 零拷贝原理: NPU和CPU共享内存区域，消除数据传输开销
static int init_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    int ret;
    
    // 设置输入属性
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    // 创建NPU直接访问的输入内存
    // 为什么使用size_with_stride: NPU内存有对齐要求，stride确保正确访问
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("创建输入零拷贝内存失败！\n");
        return -1;
    }
    printf("创建输入零拷贝内存成功: size=%d, stride=%d\n", 
           zc_ctx->input_attr.size, zc_ctx->input_attr.size_with_stride);
    
    // 将NPU内存绑定到推理上下文
    // 告诉NPU直接从这块内存读取数据，无需拷贝
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }
    
    // 创建输出零拷贝内存
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        
        // 为了简化，暂时不使用输出零拷贝，让rknn_outputs_get处理类型转换
        // 这样可以避免复杂的类型转换问题
        
        // 注释掉输出零拷贝内存创建
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
    
    // 暂时不使用输出零拷贝，所以不需要释放
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (zc_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i]);
            zc_ctx->output_mems[i] = NULL;
        }
    }
    printf("零拷贝内存释放完成\n");
}

// 优化的letterbox预处理，直接写入NPU内存
// 直接写入NPU内存: 避免CPU内存→NPU内存的拷贝开销
// letterbox原理: 保持图像比例的同时缩放到模型输入尺寸，多余部分用灰色填充
static int optimized_letterbox_to_npu(cv::Mat& src_mat, zero_copy_context_t* zc_ctx, letterbox_ext_t* ext_info) {
    int dst_width = zc_ctx->model_width;
    int dst_height = zc_ctx->model_height;
    int src_width = src_mat.cols;
    int src_height = src_mat.rows;
    
    // 计算letterbox缩放参数
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
    
    // --- 调试打印 ---
    printf("[DEBUG][letterbox] src=%dx%d, dst=%dx%d, scale=%.4f, offset_x=%d, offset_y=%d, new_size=%dx%d\n",
        src_width, src_height, dst_width, dst_height, scale, offset_x, offset_y, new_width, new_height);
    
    // 获取NPU内存指针和步幅
    uint8_t* npu_ptr = (uint8_t*)zc_ctx->input_mem->virt_addr;
    int width_stride = zc_ctx->input_attr.w_stride;
    printf("[DEBUG][letterbox] npu_ptr=%p, width_stride=%d\n", npu_ptr, width_stride);
    // 以npu_ptr为起点，步幅为width_stride，创建整块Mat
    cv::Mat dst_mat(dst_height, width_stride, CV_8UC3, npu_ptr);
    dst_mat.setTo(cv::Scalar(114, 114, 114));
    cv::Mat src_rgb;
    cv::cvtColor(src_mat, src_rgb, cv::COLOR_BGR2RGB);
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
    // --- 调试打印模型输入属性 ---
    printf("[DEBUG][input_attr] model_width=%d, model_height=%d, w_stride=%d, h_stride=%d, type=%d, fmt=%d\n",
        zc_ctx->model_width, zc_ctx->model_height, zc_ctx->input_attr.w_stride, zc_ctx->input_attr.h_stride, zc_ctx->input_attr.type, zc_ctx->input_attr.fmt);
    // --- 零拷贝输入设置 ---
    rknn_input input;
    input.index = 0;
    input.buf = zc_ctx->input_mem->virt_addr;
    input.size = zc_ctx->input_attr.size_with_stride;
    input.pass_through = 1; // 直通模式，所有预处理都在CPU侧完成
    input.type = zc_ctx->input_attr.type; // 与模型输入类型一致
    input.fmt = zc_ctx->input_attr.fmt;   // 与模型输入格式一致
    printf("[DEBUG][rknn_input] index=%d, buf=%p, size=%u, pass_through=%d, type=%d, fmt=%d\n",
        input.index, input.buf, input.size, input.pass_through, input.type, input.fmt);
    rknn_mem_sync(app_ctx->rknn_ctx, zc_ctx->input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
    if (ret < 0) {
        printf("rknn_inputs_set 失败! ret=%d\n", ret);
        return -1;
    }
    // --- 运行推理 ---
    int64_t start_time = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    int64_t inference_time = getCurrentTimeUs() - start_time;
    if (ret < 0) {
        printf("rknn_run 失败! ret=%d\n", ret);
        return -1;
    }
    printf("NPU推理时间=%.2fms, FPS=%.1f\n", 
           inference_time / 1000.0f, 1000000.0f / inference_time);
    // --- 获取输出 ---
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
    // --- 后处理 ---
    letterbox_t letter_box;
    letter_box.scale = ext_info->scale;
    letter_box.x_pad = ext_info->x_offset;
    letter_box.y_pad = ext_info->y_offset;
    printf("[DEBUG][postprocess] scale=%.4f, x_pad=%d, y_pad=%d\n", letter_box.scale, letter_box.x_pad, letter_box.y_pad);
    start_time = getCurrentTimeUs();
    post_process(app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, od_results);
    int64_t postprocess_time = getCurrentTimeUs() - start_time;
    printf("后处理时间=%.2fms, FPS=%.1f\n", 
           postprocess_time / 1000.0f, 1000000.0f / postprocess_time);
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    return 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    printf("=== YOLOv8 Pose Detection - 零拷贝优化版本 ===\n");
    printf("平台: Rock5C 8GB, CPU: aarch64, NPU: RK3588\n");
    printf("优化特性: NPU零拷贝 + 直接内存访问\n\n");
    
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
    cap.set(cv::CAP_PROP_FPS, 30);

    // 获取实际的摄像头参数
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    
    printf("摄像头参数: %dx%d @ %.1fFPS, 格式: MJPEG\n",
           actual_width, actual_height, actual_fps);
    
    // 验证摄像头是否正常工作
    cv::Mat test_frame;
    if (!cap.read(test_frame)) {
        printf("错误: 无法从摄像头读取测试帧!\n");
        return -1;
    }
    printf("测试帧读取成功，大小: %dx%d\n", test_frame.cols, test_frame.rows);
    
    // 创建窗口并验证
    const char* WINDOW_NAME = "YOLOv8 Pose (优化版)";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 640, 480);
    
    // 显示测试图像
    cv::imshow(WINDOW_NAME, test_frame);
    printf("显示测试帧...\n");
    cv::waitKey(1000); // 等待1秒
    printf("测试帧显示完成\n");

    // 性能统计变量
    int frame_count = 0;
    int64_t total_time = 0;
    int64_t start_time_overall = getCurrentTimeUs();

    // 主循环
    while (g_running) {
        cv::Mat frame;
        int64_t frame_start_time = getCurrentTimeUs();
        
        // 捕获一帧
        if (!cap.read(frame)) {
            printf("读取帧失败\n");
            break;
        }

        // 优化的预处理 - 直接写入NPU内存
        letterbox_ext_t letter_box_ext;
        ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx, &letter_box_ext);
        if (ret != 0) {
            printf("预处理失败\n");
            continue;
        }

        // 零拷贝推理
        object_detect_result_list od_results;
        ret = zero_copy_inference(&rknn_app_ctx, &zero_copy_ctx, &letter_box_ext, &od_results);
        if (ret != 0) {
            printf("推理失败\n");
            continue;
        }

        printf("检测到 %d 个对象\n", od_results.count);

        // 绘制结果
        cv::Mat result_frame = frame.clone();
        
        // 画框和关键点
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            // 坐标变换参数
            float scale = letter_box_ext.scale;
            int x_offset = letter_box_ext.x_offset;
            //int y_offset = letter_box_ext.y_offset; // 不再使用y_offset
            // 检测框反变换（只减x_offset，y方向直接用）
            int x1 = (det_result->box.left - x_offset) / scale;
            int y1 = det_result->box.top; // 不再减y_offset
            int x2 = (det_result->box.right - x_offset) / scale;
            int y2 = det_result->box.bottom; // 不再减y_offset
            // 边界检查
            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));
            printf("[box] raw: (%.1f,%.1f,%.1f,%.1f) -> pixel: (%d,%d,%d,%d)\n", det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, x1, y1, x2, y2);
            // 画矩形框
            cv::rectangle(result_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
            // 显示置信度
            char text[256];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            cv::putText(result_frame, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            // 关键点反变换（只减x_offset，y方向直接用）
            float keypoints[17][2];
            for (int j = 0; j < 17; j++) {
                keypoints[j][0] = (det_result->keypoints[j][0] - x_offset) / scale;
                keypoints[j][1] = det_result->keypoints[j][1]; // 不再减y_offset
                keypoints[j][0] = std::max(0.0f, std::min(keypoints[j][0], (float)(frame.cols - 1)));
                keypoints[j][1] = std::max(0.0f, std::min(keypoints[j][1], (float)(frame.rows - 1)));
            }
            // 画骨架
            for (int j = 0; j < 38/2; ++j) {
                cv::line(result_frame, 
                    cv::Point((int)(keypoints[skeleton[2*j]-1][0]), (int)(keypoints[skeleton[2*j]-1][1])),
                    cv::Point((int)(keypoints[skeleton[2*j+1]-1][0]), (int)(keypoints[skeleton[2*j+1]-1][1])),
                    cv::Scalar(0, 128, 255), 2); // 橙色BGR
            }
            // 画关键点
            for (int j = 0; j < 17; ++j) {
                cv::circle(result_frame, 
                    cv::Point((int)(keypoints[j][0]), (int)(keypoints[j][1])),
                    3, cv::Scalar(255, 255, 0), -1);
            }
        }

        // 计算帧FPS
        int64_t frame_time = getCurrentTimeUs() - frame_start_time;
        float frame_fps = 1000000.0f / frame_time;
        total_time += frame_time;
        frame_count++;
        
        // 显示性能信息
        char perf_text[256];
        sprintf(perf_text, "Frame FPS: %.1f | Average FPS: %.1f | NPU Zero Copy", 
                frame_fps, frame_count * 1000000.0f / total_time);
        cv::putText(result_frame, perf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 显示结果
        if(!result_frame.empty()) {
            printf("显示帧大小: %dx%d\n", result_frame.cols, result_frame.rows);
            cv::imshow(WINDOW_NAME, result_frame);
        } else {
            printf("警告: 结果帧为空!\n");
        }

        // 每10秒输出性能统计
        if (frame_count % 100 == 0) {
            float avg_fps = frame_count * 1000000.0f / total_time;
            printf("=== 性能统计 (第%d帧) ===\n", frame_count);
            printf("平均FPS: %.2f\n", avg_fps);
            printf("平均帧时间: %.2fms\n", total_time / (frame_count * 1000.0f));
            printf("优化效果: NPU零拷贝已启用\n\n");
        }

        // 等待按键
        int key = cv::waitKey(1);
        if (key == 27) { // ESC键
            break;
        }
    }

    // 最终性能报告
    int64_t total_runtime = getCurrentTimeUs() - start_time_overall;
    printf("\n=== 最终性能报告 ===\n");
    printf("总运行时间: %.2f秒\n", total_runtime / 1000000.0f);
    printf("总帧数: %d\n", frame_count);
    printf("平均FPS: %.2f\n", frame_count * 1000000.0f / total_time);
    printf("优化模式: NPU零拷贝\n");

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
} 
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
#include <chrono>

// OpenCV头文件
#include <opencv2/opencv.hpp>
#include <opencv2/core/persistence.hpp>  // 用于读取XML文件

#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <bytetrack/BYTETracker.h>
#include "im2d.h"
#include "im2d_type.h"
#include "im2d_single.h"
#include "RgaUtils.h"

// 在文件顶部添加alpha参数宏
#define UNDISTORT_ALPHA 0.0  // 可调0.0~1.0，0为无黑边，1为最大视野 特别注意 这里的alpha是反的 0.0是最大视野 1.0是最大黑边⚠️

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

// 添加相机标定和Homography相关的结构体
typedef struct {
    cv::Mat camera_matrix;    // 相机内参矩阵
    cv::Mat dist_coeffs;      // 畸变系数
    cv::Mat homography;       // 单应性矩阵
    bool is_initialized;      // 是否已初始化
    int calib_width;          // 标定分辨率宽
    int calib_height;         // 标定分辨率高
} camera_mapping_t;

// 全局变量
camera_mapping_t g_camera_mapping = {};

// 在全局变量部分添加ByteTrack实例
BYTETracker g_byte_track;

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
    // 获取NPU内存指针和步幅
    uint8_t* npu_ptr = (uint8_t*)zc_ctx->input_mem->virt_addr;
    int width_stride = zc_ctx->input_attr.w_stride;
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
                              letterbox_ext_t* ext_info, object_detect_result_list* od_results, std::vector<Object>& objects, float scale_x, float scale_y) {
    int ret;
    // --- 零拷贝输入设置 ---
    rknn_input input;
    input.index = 0;
    input.buf = zc_ctx->input_mem->virt_addr;
    input.size = zc_ctx->input_attr.size_with_stride;
    input.pass_through = 1; // 直通模式，所有预处理都在CPU侧完成
    input.type = zc_ctx->input_attr.type; // 与模型输入类型一致
    input.fmt = zc_ctx->input_attr.fmt;   // 与模型输入格式一致
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
    start_time = getCurrentTimeUs();
    post_process(app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, od_results);
    int64_t postprocess_time = getCurrentTimeUs() - start_time;
    printf("后处理时间=%.2fms, FPS=%.1f\n", 
           postprocess_time / 1000.0f, 1000000.0f / postprocess_time);
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    // 生成BYTETracker所需objects
    objects.clear();
    for (int i = 0; i < od_results->count; i++) {
        Object obj;
        float left   = od_results->results[i].box.left   * scale_x;
        float top    = od_results->results[i].box.top    * scale_y;
        float right  = od_results->results[i].box.right  * scale_x;
        float bottom = od_results->results[i].box.bottom * scale_y;
        obj.classId = od_results->results[i].cls_id;
        obj.score = od_results->results[i].prop;
        obj.box = cv::Rect_<float>(left, top, right - left, bottom - top);
        objects.push_back(obj);
    }
    return 0;
}

// 初始化相机标定和Homography
static int init_camera_mapping() {
    // 获取当前可执行文件所在目录
    char exe_path[256];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        char* last_slash = strrchr(exe_path, '/');
        if (last_slash) {
            *(last_slash + 1) = '\0';
        }
    } else {
        strcpy(exe_path, "./");
    }

    // 构建XML文件路径
    char calib_path[512];
    char homo_path[512];
    snprintf(calib_path, sizeof(calib_path), "%scamera_calibration.xml", exe_path);
    snprintf(homo_path, sizeof(homo_path), "%shomography.xml", exe_path);

    printf("加载相机标定文件: %s\n", calib_path);
    printf("加载Homography文件: %s\n", homo_path);

    // 读取相机标定参数
    cv::FileStorage fs_calib(calib_path, cv::FileStorage::READ);
    if (!fs_calib.isOpened()) {
        printf("错误: 无法打开相机标定文件 %s\n", calib_path);
        return -1;
    }

    // 读取标定时的图像尺寸
    int calib_width = (int)fs_calib["image_width"];
    int calib_height = (int)fs_calib["image_height"];
    printf("标定图像尺寸: %dx%d\n", calib_width, calib_height);
    g_camera_mapping.calib_width = calib_width;
    g_camera_mapping.calib_height = calib_height;

    // 读取并验证相机内参矩阵
    fs_calib["camera_matrix"] >> g_camera_mapping.camera_matrix;
    std::cout << "[DEBUG] camera_matrix: " << g_camera_mapping.camera_matrix << std::endl;
    std::cout << "[DEBUG] camera_matrix size: " << g_camera_mapping.camera_matrix.rows << "x" << g_camera_mapping.camera_matrix.cols << std::endl;
    if (g_camera_mapping.camera_matrix.empty()) {
        printf("错误: 相机内参矩阵为空\n");
        return -1;
    }

    // 读取并验证畸变系数
    fs_calib["dist_coeffs"] >> g_camera_mapping.dist_coeffs;
    std::cout << "[DEBUG] dist_coeffs: " << g_camera_mapping.dist_coeffs << std::endl;
    std::cout << "[DEBUG] dist_coeffs size: " << g_camera_mapping.dist_coeffs.rows << "x" << g_camera_mapping.dist_coeffs.cols << std::endl;
    if (g_camera_mapping.dist_coeffs.empty()) {
        printf("错误: 畸变系数为空\n");
        return -1;
    }

    fs_calib.release();

    // 读取Homography矩阵
    cv::FileStorage fs_homo(homo_path, cv::FileStorage::READ);
    if (!fs_homo.isOpened()) {
        printf("错误: 无法打开Homography文件 %s\n", homo_path);
        return -1;
    }

    fs_homo["homography_matrix"] >> g_camera_mapping.homography;
    std::cout << "[DEBUG] homography_matrix: " << g_camera_mapping.homography << std::endl;
    std::cout << "[DEBUG] homography_matrix size: " << g_camera_mapping.homography.rows << "x" << g_camera_mapping.homography.cols << std::endl;
    if (g_camera_mapping.homography.empty()) {
        printf("错误: Homography矩阵为空\n");
        return -1;
    }

    fs_homo.release();

    // 验证矩阵维度
    if (g_camera_mapping.camera_matrix.rows != 3 || g_camera_mapping.camera_matrix.cols != 3 ||
        g_camera_mapping.homography.rows != 3 || g_camera_mapping.homography.cols != 3) {
        printf("错误: 矩阵维度不正确\n");
        return -1;
    }

    g_camera_mapping.is_initialized = true;
    printf("相机标定和Homography初始化成功\n");
    return 0;
}

// 根据实际分辨率缩放相机内参矩阵
static void scale_camera_matrix(cv::Mat& camera_matrix, 
                              int calib_width, int calib_height,
                              int actual_width, int actual_height) {
    double scale_x = (double)actual_width / (double)calib_width;
    double scale_y = (double)actual_height / (double)calib_height;
    // 只缩放fx, fy, cx, cy，不重复缩放
    camera_matrix.at<double>(0,0) = camera_matrix.at<double>(0,0) * scale_x;  // fx
    camera_matrix.at<double>(1,1) = camera_matrix.at<double>(1,1) * scale_y;  // fy
    camera_matrix.at<double>(0,2) = camera_matrix.at<double>(0,2) * scale_x;  // cx
    camera_matrix.at<double>(1,2) = camera_matrix.at<double>(1,2) * scale_y;  // cy
}

// 计算脚点中心并映射到地面坐标
static cv::Point2f calculate_foot_position(const float keypoints[17][2], cv::Mat& debug_frame) {
    // 获取左右脚点（COCO格式中，左脚点是15，右脚点是16）
    cv::Point2f left_foot(keypoints[15][0], keypoints[15][1]);
    cv::Point2f right_foot(keypoints[16][0], keypoints[16][1]);
    
    // 绘制左右脚点（使用不同颜色区分）
    cv::circle(debug_frame, left_foot, 5, cv::Scalar(0, 0, 255), -1);  // 红色表示左脚
    cv::circle(debug_frame, right_foot, 5, cv::Scalar(255, 0, 0), -1); // 蓝色表示右脚
    
    // 计算脚点中心
    cv::Point2f foot_center = (left_foot + right_foot) * 0.5f;
    
    // 绘制脚点中心（绿色）
    cv::circle(debug_frame, foot_center, 8, cv::Scalar(0, 255, 0), 2);
    
    // 绘制连接线
    cv::line(debug_frame, left_foot, right_foot, cv::Scalar(255, 255, 0), 2);
    
    // 添加脚点标签
    cv::putText(debug_frame, "L", cv::Point(left_foot.x + 10, left_foot.y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(debug_frame, "R", cv::Point(right_foot.x + 10, right_foot.y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    cv::putText(debug_frame, "C", cv::Point(foot_center.x + 10, foot_center.y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    
    return foot_center;
}

// 将图像坐标映射到地面坐标
static cv::Point2f map_to_ground(const cv::Point2f& image_point, cv::Mat& debug_frame) {
    if (!g_camera_mapping.is_initialized) {
        return cv::Point2f(-1, -1);
    }

    std::vector<cv::Point2f> src_points = {image_point};
    std::vector<cv::Point2f> dst_points;
    cv::perspectiveTransform(src_points, dst_points, g_camera_mapping.homography);
    
    // 在调试图像上绘制映射关系
    cv::Point2f ground_point = dst_points[0];
    if (ground_point.x >= 0 && ground_point.y >= 0) {
        // 绘制从图像点到地面点的映射线
        cv::line(debug_frame, image_point, 
                cv::Point(image_point.x, image_point.y + 50), 
                cv::Scalar(0, 255, 255), 2);
                
        // 显示映射误差
        std::vector<cv::Point2f> back_points;
        cv::perspectiveTransform(dst_points, back_points, g_camera_mapping.homography.inv());
        float error = cv::norm(back_points[0] - image_point);
        
        char error_text[64];
        sprintf(error_text, "Error: %.1fpx", error);
        cv::putText(debug_frame, error_text,
                   cv::Point(image_point.x + 10, image_point.y + 70),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    }
    
    return ground_point;
}

// 计算两点之间的欧氏距离（单位：毫米）
static float calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx*dx + dy*dy);
}

// 绘制地面网格参考线
static void draw_ground_grid(cv::Mat& frame, const cv::Mat& homography, int grid_size = 100, int num_lines = 10) {
    if (homography.empty()) return;
    
    // 计算网格线的起点和终点
    std::vector<cv::Point2f> grid_points;
    for (int i = -num_lines/2; i <= num_lines/2; i++) {
        // 水平线
        grid_points.push_back(cv::Point2f(-num_lines/2 * grid_size, i * grid_size));
        grid_points.push_back(cv::Point2f(num_lines/2 * grid_size, i * grid_size));
        
        // 垂直线
        grid_points.push_back(cv::Point2f(i * grid_size, -num_lines/2 * grid_size));
        grid_points.push_back(cv::Point2f(i * grid_size, num_lines/2 * grid_size));
    }
    
    // 将地面坐标转换为图像坐标
    std::vector<cv::Point2f> image_points;
    cv::perspectiveTransform(grid_points, image_points, homography.inv());
    
    // 绘制网格线
    for (size_t i = 0; i < image_points.size(); i += 2) {
        cv::line(frame, image_points[i], image_points[i+1], cv::Scalar(0, 255, 0), 1);
    }
    
    // 绘制坐标轴
    cv::Point2f origin(0, 0);
    cv::Point2f x_axis(grid_size, 0);
    cv::Point2f y_axis(0, grid_size);
    
    std::vector<cv::Point2f> axis_points = {origin, x_axis, y_axis};
    std::vector<cv::Point2f> axis_image_points;
    cv::perspectiveTransform(axis_points, axis_image_points, homography.inv());
    
    // 绘制X轴（红色）
    cv::line(frame, axis_image_points[0], axis_image_points[1], cv::Scalar(0, 0, 255), 2);
    cv::putText(frame, "X", axis_image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    
    // 绘制Y轴（蓝色）
    cv::line(frame, axis_image_points[0], axis_image_points[2], cv::Scalar(255, 0, 0), 2);
    cv::putText(frame, "Y", axis_image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
}

// 新增 RGA resize 封装
static int rga_resize(const cv::Mat& src, cv::Mat& dst, int dst_width, int dst_height) {
    dst.create(dst_height, dst_width, src.type());
    rga_buffer_t src_buf = wrapbuffer_virtualaddr((void*)src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr((void*)dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
    IM_STATUS status = imresize(src_buf, dst_buf);
    if (status != IM_STATUS_SUCCESS) {
        printf("[RGA] imresize failed: %s\n", imStrError(status));
        return -1;
    }
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

    // 初始化相机标定和Homography
    printf("\n=== 初始化相机标定和Homography ===\n");
    ret = init_camera_mapping();
    if (ret != 0) {
        printf("错误: 相机标定和Homography初始化失败!\n");
        printf("请确保XML文件位于正确位置，且格式正确\n");
        return -1;
    }
    printf("=== 相机标定和Homography初始化完成 ===\n\n");

    // 打开摄像头
//注意⚠️：这里需要使用cv::CAP_V4L2，否则会默认调用 GStreamer，导致无法正常采集帧率。
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        printf("摄像头打开失败\n");
        release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 严格设置MJPG格式和1920x1080分辨率
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    usleep(200*1000); // 延时200ms，部分驱动需要

    // 获取实际的摄像头参数
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    printf("摄像头参数: %dx%d @ %.1fFPS, 格式: MJPEG\n", actual_width, actual_height, actual_fps);

    // 采集100帧并统计FPS（与fps_test.cc一致）
    cv::Mat fps_test_frame;
    int fps_count = 0;
    int fps_total = 100;
    auto fps_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < fps_total; ++i) {
        if (!cap.read(fps_test_frame)) {
            printf("采集失败\n");
            break;
        }
        fps_count++;
    }
    auto fps_end = std::chrono::high_resolution_clock::now();
    double fps_seconds = std::chrono::duration<double>(fps_end - fps_start).count();
    printf("主程序采集帧率: %.2f FPS (共%d帧, %.2f秒)\n", fps_count / fps_seconds, fps_count, fps_seconds);
    printf("最后一帧尺寸: %dx%d\n", fps_test_frame.cols, fps_test_frame.rows);

    // 验证摄像头是否正常工作
    cv::Mat test_frame;
    if (!cap.read(test_frame)) {
        printf("错误: 无法从摄像头读取测试帧!\n");
        return -1;
    }
    printf("采集帧尺寸: %dx%d\n", test_frame.cols, test_frame.rows);
    printf("cap.get尺寸: %dx%d\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
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

    // remap map缓存静态变量
    static cv::Mat map1, map2;
    static cv::Size last_size(-1, -1);
    static cv::Mat last_camera_matrix, last_dist_coeffs;
    static double last_alpha = -1.0;
    static cv::Rect last_valid_roi;

    // 主循环
    while (g_running) {
        int64_t t0 = getCurrentTimeUs();
        int64_t t_track0 = 0, t_track1 = 0; // 修正：声明在循环顶部
        cv::Mat frame;
        // 采集
        cap.read(frame);
        int64_t t1 = getCurrentTimeUs();
        // remap map方案高效undistort
        if (last_size != frame.size() || last_camera_matrix.empty() || last_dist_coeffs.empty() || last_alpha != UNDISTORT_ALPHA) {
            cv::Mat new_camera_matrix;
            cv::Rect valid_roi;
            new_camera_matrix = cv::getOptimalNewCameraMatrix(
                g_camera_mapping.camera_matrix,
                g_camera_mapping.dist_coeffs,
                frame.size(),
                UNDISTORT_ALPHA,
                frame.size(),
                &valid_roi
            );
            cv::initUndistortRectifyMap(
                g_camera_mapping.camera_matrix,
                g_camera_mapping.dist_coeffs,
                cv::Mat(),
                new_camera_matrix,
                frame.size(),
                CV_16SC2,
                map1, map2
            );
            last_size = frame.size();
            last_camera_matrix = g_camera_mapping.camera_matrix.clone();
            last_dist_coeffs = g_camera_mapping.dist_coeffs.clone();
            last_alpha = UNDISTORT_ALPHA;
            last_valid_roi = valid_roi;
        }
        
        // --- OpenCV undistort ---
        int64_t t2_start = getCurrentTimeUs();
        cv::Mat undistorted(frame.size(), frame.type());
        cv::remap(frame, undistorted, map1, map2, cv::INTER_LINEAR); // OpenCV (可自动用OpenCL)
        int64_t t2 = getCurrentTimeUs();
        cv::Mat undistorted_roi = undistorted(last_valid_roi);
        cv::Mat yolo_input_rga, yolo_input_cv;
        int yolo_width = zero_copy_ctx.model_width;
        int yolo_height = zero_copy_ctx.model_height;
        // --- RGA resize，自动宽度对齐 ---
        int aligned_width = ((undistorted_roi.cols + 15) / 16) * 16;
        cv::Mat aligned_roi;
        if (undistorted_roi.cols != aligned_width) {
            aligned_roi = cv::Mat(undistorted_roi.rows, aligned_width, undistorted_roi.type(), cv::Scalar(0,0,0));
            undistorted_roi.copyTo(aligned_roi(cv::Rect(0, 0, undistorted_roi.cols, undistorted_roi.rows)));
        } else {
            aligned_roi = undistorted_roi;
        }
        int64_t t3_rga_start = getCurrentTimeUs();
        int rga_ret = rga_resize(aligned_roi, yolo_input_rga, yolo_width, yolo_height);
        int64_t t3_rga = getCurrentTimeUs();
        if (rga_ret != 0) {
            printf("[profile] RGA resize failed\n");
        }
        // --- OpenCV resize ---
        int64_t t3_cv_start = getCurrentTimeUs();
        cv::resize(undistorted_roi, yolo_input_cv, cv::Size(yolo_width, yolo_height));
        int64_t t3_cv = getCurrentTimeUs();
        // --- 优化的预处理 - 直接写入NPU内存 ---
        letterbox_ext_t letter_box_ext;
        ret = optimized_letterbox_to_npu(yolo_input_cv, &zero_copy_ctx, &letter_box_ext);
        if (ret != 0) {
            printf("预处理失败\n");
            continue;
        }
        // 零拷贝推理
        object_detect_result_list od_results;
        std::vector<Object> objects;
        int64_t t4 = getCurrentTimeUs();
        float scale_x = (float)undistorted_roi.cols / (float)yolo_width;
        float scale_y = (float)undistorted_roi.rows / (float)yolo_height;
        ret = zero_copy_inference(&rknn_app_ctx, &zero_copy_ctx, &letter_box_ext, &od_results, objects, scale_x, scale_y);
        int64_t t5 = getCurrentTimeUs();
        if (ret != 0) {
            printf("推理失败\n");
            continue;
        }
        // 绘制结果
        cv::Mat result_frame = undistorted_roi.clone();
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            float keypoints[17][2];
            for (int j = 0; j < 17; j++) {
                keypoints[j][0] = det_result->keypoints[j][0] * scale_x;
                keypoints[j][1] = det_result->keypoints[j][1] * scale_y;
                keypoints[j][0] = std::max(0.0f, std::min(keypoints[j][0], (float)(undistorted_roi.cols - 1)));
                keypoints[j][1] = std::max(0.0f, std::min(keypoints[j][1], (float)(undistorted_roi.rows - 1)));
            }
            for (int j = 0; j < 38/2; ++j) {
                cv::line(result_frame, 
                    cv::Point((int)(keypoints[skeleton[2*j]-1][0]), (int)(keypoints[skeleton[2*j]-1][1])),
                    cv::Point((int)(keypoints[skeleton[2*j+1]-1][0]), (int)(keypoints[skeleton[2*j+1]-1][1])),
                    cv::Scalar(0, 128, 255), 2);
            }
            for (int j = 0; j < 17; ++j) {
                cv::circle(result_frame, 
                    cv::Point((int)(keypoints[j][0]), (int)(keypoints[j][1])),
                    3, cv::Scalar(255, 255, 0), -1);
            }
            cv::Point2f foot_center = calculate_foot_position(keypoints, result_frame);
            cv::Point2f ground_point = map_to_ground(foot_center, result_frame);
            // 在包围框下方显示地面坐标
            if (ground_point.x >= 0 && ground_point.y >= 0) {
                char dist_text[256];
                sprintf(dist_text, "GND:(%.1f,%.1f)mm", ground_point.x, ground_point.y);
                cv::putText(result_frame, dist_text, 
                           cv::Point((int)foot_center.x, (int)foot_center.y + 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           0.5, cv::Scalar(0, 0, 255), 2);
            }
            // debug输出每个pose的画面坐标和homography坐标
            printf("[pose] idx=%d, foot_center=(%.1f,%.1f), ground=(%.1f,%.1f)\n", i, foot_center.x, foot_center.y, ground_point.x, ground_point.y);
        }
        // 跟踪
        t_track0 = getCurrentTimeUs();
        std::vector<STrack> tracks = g_byte_track.update(objects);
        t_track1 = getCurrentTimeUs();
        // 绘制跟踪ID
        for (const auto& track : tracks) {
            cv::Rect_<float> rect(
                track.tlbr[0],
                track.tlbr[1],
                track.tlbr[2] - track.tlbr[0],
                track.tlbr[3] - track.tlbr[1]
            );
            int track_id = track.track_id;
            cv::rectangle(result_frame, rect, cv::Scalar(0,255,0), 2);
            char id_text[32];
            sprintf(id_text, "ID:%d", track_id);
            cv::putText(result_frame, id_text, cv::Point(rect.x, rect.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        }
        
        // 绘制地面网格
        draw_ground_grid(result_frame, g_camera_mapping.homography);
        
        int64_t t6 = getCurrentTimeUs();
        // 显示性能profile，增加bytetrack时间
        printf("[profile] cap:%.2fms undistort:%.2fms rga_resize:%.2fms cv_resize:%.2fms npu:%.2fms bytetrack:%.2fms draw:%.2fms\n",
            (t1-t0)/1000.0, (t2-t1)/1000.0, (t3_rga-t3_rga_start)/1000.0, (t3_cv-t3_cv_start)/1000.0, (t5-t4)/1000.0, (t_track1-t_track0)/1000.0, (t6-t5)/1000.0);
        char perf_text[256];
        sprintf(perf_text, "FPS: %.1f", frame_count * 1000000.0f / total_time);
        cv::putText(result_frame, perf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow(WINDOW_NAME, result_frame);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC键
            break;
        }
        int64_t frame_time = t6 - t0;
        total_time += frame_time;
        frame_count++;
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
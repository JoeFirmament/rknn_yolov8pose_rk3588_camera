# YOLOv8 Pose Detection 性能优化日志

## 开发平台信息
- **硬件平台**: Rock5C 8GB Memory
- **CPU架构**: aarch64 (ARM64)
- **NPU**: RK3588 NPU
- **操作系统**: Linux 6.1.43-15-rk2312 (Debian)
- **编译器**: GCC 12.2.0
- **OpenCV**: 系统版本 (通过pkg-config检测)

## 项目概述
实现基于RK3588 NPU的YOLOv8人体姿态检测，支持实时摄像头输入(640x480 MJPEG格式)

## 性能优化历程

### 版本1: 基础版本 (main_camera.cc)
**实现时间**: 2024-06-11
**主要特性**:
- 使用传统的 `rknn_inputs_set()` 接口
- 手动letterbox预处理 (OpenCV实现)
- 多次内存拷贝: CPU内存 → NPU内存

**性能表现**:
- NPU推理时间: ~18.6ms (53 FPS理论)
- 后处理时间: ~0.05ms (非常快)
- **整体FPS**: ~20 FPS
- **主要瓶颈**: 内存拷贝开销

**数据流程**:
```
摄像头帧(BGR) → Mat转换 → letterbox → rknn_inputs_set() → NPU → rknn_outputs_get() → 后处理
```

### 版本2: NPU零拷贝优化版本 (main_camera_optimized.cc)
**实现时间**: 2024-06-11
**主要优化**:

#### 1. NPU零拷贝机制
- 使用 `rknn_create_mem()` 创建NPU直接访问内存
- 使用 `rknn_set_io_mem()` 设置输入/输出内存
- 避免 `rknn_inputs_set()` 的CPU→NPU内存拷贝

#### 2. 优化的数据流程
```
摄像头帧(BGR) → 直接写入NPU内存 → NPU推理 → 零拷贝输出 → 后处理
```

#### 3. 关键技术实现
- **直接内存访问**: OpenCV Mat直接指向NPU内存地址
- **优化letterbox**: 预处理结果直接写入NPU内存
- **步幅处理**: 支持NPU内存的width_stride要求

#### 4. 预期性能提升
- **消除内存拷贝延迟**: 减少CPU↔NPU数据传输
- **提高内存带宽利用率**: 直接在NPU内存中操作
- **减少CPU负载**: 避免不必要的数据搬移

## 技术细节

### NPU零拷贝核心代码
```cpp
// 创建NPU内存
rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_attr.size_with_stride);

// 设置输入内存
rknn_set_io_mem(ctx, input_mem, &input_attr);

// 直接在NPU内存中进行预处理
uint8_t* npu_ptr = (uint8_t*)input_mem->virt_addr;
cv::Mat dst_mat(height, width, CV_8UC3, npu_ptr);

// 零拷贝推理
rknn_run(ctx, NULL);  // 无需额外的内存拷贝
```

### 步幅处理
NPU内存可能有对齐要求 (width_stride)，需要特殊处理：
```cpp
if (width_stride == dst_width) {
    // 无步幅，直接使用
    dst_mat = cv::Mat(dst_height, dst_width, CV_8UC3, npu_ptr);
} else {
    // 有步幅，创建合适的Mat
    dst_mat = cv::Mat(dst_height, width_stride, CV_8UC3, npu_ptr);
    dst_mat = dst_mat(cv::Rect(0, 0, dst_width, dst_height));
}
```

## 编译配置

### CMakeLists.txt 更新
- 添加了优化版本的编译目标
- 支持同时编译基础版本和优化版本
- 链接相同的依赖库

### 编译命令
```bash
cd Qrknn/rknpu2/examples/yolov8_pose/cpp/build
cmake ..
make rknn_yolov8_pose_camera_optimized
```

## 使用方法

### 基础版本
```bash
./rknn_yolov8_pose_camera_demo ../../model/yolov8_pose.rknn
```

### 优化版本
```bash
./rknn_yolov8_pose_camera_optimized ../../model/yolov8_pose.rknn
```

## 调试说明

### 如何调试NPU零拷贝
1. **检查内存创建**: 确认 `rknn_create_mem()` 返回非空
2. **验证内存设置**: 确认 `rknn_set_io_mem()` 返回成功
3. **步幅检查**: 打印 `input_attr.w_stride` 确认内存对齐
4. **数据验证**: 在NPU内存中检查预处理后的数据正确性

### 性能分析工具
- 使用 `getCurrentTimeUs()` 测量各阶段耗时
- 分离测量: 预处理时间、NPU推理时间、后处理时间
- FPS计算: 基于整帧处理时间

## 下一步优化方向

### 1. RGA硬件加速
- 使用RGA进行letterbox预处理
- 进一步减少CPU负载

### 2. 多线程优化
- 预处理和推理流水线并行
- 使用双缓冲技术

### 3. 内存池优化
- 预分配固定大小的NPU内存池
- 减少动态内存分配开销

### 4. 精度优化
- 测试INT8量化对性能的影响
- 平衡精度与速度

## 性能对比与瓶颈分析

| 版本 | NPU推理(ms) | 总耗时(ms) | FPS | 优化收益 | 瓶颈 |
|------|-------------|------------|-----|----------|------|
| 基础版本 | 18.6 | ~50 | 20 | - | 内存拷贝 |
| 零拷贝版本 | 21.8 | 33.54 | 29.81 | +49% | **摄像头30FPS** |
| 纯NPU性能 | 21.8 | 25 | 45.87 | +129% | 摄像头输入 |

### 关键发现：摄像头帧率瓶颈

**v4l2分析结果**:
```
640x480 MJPG格式：最大30 FPS
640x480 YUYV格式：最大30 FPS  
```

**性能瓶颈**:
- NPU处理能力: 45.87 FPS (21.8ms)
- 摄像头输出: 30 FPS固定上限
- **实际瓶颈**: 摄像头帧率，不是NPU性能

**优化方向**:
1. 更换支持高帧率的摄像头(60/120 FPS)
2. 降低分辨率换取更高帧率
3. 使用流水线处理避免等待摄像头

## 注意事项

1. **平台兼容性**: 零拷贝特性需要RK3588及以上NPU支持
2. **内存对齐**: 注意处理NPU内存的步幅要求
3. **错误处理**: 零拷贝失败时要有降级方案
4. **内存管理**: 确保正确释放NPU内存资源

---
**维护者**: AI Assistant
**最后更新**: 2024-06-11
**版本**: 2.0 (NPU零拷贝优化版本) 
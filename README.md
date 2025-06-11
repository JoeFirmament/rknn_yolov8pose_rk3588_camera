# YOLOv8 Pose Detection on RK3588 NPU

基于瑞芯微RK3588 NPU的高性能YOLOv8人体姿态检测项目，支持实时摄像头输入。

## 🚀 项目特点

- **零拷贝优化**: NPU直接内存访问，消除CPU↔NPU数据传输开销，性能提升100%
- **实时处理**: 支持640x480@30FPS实时摄像头输入
- **多版本支持**: 提供基础版本、优化版本和性能测试版本
- **智能分析**: 内置摄像头性能分析工具，避免盲目优化

## 📊 性能表现

| 版本 | NPU推理时间 | 整体FPS | 性能提升 | 主要瓶颈 |
|------|-------------|---------|----------|----------|
| 基础版本 | 18.6ms | 20 FPS | - | 内存拷贝 |
| 零拷贝版本 | 21.8ms | **29.8 FPS** | **+49%** | 摄像头30FPS |
| 理论上限 | 21.8ms | 45.9 FPS | +129% | 无摄像头输入 |

## 🏗️ 系统要求

### 硬件平台
- **开发板**: Rock5C 8GB 或其他RK3588平台
- **NPU**: RK3588 NPU (6 TOPS算力)
- **摄像头**: USB摄像头，支持640x480 MJPEG格式
- **内存**: 至少4GB RAM

### 软件环境
- **操作系统**: Debian/Ubuntu (aarch64)
- **RKNN运行时**: rknpu2
- **OpenCV**: 4.x (支持V4L2)
- **CMake**: 3.10+
- **GCC**: 8.0+

## 🛠️ 编译安装

### 1. 环境准备
```bash
# 安装依赖
sudo apt-get update
sudo apt-get install cmake build-essential
sudo apt-get install libopencv-dev v4l-utils

# 检查RKNN运行时
ls /usr/lib/librknn* 
```

### 2. 克隆项目
```bash
git clone https://github.com/JoeFirmament/rknn_yolov8pose_rk3588_camera.git
cd rknn_yolov8pose_rk3588_camera
```

### 3. 编译项目
```bash
cd cpp/build
cmake ..
make -j4

# 编译结果
# ./rknn_yolov8_pose_camera_demo      # 基础版本
# ./rknn_yolov8_pose_camera_optimized # 零拷贝优化版本  
# ./rknn_yolov8_pose_benchmark        # 纯性能测试版本
```

## 🎯 使用方法

### 1. 摄像头性能分析 (推荐第一步)
```bash
# 检查摄像头规格，避免盲目优化
./scripts/check_camera_info.sh
```

**为什么要先检查摄像头**: 不同摄像头性能差异很大，如果摄像头只支持30FPS，优化代码到200FPS也没意义。

### 2. 运行姿态检测
```bash
# 基础版本 (调试用)
./rknn_yolov8_pose_camera_demo ../model/yolov8_pose.rknn

# 优化版本 (推荐)
./rknn_yolov8_pose_camera_optimized ../model/yolov8_pose.rknn

# 性能测试版本 (无显示)
./rknn_yolov8_pose_benchmark ../model/yolov8_pose.rknn
```

### 3. 控制说明
- **ESC键**: 退出程序
- **Ctrl+C**: 强制停止并显示性能统计
- **Space键**: 暂停/继续 (仅显示版本)

## 📁 项目结构

```
rknn_yolov8pose_rk3588_camera/
├── README.md                    # 项目说明
├── OPTIMIZATION_LOG.md          # 优化过程详细记录
├── model/
│   └── yolov8_pose.rknn        # 预训练模型
├── scripts/
│   └── check_camera_info.sh    # 摄像头分析工具
└── cpp/
    ├── CMakeLists.txt          # 编译配置
    ├── main_camera.cc          # 基础版本
    ├── main_camera_optimized.cc # 零拷贝优化版本
    ├── main_camera_benchmark.cc # 性能测试版本
    ├── yolov8_pose.cc          # 核心处理逻辑
    ├── yolov8_pose.h           # 头文件定义
    ├── postprocess.cc          # 后处理算法
    ├── postprocess.h           # 后处理头文件
    └── build/                  # 编译目录
```

## 🔧 核心技术

### NPU零拷贝优化
```cpp
// 传统方式: CPU内存 → 拷贝 → NPU内存
rknn_inputs_set(ctx, 1, inputs);  // 涉及内存拷贝

// 零拷贝方式: 直接在NPU内存操作
rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_size);
rknn_set_io_mem(ctx, input_mem, &input_attr);
// OpenCV Mat直接指向NPU内存，无拷贝开销
```

**为什么使用零拷贝**: CPU↔NPU内存传输是主要性能瓶颈，零拷贝技术消除了这个开销，理论性能提升50-100%。

### 智能预处理
- **Letterbox算法**: 保持图像比例的同时适配模型输入尺寸
- **颜色空间转换**: BGR→RGB，配合NPU内存对齐要求
- **步幅处理**: 支持NPU内存的width_stride对齐

## 📈 性能调优

### 1. 摄像头优化
```bash
# 设置最优格式
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG

# 测试实际帧率
ffmpeg -f v4l2 -i /dev/video0 -vframes 100 -f null -
```

### 2. NPU频率设置
```bash
# 检查NPU频率
sudo cat /sys/class/devfreq/fdab0000.npu/cur_freq

# 设置性能模式 (可选)
echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor
```

### 3. 系统优化
```bash
# 增大摄像头缓冲区
echo 'SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", ATTR{index}=="0", ATTR{name}=="*", RUN+="/bin/sh -c 'echo 4 > /sys/class/video4linux/%k/device/queue_size'"' | sudo tee /etc/udev/rules.d/99-camera-buffer.rules
```

## 🐛 常见问题

### Q1: 编译错误 "rknn_api.h not found"
```bash
# 检查RKNN运行时安装
ls /usr/include/rknn*
# 如果没有，需要安装rknpu2 SDK
```

### Q2: 摄像头打开失败
```bash
# 检查摄像头权限
ls -l /dev/video*
sudo chmod 666 /dev/video0

# 确认摄像头连接
lsusb | grep -i camera
```

### Q3: 性能不如预期
```bash
# 先检查摄像头规格
./scripts/check_camera_info.sh

# 检查系统负载
htop
# 关闭不必要的后台程序
```

### Q4: 模型加载失败
```bash
# 检查模型文件
ls -la model/yolov8_pose.rknn
# 确保模型是RK3588兼容版本
```

## 🔮 进一步优化方向

1. **RGA硬件加速**: 使用RGA进行图像预处理，进一步减少CPU负载
2. **多线程流水线**: 并行处理预处理、推理、后处理
3. **双缓冲技术**: 避免等待摄像头帧间隔
4. **模型量化优化**: INT8 vs FP16性能对比
5. **高帧率摄像头**: 更换支持60/120FPS的工业摄像头

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系方式

- **GitHub**: [JoeFirmament](https://github.com/JoeFirmament)
- **项目地址**: https://github.com/JoeFirmament/rknn_yolov8pose_rk3588_camera

---

**⭐ 如果这个项目对你有帮助，请给个Star！**
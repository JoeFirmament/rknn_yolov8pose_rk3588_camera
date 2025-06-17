# YOLOv8 Pose RKNN 零拷贝推理调试总结

## 1. 背景
- 平台：RK3588 NPU
- 任务：YOLOv8 pose 检测，优化零拷贝输入，提升推理性能
- 现象：优化后推理速度提升，但检测框和关键点出现整体 Y 轴偏移

## 2. 主要调试过程

### 2.1 零拷贝 letterbox 输入验证
- 分别用 CPU 版和 NPU 零拷贝版 letterbox 处理同一张图片（pose.jpg），保存为 `cpu_input_letterbox.png` 和 `npu_input_letterbox.png`。
- 结论：两者内容完全一致，说明 letterbox 预处理、stride 步幅、ROI 写入等输入相关流程无误。

### 2.2 stride/padding 机制理解
- NPU 输入内存每行实际分配 w_stride 像素，padding 区域用 114 填充。
- `npu_input_fullstride.png` 展示了全 stride 内存，padding 区域与有效区域不同属正常现象。

### 2.3 后处理与坐标反变换一致性检查
- main_camera.cc 和 main_camera_optimized.cc 均调用同一个 postprocess.cc，后处理算法和公式完全一致。
- letterbox 参数（scale, x_pad/y_pad）传递和初始化方式一致。
- OpenCV 显示前，box/关键点反变换公式一致。

### 2.4 Y 轴偏移定位
- 现象：检测框和关键点整体向上偏移 80 像素。
- 通过调试打印，发现 postprocess 已经减去 y_pad，OpenCV 显示前又减了一次 y_offset，导致多减了一次。
- 修正方法：OpenCV 显示前，box/关键点反变换时，y 方向不再减 y_offset，只用 postprocess 输出的原值。

## 3. 关键结论与经验
- 零拷贝 letterbox 输入和 stride/padding 机制必须严格按 RKNN API 要求实现。
- letterbox 参数（scale, x_pad, y_pad）在前处理、后处理、显示各环节必须保持一致且只做一次反变换。
- Y 轴整体偏移常见根因是 y_pad/y_offset 被多减或少减，需逐步打印和对比每一步参数。
- 调试过程中，保存 letterbox 输入图片、详细打印参数和中间结果是定位问题的高效手段。

## 4. 参考调试代码片段
```cpp
// OpenCV 显示前，修正后的反变换
int x1 = (det_result->box.left - x_offset) / scale;
int y1 = det_result->box.top; // 不再减 y_offset
// 关键点同理
keypoints[j][0] = (det_result->keypoints[j][0] - x_offset) / scale;
keypoints[j][1] = det_result->keypoints[j][1];
```

## 5. 建议
- 任何涉及 letterbox、stride、坐标反变换的优化，务必保存中间图片和详细日志，逐步定位。
- 团队内部可用本 md 文档作为 RKNN 零拷贝 pose 推理调优的经验总结。 
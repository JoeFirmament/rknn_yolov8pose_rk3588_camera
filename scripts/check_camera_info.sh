#!/bin/bash

# ========================================
# 摄像头信息查询脚本
# 用途: 快速查看摄像头支持的格式、分辨率和帧率
# 为什么需要: 不同摄像头性能差异很大，影响整体FPS上限
# ========================================

echo "=== 摄像头信息查询工具 ==="
echo "平台: Rock5C 8GB, RK3588 NPU"
echo "用途: 优化YOLOv8性能前先了解摄像头限制"
echo ""

# 检查v4l2-ctl是否可用
if ! command -v v4l2-ctl &> /dev/null; then
    echo "错误: v4l2-ctl未安装"
    echo "安装命令: sudo apt-get install v4l-utils"
    exit 1
fi

# 1. 列出所有摄像头设备
echo "1. 可用的摄像头设备:"
echo "========================="
v4l2-ctl --list-devices
echo ""

# 2. 获取所有video设备
video_devices=$(ls /dev/video* 2>/dev/null)

if [ -z "$video_devices" ]; then
    echo "未找到摄像头设备"
    exit 1
fi

# 3. 逐个分析每个设备
for device in $video_devices; do
    echo "2. 分析设备: $device"
    echo "========================="
    
    # 检查设备是否可用
    if ! v4l2-ctl --device=$device --list-formats &> /dev/null; then
        echo "设备 $device 不可用或无权限访问"
        echo ""
        continue
    fi
    
    # 获取当前格式设置
    echo "当前设置:"
    v4l2-ctl --device=$device --get-fmt-video
    echo ""
    
    # 获取支持的格式和分辨率
    echo "支持的格式、分辨率和帧率:"
    echo "格式说明: MJPG=Motion JPEG(压缩), YUYV=YUV422(未压缩)"
    v4l2-ctl --device=$device --list-formats-ext
    echo ""
    
    # 为YOLOv8提供建议
    echo "YOLOv8优化建议 (针对 $device):"
    echo "-----------------------------"
    
    # 检查是否支持MJPG格式640x480
    mjpg_640x480=$(v4l2-ctl --device=$device --list-formats-ext | grep -A10 "MJPG" | grep -A5 "640x480")
    if [ ! -z "$mjpg_640x480" ]; then
        fps=$(echo "$mjpg_640x480" | grep "fps" | head -1 | grep -o '[0-9]*\.[0-9]*' | head -1)
        echo "✓ 支持MJPG 640x480，最大FPS: ${fps:-30}"
        echo "  推荐: 使用MJPG格式，CPU解压缩负载较低"
    else
        echo "✗ 不支持MJPG 640x480格式"
    fi
    
    # 检查YUYV格式
    yuyv_640x480=$(v4l2-ctl --device=$device --list-formats-ext | grep -A10 "YUYV" | grep -A5 "640x480")
    if [ ! -z "$yuyv_640x480" ]; then
        fps=$(echo "$yuyv_640x480" | grep "fps" | head -1 | grep -o '[0-9]*\.[0-9]*' | head -1)
        echo "✓ 支持YUYV 640x480，最大FPS: ${fps:-30}"
        echo "  注意: YUYV格式数据量大，可能影响传输速度"
    else
        echo "✗ 不支持YUYV 640x480格式"
    fi
    
    # 性能评估
    echo ""
    echo "性能预估:"
    max_fps=$(v4l2-ctl --device=$device --list-formats-ext | grep "640x480" -A2 | grep "fps" | grep -o '[0-9]*\.[0-9]*' | sort -nr | head -1)
    if [ ! -z "$max_fps" ]; then
        if (( $(echo "$max_fps >= 60" | bc -l) )); then
            echo "🟢 高性能摄像头 (${max_fps} FPS) - 适合实时应用"
        elif (( $(echo "$max_fps >= 30" | bc -l) )); then
            echo "🟡 标准摄像头 (${max_fps} FPS) - 基本满足需求"
        else
            echo "🔴 低速摄像头 (${max_fps} FPS) - 可能成为性能瓶颈"
        fi
    fi
    
    echo ""
    echo "======================================="
    echo ""
done

echo "3. 使用建议:"
echo "==========="
echo "• 如果摄像头FPS < 40，考虑更换高速摄像头"
echo "• NPU推理能力约45 FPS，摄像头是当前瓶颈"
echo "• 优先使用MJPG格式，减少数据传输量"
echo "• 可以降低分辨率换取更高帧率进行测试"
echo ""
echo "4. 测试命令:"
echo "==========="
echo "设置摄像头参数: v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG"
echo "查看实时帧率: ffmpeg -f v4l2 -i /dev/video0 -vframes 100 -f null - 2>&1 | grep fps" 
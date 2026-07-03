#!/usr/bin/env python3
from ultralytics import YOLO
import os

def train_model():
    """训练 YOLOv8 模型"""

    # 1. 加载预训练模型
    # 可选: yolov8n.pt (最快), yolov8s.pt (平衡), yolov8m.pt (精度最高)
    model = YOLO("yolov8s.pt")

    # 2. 训练参数配置
    train_args = {
        # 数据配置
        "data": "datasets/data.yaml",

        # 训练参数
        "epochs": 100,              # 训练轮数
        "imgsz": 640,               # 输入图像尺寸
        "batch": 16,                # 批次大小，根据显存调整 (8G显存用16, 4G用8)

        # 优化器参数
        "optimizer": "auto",        # 自动选择优化器
        "lr0": 0.01,                # 初始学习率
        "lrf": 0.01,                # 最终学习率 (相对初始学习率)

        # 数据增强
        "hsv_h": 0.015,             # HSV 色调增强
        "hsv_s": 0.7,               # HSV 饱和度增强
        "hsv_v": 0.4,               # HSV 明度增强
        "degrees": 0.0,             # 旋转角度
        "translate": 0.1,           # 平移
        "scale": 0.5,               # 缩放
        "fliplr": 0.5,              # 左右翻转
        "mosaic": 1.0,              # Mosaic 增强
        "mixup": 0.1,               # MixUp 增强

        # 训练控制
        "patience": 20,             # 早停耐心值，20轮无改善则停止
        "save": True,               # 保存最佳模型
        "save_period": 10,          # 每10轮保存一次

        # 设备
        "device": "mps",              # Apple Silicon GPU加速，无GPU用"cpu"

        # 输出配置
        "project": "runs/detect",   # 输出目录
        "name": "csgo_detect",      # 实验名称
        "exist_ok": False,          # 如果目录存在则报错

        # 其他
        "verbose": True,            # 详细输出
        "pretrained": True,         # 使用预训练权重
    }

    print("=" * 60)
    print("🚀 开始训练 CSGO 目标检测模型")
    print("=" * 60)
    print(f"📁 数据路径: {train_args['data']}")
    print(f"🔄 训练轮数: {train_args['epochs']}")
    print(f"📏 图像尺寸: {train_args['imgsz']}")
    print(f"📦 批次大小: {train_args['batch']}")
    print(f"🎯 预训练模型: YOLOv8s")
    print("=" * 60)

    # 3. 开始训练
    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"📊 结果保存位置: runs/detect/csgo_detect/")
    print(f"⚖️ 最佳权重: runs/detect/csgo_detect/weights/best.pt")
    print(f"📈 训练曲线: runs/detect/csgo_detect/results.png")
    print("=" * 60)

    return results

def export_model():
    """导出模型为不同格式"""

    print("\n📦 导出模型...")

    # 加载最佳模型
    model = YOLO("runs/detect/csgo_detect/weights/best.pt")

    # 导出为 ONNX (通用格式，可部署到多种平台)
    print("📤 导出 ONNX 格式...")
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)

    # 导出为 TorchScript (PyTorch 优化格式)
    print("📤 导出 TorchScript 格式...")
    model.export(format="torchscript", imgsz=640)

    print("\n✅ 模型导出完成！")
    print("📁 导出文件位置: runs/detect/csgo_detect/weights/")

def evaluate_model():
    """评估模型性能"""

    print("\n📊 评估模型...")

    model = YOLO("runs/detect/csgo_detect/weights/best.pt")

    # 在验证集上评估
    metrics = model.val(data="datasets/data.yaml", imgsz=640)

    print("\n" + "=" * 60)
    print("📈 模型性能指标")
    print("=" * 60)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSGO 目标检测训练脚本")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "export", "eval"],
                       help="运行模式: train(训练), export(导出), eval(评估)")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "export":
        export_model()
    elif args.mode == "eval":
        evaluate_model()

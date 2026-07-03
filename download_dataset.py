#!/usr/bin/env python3
"""
CSGO 数据集下载脚本
从 Roboflow Universe 下载 csgo-minel 数据集
"""

from roboflow import Roboflow
from dotenv import load_dotenv
import os

# 从 .env 文件加载环境变量
load_dotenv('.env')

# 配置
WORKSPACE = "csgo-nqkf0"
PROJECT = "csgo-minel"
VERSION = 10  # 数据集版本号
DOWNLOAD_FORMAT = "yolov8"
DOWNLOAD_DIR = "./datasets"

def download_dataset():
    """下载数据集"""
    # 确保下载目录存在
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # 从 .env 读取 API Key
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("❌ 错误：请设置 ROBOFLOW_API_KEY 环境变量")
        print("\n获取 API Key 步骤：")
        print("1. 访问 https://app.roboflow.com 注册账号")
        print("2. 登录后进入 Settings → API Key")
        print("3. 复制 API Key")
        print("\n设置方式（二选一）：")
        print("方式1 - 环境变量:")
        print("  export ROBOFLOW_API_KEY='your_api_key_here'")
        print("\n方式2 - 直接修改本脚本:")
        print("  将 api_key = os.getenv('ROBOFLOW_API_KEY') 改为")
        print("  api_key = 'your_api_key_here'")
        return None

    print(f"🚀 开始连接 Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"📦 正在获取项目信息: {WORKSPACE}/{PROJECT}")
    project = rf.workspace(WORKSPACE).project(PROJECT)

    print(f"📥 正在下载数据集（版本 {VERSION}，格式: {DOWNLOAD_FORMAT}）...")
    dataset = project.version(VERSION).download(
        DOWNLOAD_FORMAT,
        location=DOWNLOAD_DIR
    )

    print(f"\n✅ 数据集下载完成！")
    print(f"📁 保存位置: {os.path.abspath(DOWNLOAD_DIR)}")
    print(f"\n数据集结构:")
    print(f"  {DOWNLOAD_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── labels/")
    print(f"  ├── valid/")
    print(f"  │   ├── images/")
    print(f"  │   └── labels/")
    print(f"  └── data.yaml")

    return dataset

if __name__ == "__main__":
    download_dataset()

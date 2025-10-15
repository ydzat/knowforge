import os
import cv2
import numpy as np
import torch
import easyocr
import time

def debug_ocr():
    print("\n===== OCR 调试工具 =====")
    
    # 检查 GPU 状态
    print("\n1. 检查 GPU 状态:")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 图片路径
    image_path = "input/images/ocr_test_sample.png"
    print(f"\n2. 测试图像: {image_path}")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像 {image_path} 不存在!")
        return
    
    # 读取和显示图像信息
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}!")
        return
    
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}像素")
    print(f"图像类型: {img.dtype}, 通道数: {img.shape[2] if len(img.shape) > 2 else 1}")
    
    # 保存灰度预处理图像用于调试
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    debug_dir = "workspace/debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "debug_gray_image.png")
    cv2.imwrite(debug_path, gray)
    print(f"灰度预处理图像已保存到 {debug_path}")
    
    # 初始化 OCR 读取器
    print("\n3. 初始化 EasyOCR (可能需要下载模型):")
    start_time = time.time()
    
    # 测试用GPU模式
    print("\n--- 使用GPU模式 ---")
    try:
        reader_gpu = easyocr.Reader(['ch_sim', 'en'], gpu=True, 
                             model_storage_directory='workspace/ocr_models',
                             download_enabled=True,
                             verbose=True)  # 启用详细输出
        
        print("\n4. 执行 OCR 识别 (GPU):")
        results_gpu = reader_gpu.readtext(img)
        
        print(f"识别结果数量: {len(results_gpu)}")
        for i, (bbox, text, conf) in enumerate(results_gpu):
            print(f"结果 {i+1}: 文本='{text}', 置信度={conf:.4f}, 位置={bbox}")
    except Exception as e:
        print(f"GPU模式OCR执行失败: {str(e)}")
    
    # 测试用CPU模式
    print("\n--- 使用CPU模式 ---")
    try:
        reader_cpu = easyocr.Reader(['ch_sim', 'en'], gpu=False, 
                             model_storage_directory='workspace/ocr_models',
                             download_enabled=True)
        
        print("\n5. 执行 OCR 识别 (CPU):")
        results_cpu = reader_cpu.readtext(img)
        
        print(f"识别结果数量: {len(results_cpu)}")
        for i, (bbox, text, conf) in enumerate(results_cpu):
            print(f"结果 {i+1}: 文本='{text}', 置信度={conf:.4f}, 位置={bbox}")
    except Exception as e:
        print(f"CPU模式OCR执行失败: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"\n总执行时间: {elapsed:.2f}秒")
    print("\n===== OCR 调试结束 =====")

if __name__ == "__main__":
    debug_ocr()

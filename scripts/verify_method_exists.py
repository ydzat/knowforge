#!/usr/bin/env python3
"""
最小化测试 - 验证 enhance_ocr_with_memory 方法
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("开始测试OCR增强功能")

try:
    # 尝试导入需要测试的类
    from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
    print("成功导入AdvancedMemoryManager类")
    
    # 检查enhance_ocr_with_memory方法是否存在
    if hasattr(AdvancedMemoryManager, 'enhance_ocr_with_memory'):
        print("enhance_ocr_with_memory方法存在")
    else:
        print("错误: enhance_ocr_with_memory方法不存在!")
        methods = [m for m in dir(AdvancedMemoryManager) if callable(getattr(AdvancedMemoryManager, m)) and not m.startswith('_')]
        print(f"可用方法: {methods}")

except Exception as e:
    print(f"导入错误: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")

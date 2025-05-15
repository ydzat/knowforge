#!/usr/bin/env python3
"""
集成测试OCR与内存增强功能
"""
import os
import sys

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # 导入需要测试的函数 - 直接导入这个函数以便单独测试
    from src.note_generator.advanced_memory_manager import enhance_ocr_with_memory
    print("成功导入enhance_ocr_with_memory函数")
except ImportError as e:
    print(f"导入enhance_ocr_with_memory函数失败: {e}")
    sys.exit(1)

try:
    from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
    print("成功导入AdvancedMemoryManager类")
except ImportError as e:
    print(f"导入AdvancedMemoryManager类失败: {e}")
    # 不退出，继续测试

print("\n正在初始化AdvancedMemoryManager...")

# 创建工作目录
workspace_dir = "./workspace"
os.makedirs(workspace_dir, exist_ok=True)

try:
    # 创建AdvancedMemoryManager实例，使用本地文件夹
    memory_manager = AdvancedMemoryManager(
        chroma_db_path=os.path.join(workspace_dir, "test_memory"),
        config={
            "capacity": 100,
            "working_memory_capacity": 10
        }
    )
    print("成功创建AdvancedMemoryManager实例")
except Exception as e:
    print(f"创建AdvancedMemoryManager实例失败: {e}")
    sys.exit(1)

# 测试添加知识
try:
    print("\n向记忆系统中添加测试知识...")
    knowledge_id = memory_manager.add_knowledge(
        "KnowForge是一个高级内存管理系统，可以增强OCR结果",
        {"source": "test", "importance": "0.8"}
    )
    print(f"成功添加知识，ID: {knowledge_id}")
except Exception as e:
    print(f"添加知识失败: {e}")
    sys.exit(1)

# 测试OCR增强功能
try:
    print("\n测试OCR结果增强功能...")
    ocr_text = "KnovForge是一个高级内存蕾理系统，可以增强OCR结采"  # 故意添加错误
    context = "测试OCR增强"
    
    print(f"原始OCR文本: {ocr_text}")
    
    # 尝试使用记忆系统增强OCR结果
    from unittest.mock import patch, MagicMock
    
    with patch('src.note_generator.llm_caller.LLMCaller') as mock_llm:
        mock_instance = MagicMock()
        mock_instance.call_model.return_value = "KnowForge是一个高级内存管理系统，可以增强OCR结果"
        mock_llm.return_value = mock_instance
        
        result = memory_manager.enhance_ocr_with_memory(ocr_text, context)
        
        print("\n增强结果:")
        print(f"  原始文本: {result.get('original', '未获取')}")
        print(f"  增强文本: {result.get('enhanced', '未增强')}")
        print(f"  置信度: {result.get('confidence', 0):.2f}")
        print(f"  参考资料数量: {len(result.get('references', []))}")
        
        if mock_instance.call_model.called:
            print("\nLLM调用成功，完成文本校正")
        else:
            print("\nLLM未被调用，请检查实现")
        
except Exception as e:
    print(f"测试OCR结果增强功能失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n集成测试完成!")

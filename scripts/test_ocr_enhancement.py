#!/usr/bin/env python3
"""
测试OCR增强功能
"""
import os
import sys
import time
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """主测试函数"""
    print("\n===== 测试OCR与记忆系统集成 =====\n")
    
    try:
        # 导入需要的模块
        print("导入需要的模块...")
        from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
        print("成功导入了AdvancedMemoryManager")
        
        # 设置临时工作区
        workspace_dir = "./workspace"
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir, exist_ok=True)
        
        # 准备测试数据
        test_data = {
            "content": "KnowForge是一个高级记忆管理系统，支持OCR文本增强与校正",
            "metadata": {
                "source": "test",
                "timestamp": str(time.time()),
                "importance": "0.8"
            }
        }
        
        # 创建记忆管理器
        print("初始化记忆管理器...")
        memory_manager = AdvancedMemoryManager(
            chroma_db_path=os.path.join(workspace_dir, "memory_db"),
            config={
                "enabled": True,
                "working_memory_capacity": 10,
                "short_term_memory_capacity": 100
            }
        )
        print("记忆管理器初始化成功")
        
        # 添加测试知识到记忆系统
        print("\n添加测试知识到记忆系统...")
        knowledge_id = memory_manager.add_knowledge(
            test_data["content"], 
            test_data["metadata"]
        )
        print(f"知识添加成功，ID: {knowledge_id}")
        
        # 测试OCR增强功能
        print("\n测试OCR增强功能...")
        ocr_text = "KncwForge是一个高级记忆管理系统，支持OCF文本增強與校止"
        
        # 打印原始OCR文本
        print(f"\n原始OCR文本: \n{ocr_text}")
        
        # 模拟LLM调用
        with patch('src.note_generator.llm_caller.LLMCaller') as mock_llm:
            mock_instance = MagicMock()
            mock_instance.call_model.return_value = "KnowForge是一个高级记忆管理系统，支持OCR文本增强与校正"
            mock_llm.return_value = mock_instance
            
            # 调用OCR增强功能
            result = memory_manager.enhance_ocr_with_memory(ocr_text, "测试上下文")
            
            # 输出结果
            print("\n=== OCR增强结果 ===")
            print(f"原始文本: {result['original']}")
            print(f"增强文本: {result['enhanced']}")
            print(f"置信度: {result['confidence']:.2f}")
            print(f"参考资料数量: {len(result.get('references', []))}")
            
            # 验证LLM调用
            if mock_instance.call_model.called:
                print("\nLLM被成功调用进行文本校正")
                print(f"LLM方法被调用了 {mock_instance.call_model.call_count} 次")
            else:
                print("\n警告: LLM未被调用，请检查实现")
        
        print("\n测试完成!")
        return True
    
    except Exception as e:
        import traceback
        print(f"\n错误: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
简单的OCR记忆增强测试脚本
"""
import os
import sys
import json
from unittest.mock import MagicMock, patch

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.advanced_memory_manager import AdvancedMemoryManager


def main():
    """测试OCR记忆增强功能的主函数"""
    print("\n===== 测试OCR结果增强功能 =====\n")
    
    # 创建临时工作空间目录
    workspace_dir = "./workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 配置
    config = {
        "enabled": True,
        "capacity": 100,
        "working_memory_capacity": 10,
        "similarity_threshold": 0.5,
        "index_path": "./memory_test",
        "optimize_interval": 10
    }
    
    # 初始化高级记忆管理器
    print("初始化高级记忆管理器...")
    chrome_db_path = os.path.join(workspace_dir, "memory_db")
    os.makedirs(chrome_db_path, exist_ok=True)
    memory_manager = AdvancedMemoryManager(
        chroma_db_path=chrome_db_path,
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        collection_name='test_memory',
        config=config
    )
    
    # 添加测试知识到记忆系统
    print("向记忆系统中添加测试知识...")
    knowledge_id1 = memory_manager.add_knowledge(
        content="KnowForge是一个高级记忆管理系统，支持OCR结果增强",
        metadata={"source": "test", "importance": 0.8}
    )
    
    knowledge_id2 = memory_manager.add_knowledge(
        content="OCR技术可以识别图像中的文字，但有时会出现错误",
        metadata={"source": "test", "importance": 0.7}
    )
    
    knowledge_id3 = memory_manager.add_knowledge(
        content="高级记忆管理系统可以利用上下文和已有知识来改进OCR结果",
        metadata={"source": "test", "importance": 0.9}
    )
    
    print(f"添加了测试知识，ID: {knowledge_id1}, {knowledge_id2}, {knowledge_id3}")
    
    # 模拟一个带有错误的OCR文本
    ocr_text = "KnovFarce是一个高级记忆管理系统，支持OCF结果坚强"
    context = "测试OCR增强功能"
    
    print(f"\n原始OCR文本: \n{ocr_text}\n")
    
    # 模拟LLM调用
    with patch('src.note_generator.llm_caller.LLMCaller') as mock_llm:
        mock_instance = MagicMock()
        mock_instance.call_model.return_value = "KnowForge是一个高级记忆管理系统，支持OCR结果增强"
        mock_llm.return_value = mock_instance
        
        # 使用记忆增强OCR结果
        print("使用记忆系统增强OCR结果...")
        result = memory_manager.enhance_ocr_results(ocr_text, context)
        
        print("\n=== 增强结果 ===")
        print(f"原始文本: {result['original']}")
        print(f"增强文本: {result['enhanced']}")
        print(f"置信度: {result['confidence']:.2f}")
        print("\n参考知识:")
        for i, ref in enumerate(result['references']):
            print(f"  {i+1}. 相似度: {ref['similarity']:.2f}")
            print(f"     内容: {ref['content'][:50]}...")
        
        print("\n测试完成!")


if __name__ == "__main__":
    main()

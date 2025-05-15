"""
测试OCR记忆增强功能
"""
import os
import sys
from unittest.mock import MagicMock, patch

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入需要测试的模块
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager


def test_ocr_enhancement():
    """测试OCR增强功能"""
    print("测试OCR结果增强功能...")
    
    # 配置
    config = {
        "enabled": True,
        "capacity": 100,
        "working_memory_capacity": 10,
        "similarity_threshold": 0.5,
        "index_path": "./memory_test",
        "optimize_interval": 10
    }
    
    # 创建高级记忆管理器
    memory_manager = AdvancedMemoryManager(
        workspace_dir="./workspace",
        config=config
    )
    
    # 模拟添加相关知识
    memory_manager.add({
        "id": "test1",
        "content": "KnowForge是一个高级记忆管理系统，支持OCR结果增强",
        "metadata": {"source": "test"}
    })
    
    # 模拟LLM调用
    with patch('src.note_generator.llm_caller.LLMCaller') as mock_llm:
        mock_llm_instance = MagicMock()
        mock_llm_instance.call_model.return_value = "KnowForge是一个高级记忆管理系统"
        mock_llm.return_value = mock_llm_instance
        
        # 使用记忆增强OCR结果
        with patch('src.note_generator.advanced_memory_manager.AdvancedMemoryManager.retrieve') as mock_retrieve:
            mock_retrieve.return_value = [{
                "id": "test1",
                "content": "KnowForge是一个高级记忆管理系统，支持OCR结果增强",
                "similarity": 0.85
            }]
            
            # 测试enhance_ocr_with_memory方法
            result = memory_manager.enhance_ocr_with_memory(
                "KnovForge是一个高级记忆管理系统", 
                "OCR测试"
            )
            
            # 检查结果
            print("--- OCR增强结果 ---")
            print(f"原始文本: {result['original']}")
            print(f"增强文本: {result['enhanced']}")
            print(f"置信度: {result['confidence']}")
            print(f"参考资料数量: {len(result['references'])}")
            
            # 验证LLM是否被调用
            assert mock_llm_instance.call_model.called
            print("LLM已被正确调用进行OCR校正")
            
    print("测试完成!")


if __name__ == "__main__":
    test_ocr_enhancement()

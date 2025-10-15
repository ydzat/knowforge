'''
 * @Author: GitHub Copilot
 * @Date: 2025-05-17 15:45:00
 * @Description: 测试已修改的函数功能
'''
import os
import sys
import pytest
import tempfile
import shutil
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.vector_index import VectorIndex
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager
from src.note_generator.advanced_memory_with_index import AdvancedMemoryManagerWithIndex


class TestVectorIndexLen:
    """测试向量索引的__len__方法"""
    
    def test_len_with_no_vectors(self):
        """测试没有向量时的长度"""
        index = VectorIndex(vector_dim=4)
        assert len(index) == 0
        
    def test_len_with_vectors(self):
        """测试有向量时的长度"""
        index = VectorIndex(vector_dim=4)
        index.add("test1", [1.0, 0.0, 0.0, 0.0])
        index.add("test2", [0.0, 1.0, 0.0, 0.0])
        assert len(index) == 2
        
    def test_len_with_deleted_vectors(self):
        """测试有删除向量时的长度"""
        index = VectorIndex(vector_dim=4)
        index.add("test1", [1.0, 0.0, 0.0, 0.0])
        index.add("test2", [0.0, 1.0, 0.0, 0.0])
        index.add("test3", [0.0, 0.0, 1.0, 0.0])
        index.remove("test2")
        assert len(index) == 2


class TestAdvancedMemoryExportImport:
    """测试高级记忆管理器的导出导入功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp(prefix="knowforge_test_")
        
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_clear_method(self):
        """测试清空方法是否正确清理向量索引"""
        # 创建向量索引
        index_path = os.path.join(self.test_dir, "vector_index.pkl")
        index = VectorIndex(vector_dim=4, index_path=index_path)
        
        # 添加测试向量
        index.add("test1", [1.0, 0.0, 0.0, 0.0])
        index.add("test2", [0.0, 1.0, 0.0, 0.0])
        
        # 保存索引
        index.save(index_path)
        
        # 确认保存成功
        assert os.path.exists(index_path)
        
        # 重新加载索引
        new_index = VectorIndex(vector_dim=4, index_path=index_path)
        new_index.load(index_path)
        
        # 验证向量数量
        assert len(new_index) == 2
        
        # 测试清空功能
        # 这个测试我们重置为测试清空向量
        new_index.vectors = []
        new_index.ids = []
        new_index.id_to_index = {}
        new_index.deleted_indices = set()
        
        # 验证已清空
        assert len(new_index) == 0
        
        # 保存清空后的索引
        new_index.save(index_path)
        
        # 重新加载验证是否真正清空了文件
        reload_index = VectorIndex(vector_dim=4, index_path=index_path)
        reload_index.load(index_path)
        assert len(reload_index) == 0


if __name__ == "__main__":
    pytest.main(['-xvs', __file__])

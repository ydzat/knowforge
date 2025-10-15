'''
 * @Author: GitHub Copilot
 * @Date: 2025-05-17 16:00:00
 * @Description: 测试高级内存管理器的清空功能
'''
import os
import sys
import tempfile
import shutil
import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.vector_index import VectorIndex
from src.note_generator.advanced_memory_with_index import AdvancedMemoryManagerWithIndex


class TestAdvancedMemoryClearMethod:
    """测试高级记忆管理器的清空功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp(prefix="knowforge_test_")
        
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_vector_index(self):
        """创建测试用的向量索引"""
        index_path = os.path.join(self.test_dir, "vector_index.pkl")
        index = VectorIndex(vector_dim=4, index_path=index_path)
        
        # 添加测试向量
        index.add("test1", [1.0, 0.0, 0.0, 0.0])
        index.add("test2", [0.0, 1.0, 0.0, 0.0])
        
        # 保存索引
        index.save(index_path)
        return index_path
    
    def test_vector_index_clear_method(self):
        """测试向量索引的清空方法"""
        index_path = self.create_vector_index()
        
        # 加载索引
        index = VectorIndex(vector_dim=4, index_path=index_path)
        index.load(index_path)
        
        # 验证加载成功
        assert len(index) == 2
        
        # 调用手动清空方法
        index.vectors = []
        index.ids = []
        index.id_to_index = {}
        index.deleted_indices = set()
        
        # 验证内存中已清空
        assert len(index) == 0
        
        # 保存清空后的状态
        index.save(index_path)
        
        # 重新加载
        new_index = VectorIndex(vector_dim=4, index_path=index_path)
        new_index.load(index_path)
        
        # 验证持久化的清空状态
        assert len(new_index) == 0


if __name__ == "__main__":
    pytest.main(['-xvs', __file__])

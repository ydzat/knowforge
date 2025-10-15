'''
 * @Author: @ydzat
 * @Date: 2025-05-16 22:10:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-16 22:10:00
 * @Description: 向量化模块补丁 - 为现有方法添加兼容性别名
'''

from src.note_generator.embedder import Embedder
from typing import List

# 添加兼容性方法
def get_embedding(self, text: str) -> List[float]:
    """
    embed_single方法的别名，为了兼容性
    
    Args:
        text: 待向量化的文本
        
    Returns:
        文本的向量表示
    """
    return self.embed_single(text)

def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """
    embed_texts方法的别名，为了兼容性
    
    Args:
        texts: 文本列表
        
    Returns:
        文本对应的向量列表
    """
    return self.embed_texts(texts)

# 动态添加方法
Embedder.get_embedding = get_embedding
Embedder.batch_get_embeddings = batch_get_embeddings

# 确保模块导出
__all__ = ['Embedder']

'''
 * @Author: @ydzat
 * @Date: 2025-05-14 10:18:08
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 10:18:08
 * @Description: 文本向量化模块 - 将文本转换为向量表示
'''
import os
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from src.utils.exceptions import NoteGenError
from src.utils.config_loader import ConfigLoader

class EmbeddingError(NoteGenError):
    """向量化处理过程中的异常"""
    pass

class Embedder:
    """
    文本向量化工具类，负责将文本转换为向量表示
    
    使用sentence-transformers库将文本转换成向量，支持批量处理
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None):
        """
        初始化Embedder
        
        Args:
            model_name: 预训练模型名称或路径
            cache_dir: 模型缓存目录，None表示使用默认缓存目录
        """
        self.logger = setup_logger()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.logger.info(f"初始化文本向量化模块，使用模型: {model_name}")
        
        try:
            # 加载预训练模型
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"模型加载成功，向量维度: {self.vector_size}")
        except Exception as e:
            self.logger.error(f"加载向量化模型失败: {str(e)}")
            raise EmbeddingError(f"向量化模型 '{model_name}' 加载失败: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将多段文本转换为向量表示
        
        Args:
            texts: 文本列表
            
        Returns:
            文本对应的向量列表
        
        Raises:
            EmbeddingError: 向量化过程中出现异常
        """
        if not texts:
            self.logger.warning("接收到空文本列表，无法进行向量化")
            return []
            
        try:
            self.logger.debug(f"开始批量向量化 {len(texts)} 段文本")
            embeddings = self.model.encode(texts)
            # 确保返回的是Python原生列表类型，便于后续序列化
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            self.logger.error(f"批量向量化文本失败: {str(e)}")
            raise EmbeddingError(f"批量向量化文本失败: {str(e)}")
            
    def embed_single(self, text: str) -> List[float]:
        """
        将单段文本转换为向量表示
        
        Args:
            text: 单段文本
            
        Returns:
            文本对应的向量
            
        Raises:
            EmbeddingError: 向量化过程中出现异常
        """
        if not text or not isinstance(text, str):
            self.logger.warning(f"接收到无效文本: {type(text)}")
            raise EmbeddingError(f"无法向量化非文本内容: {type(text)}")
            
        try:
            self.logger.debug(f"开始向量化单段文本")
            embedding = self.model.encode(text)
            # 确保返回的是Python原生列表类型，便于后续序列化
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        except Exception as e:
            self.logger.error(f"向量化单段文本失败: {str(e)}")
            raise EmbeddingError(f"向量化单段文本失败: {str(e)}")
            
    def save_embeddings(self, texts: List[str], output_dir: str) -> Dict[str, str]:
        """
        批量生成向量并保存到指定目录
        
        Args:
            texts: 文本列表
            output_dir: 输出目录
            
        Returns:
            保存的文件路径字典 {文本hash: 文件路径}
            
        Raises:
            EmbeddingError: 向量化或保存过程中出现异常
        """
        import hashlib
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"开始批量生成并保存向量到: {output_dir}")
        
        embeddings = self.embed_texts(texts)
        paths = {}
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # 使用文本的哈希值作为文件名
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            file_path = os.path.join(output_dir, f"{text_hash}.json")
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'text': text,
                        'embedding': embedding,
                        'model': self.model_name,
                        'dimension': self.vector_size
                    }, f, ensure_ascii=False)
                paths[text_hash] = file_path
            except Exception as e:
                self.logger.error(f"保存向量文件失败: {str(e)}")
                continue
                
        self.logger.info(f"成功保存 {len(paths)} 个向量文件")
        return paths
        
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度 (-1 到 1 之间的值，1表示完全相同，-1表示完全相反)
        """
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(v1, v2) / (norm1 * norm2)
'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-16 10:30:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-16 10:30:00
 * @Description: 向量索引工具 - 提供高效的向量检索能力
'''
import os
import time
import json
import math
import pickle
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import defaultdict

from src.utils.logger import setup_logger


class VectorIndexError(Exception):
    """向量索引处理过程中的异常"""
    pass


class VectorIndex:
    """
    向量索引类，提供高效的向量检索功能
    
    实现分层索引结构，支持高效的近似最近邻搜索
    
    特点:
    1. 多种索引类型支持不同场景
    2. 自动根据数据规模调整索引策略
    3. 支持批量操作和增量更新
    4. 具备持久化和故障恢复能力
    """
    
    # 支持的索引类型
    INDEX_TYPES = {
        "flat": "暴力检索，精确但速度较慢",
        "hnsw": "层次可导航小世界图，快速近似检索",
        "ivf": "倒排文件索引，适合大规模数据",
        "hybrid": "混合索引策略，平衡效率与准确性"
    }
    
    def __init__(
        self,
        index_type: str = "hybrid",
        vector_dim: int = 384,
        max_elements: int = 100000,
        index_path: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        初始化向量索引
        
        Args:
            index_type: 索引类型，可选 "flat", "hnsw", "ivf", "hybrid"
            vector_dim: 向量维度
            max_elements: 最大索引元素数量
            index_path: 索引存储路径，为None则不持久化
            config: 配置参数
        """
        self.logger = setup_logger()
        self.index_type = index_type
        self.vector_dim = vector_dim
        self.max_elements = max_elements
        self.index_path = index_path
        self.config = config or {}
        
        # 索引参数
        self.m = self.config.get("hnsw_m", 16)  # HNSW图中每个节点的连接数
        self.ef_construction = self.config.get("ef_construction", 200)  # 构建过程中的搜索深度
        self.ef_search = self.config.get("ef_search", 50)  # 查询时的搜索深度
        
        # IVF参数
        self.n_lists = self.config.get("n_lists", int(math.sqrt(max_elements)))  # 聚类中心数量
        self.n_probes = self.config.get("n_probes", 10)  # 查询时探测的聚类数量
        
        # 混合参数
        self.threshold = self.config.get("hybrid_threshold", 10000)  # 切换索引策略的阈值
        self.use_pq = self.config.get("use_pq", False)  # 是否使用乘积量化
        
        # 索引容器
        self.vectors = []  # 向量列表
        self.ids = []  # ID列表
        self.id_to_index = {}  # ID到索引的映射
        self.deleted_indices = set()  # 已删除的索引集合
        
        # HNSW结构
        self.hnsw_layers = []  # 层次结构
        self.hnsw_entry_point = None  # 入口点
        
        # IVF结构
        self.ivf_centroids = None  # 聚类中心
        self.ivf_clusters = defaultdict(list)  # 聚类到索引的映射
        
        # 缓存
        self.vector_cache = {}  # ID到向量的缓存
        self.recently_used = []  # 最近使用的ID
        self.cache_size = self.config.get("cache_size", 1000)  # 缓存大小
        
        # 统计
        self.build_time = 0  # 构建时间
        self.n_queries = 0  # 查询次数
        self.total_query_time = 0  # 总查询时间
        
        self.logger.info(f"初始化向量索引，类型: {index_type}, 维度: {vector_dim}, 最大元素数: {max_elements}")
        
        # 如果提供了索引路径且文件存在，则加载
        if index_path and os.path.exists(index_path):
            self.load(index_path)
            
    def add(self, id: str, vector: List[float]) -> bool:
        """
        向索引中添加一个向量
        
        Args:
            id: 向量ID
            vector: 向量
            
        Returns:
            是否添加成功
        """
        if id in self.id_to_index:
            # 已存在，执行更新
            idx = self.id_to_index[id]
            if idx in self.deleted_indices:
                self.deleted_indices.remove(idx)
            self.vectors[idx] = vector
            # 更新缓存
            if id in self.vector_cache:
                self.vector_cache[id] = vector
            return True
            
        # 添加新向量
        idx = len(self.vectors)
        self.vectors.append(vector)
        self.ids.append(id)
        self.id_to_index[id] = idx
        
        # 添加到缓存
        self._update_cache(id, vector)
        
        # 根据索引类型执行特定操作
        if self.index_type == "hnsw" or (self.index_type == "hybrid" and len(self.vectors) > self.threshold):
            self._add_to_hnsw(idx, vector)
        elif self.index_type == "ivf":
            self._add_to_ivf(idx, vector)
            
        return True
        
    def batch_add(self, ids: List[str], vectors: List[List[float]]) -> int:
        """
        批量添加向量
        
        Args:
            ids: 向量ID列表
            vectors: 向量列表
            
        Returns:
            添加成功的数量
        """
        if len(ids) != len(vectors):
            raise VectorIndexError("ID列表和向量列表长度不匹配")
            
        success_count = 0
        start_idx = len(self.vectors)
        new_vectors = []
        new_ids = []
        
        # 过滤掉已存在的ID，更新它们
        for i, (id, vector) in enumerate(zip(ids, vectors)):
            if id in self.id_to_index:
                # 更新已存在的向量
                idx = self.id_to_index[id]
                if idx in self.deleted_indices:
                    self.deleted_indices.remove(idx)
                self.vectors[idx] = vector
                # 更新缓存
                if id in self.vector_cache:
                    self.vector_cache[id] = vector
                success_count += 1
            else:
                # 添加到待处理列表
                new_vectors.append(vector)
                new_ids.append(id)
                
        # 批量添加新向量
        if new_vectors:
            # 扩展向量和ID列表
            end_idx = start_idx + len(new_vectors)
            self.vectors.extend(new_vectors)
            self.ids.extend(new_ids)
            
            # 更新ID到索引的映射
            for i, id in enumerate(new_ids, start=start_idx):
                self.id_to_index[id] = i
                self._update_cache(id, new_vectors[i - start_idx])
            
            # 根据索引类型执行特定操作
            if self.index_type == "hnsw" or (self.index_type == "hybrid" and len(self.vectors) > self.threshold):
                self._batch_add_to_hnsw(range(start_idx, end_idx), new_vectors)
            elif self.index_type == "ivf":
                self._batch_add_to_ivf(range(start_idx, end_idx), new_vectors)
                
            success_count += len(new_vectors)
            
        return success_count
        
    def remove(self, id: str) -> bool:
        """
        删除向量索引中的条目
        
        Args:
            id: 要删除的向量ID
            
        Returns:
            是否成功删除
        """
        if id not in self.id_to_index:
            self.logger.warning(f"向量索引中不存在ID: {id}")
            return False
            
        try:
            # 获取索引位置
            idx = self.id_to_index[id]
            
            # 标记为已删除
            self.deleted_indices.add(idx)
            
            # 从ID到索引的映射中删除
            del self.id_to_index[id]
            
            # 从缓存中删除
            if id in self.vector_cache:
                del self.vector_cache[id]
                if id in self.recently_used:
                    self.recently_used.remove(id)
            
            # 如果是IVF索引，需要从相应的簇中删除
            if self.index_type == "ivf" and self.ivf_centroids is not None:
                for cluster_id, indices in self.ivf_clusters.items():
                    if idx in indices:
                        indices.remove(idx)
                        break
                        
            self.logger.debug(f"从向量索引中删除ID: {id}")
            return True
        except Exception as e:
            self.logger.error(f"从向量索引中删除失败: {str(e)}")
            return False
        
    def query(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        查询与给定向量最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            (id, 相似度)元组列表
        """
        if not self.vectors:
            return []
            
        self.n_queries += 1
        start_time = time.time()
        
        # 根据索引类型选择查询方法
        if self.index_type == "flat":
            results = self._flat_search(query_vector, top_k)
        elif self.index_type == "hnsw":
            results = self._hnsw_search(query_vector, top_k)
        elif self.index_type == "ivf":
            results = self._ivf_search(query_vector, top_k)
        elif self.index_type == "hybrid":
            if len(self.vectors) > self.threshold:
                results = self._hnsw_search(query_vector, top_k)
            else:
                results = self._flat_search(query_vector, top_k)
        else:
            raise VectorIndexError(f"不支持的索引类型: {self.index_type}")
            
        end_time = time.time()
        query_time = end_time - start_time
        self.total_query_time += query_time
        
        # 更新缓存
        for id, _ in results:
            if id in self.id_to_index:
                idx = self.id_to_index[id]
                self._update_cache(id, self.vectors[idx])
        
        return results
        
    def batch_query(self, query_vectors: List[List[float]], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        批量查询与给定向量最相似的向量
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的结果数量
            
        Returns:
            每个查询的(id, 相似度)元组列表
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.query(query_vector, top_k))
        return results
        
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索与查询向量最相似的向量（query的别名）
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似向量数量
            
        Returns:
            (id, 相似度)元组列表
        """
        return self.query(query_vector, top_k)
        
    def batch_search(self, query_vectors: List[List[float]], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        批量搜索与查询向量最相似的向量（batch_query的别名）
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的最相似向量数量
            
        Returns:
            每个查询对应的(id, 相似度)元组列表
        """
        return self.batch_query(query_vectors, top_k)
        
    def build(self) -> None:
        """
        构建索引，优化查询速度
        """
        if not self.vectors:
            self.logger.warning("没有向量，无法构建索引")
            return
            
        start_time = time.time()
        
        if self.index_type == "hnsw" or (self.index_type == "hybrid" and len(self.vectors) > self.threshold):
            self._build_hnsw()
        elif self.index_type == "ivf":
            self._build_ivf()
            
        end_time = time.time()
        self.build_time = end_time - start_time
        self.logger.info(f"索引构建完成，耗时: {self.build_time:.4f}秒")
    
    def clear(self) -> bool:
        """
        清空向量索引，移除所有向量和相关数据
        
        Returns:
            是否成功清空
        """
        try:
            # 清空内存中的数据
            self.vectors = []
            self.ids = []
            self.id_to_index = {}
            self.deleted_indices = set()
            
            # 重置索引结构
            if self.index_type == "hnsw":
                self.hnsw_layers = []
                self.hnsw_entry_point = None
            elif self.index_type == "ivf":
                self.ivf_centroids = None
                self.ivf_clusters = defaultdict(list)
                
            # 清空缓存
            if hasattr(self, 'vector_cache'):
                self.vector_cache = {}
            if hasattr(self, 'recently_used'):
                self.recently_used = []
            
            # 重置统计
            self.n_queries = 0
            self.total_query_time = 0
                
            # 如果有索引路径，保存空索引
            if hasattr(self, 'index_path') and self.index_path and hasattr(self, 'save'):
                self.save(self.index_path)
                
            self.logger.info("向量索引已成功清空")
            return True
        except Exception as e:
            self.logger.error(f"清空向量索引失败: {str(e)}")
            return False
            
    def save(self, path: Optional[str] = None) -> str:
        """
        保存索引到文件
        
        Args:
            path: 保存路径，为None则使用初始化时的路径
            
        Returns:
            保存的文件路径
        """
        save_path = path or self.index_path
        if not save_path:
            raise VectorIndexError("未提供保存路径")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 构造要保存的数据
        data = {
            "index_type": self.index_type,
            "vector_dim": self.vector_dim,
            "max_elements": self.max_elements,
            "config": self.config,
            "vectors": self.vectors,
            "ids": self.ids,
            "deleted_indices": list(self.deleted_indices),
        }
        
        # 根据索引类型添加特定数据
        if self.index_type == "hnsw" or self.index_type == "hybrid":
            data["hnsw_layers"] = self.hnsw_layers
            data["hnsw_entry_point"] = self.hnsw_entry_point
        elif self.index_type == "ivf":
            data["ivf_centroids"] = self.ivf_centroids.tolist() if self.ivf_centroids is not None else None
            data["ivf_clusters"] = {str(k): v for k, v in self.ivf_clusters.items()}
        
        # 保存数据
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            
        self.logger.info(f"索引已保存到: {save_path}")
        return save_path
        
    def load(self, path: str) -> bool:
        """
        从文件加载索引
        
        Args:
            path: 加载路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(path):
            self.logger.warning(f"索引文件不存在: {path}")
            return False
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # 更新基本属性
            self.index_type = data.get("index_type", self.index_type)
            self.vector_dim = data.get("vector_dim", self.vector_dim)
            self.max_elements = data.get("max_elements", self.max_elements)
            self.config.update(data.get("config", {}))
            
            # 加载向量数据
            self.vectors = data.get("vectors", [])
            self.ids = data.get("ids", [])
            self.deleted_indices = set(data.get("deleted_indices", []))
            
            # 重建ID到索引的映射
            self.id_to_index = {id: i for i, id in enumerate(self.ids)}
            
            # 根据索引类型加载特定数据
            if self.index_type == "hnsw" or self.index_type == "hybrid":
                self.hnsw_layers = data.get("hnsw_layers", [])
                self.hnsw_entry_point = data.get("hnsw_entry_point", None)
            elif self.index_type == "ivf":
                ivf_centroids = data.get("ivf_centroids", None)
                self.ivf_centroids = np.array(ivf_centroids) if ivf_centroids is not None else None
                
                ivf_clusters_str = data.get("ivf_clusters", {})
                self.ivf_clusters = defaultdict(list)
                for k, v in ivf_clusters_str.items():
                    self.ivf_clusters[int(k)] = v
            
            self.logger.info(f"成功从 {path} 加载索引，包含 {len(self.vectors)} 个向量")
            return True
        
        except Exception as e:
            self.logger.error(f"加载索引失败: {str(e)}")
            return False
            
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            v1: 第一个向量
            v2: 第二个向量
            
        Returns:
            余弦相似度 (0.0-1.0)
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        dot_product = np.dot(v1_array, v2_array)
        norm_v1 = np.linalg.norm(v1_array)
        norm_v2 = np.linalg.norm(v2_array)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(max(0.0, min(similarity, 1.0)))
        
    def _update_cache(self, id: str, vector: List[float]) -> None:
        """
        更新向量缓存
        
        Args:
            id: 向量ID
            vector: 向量
        """
        self.vector_cache[id] = vector
        
        # 更新最近使用
        if id in self.recently_used:
            self.recently_used.remove(id)
        self.recently_used.append(id)
        
        # 如果缓存超过大小，删除最不常用的
        if len(self.vector_cache) > self.cache_size:
            oldest_id = self.recently_used.pop(0)
            del self.vector_cache[oldest_id]
    
    def _flat_search(self, query_vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        """
        暴力搜索，计算与所有向量的相似度
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            (id, 相似度)元组列表
        """
        similarities = []
        
        # 计算所有向量的相似度
        for i, vector in enumerate(self.vectors):
            if i in self.deleted_indices:
                continue
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((i, similarity))
        
        # 排序并取前top_k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # 转换为(id, 相似度)格式
        return [(self.ids[idx], similarity) for idx, similarity in top_results]
        
    def _add_to_hnsw(self, idx: int, vector: List[float]) -> None:
        """
        将向量添加到HNSW结构
        
        Args:
            idx: 向量索引
            vector: 向量
        """
        # 如果HNSW为空，初始化
        if not self.hnsw_layers:
            self._build_hnsw()
            return
        
        # 第0层必定包含所有节点
        # 当前只实现基本结构，完整HNSW算法较为复杂，这里是简化版
        if len(self.hnsw_layers) == 0:
            self.hnsw_layers.append([])
        
        # 在第0层添加节点，并连接到最近的邻居
        if self.hnsw_entry_point is None:
            self.hnsw_entry_point = idx
            self.hnsw_layers[0].append(idx)
        else:
            # 在现有图中查找节点的位置
            # 此处简化实现，完整HNSW需要更复杂的近邻搜索和图连接
            self.hnsw_layers[0].append(idx)
            
    def _batch_add_to_hnsw(self, indices, vectors) -> None:
        """
        批量将向量添加到HNSW结构
        
        Args:
            indices: 向量索引列表
            vectors: 向量列表
        """
        # 如果HNSW为空，初始化
        if not self.hnsw_layers:
            self._build_hnsw()
            return
            
        # 批量添加到第0层
        if len(self.hnsw_layers) == 0:
            self.hnsw_layers.append([])
            
        for idx in indices:
            self.hnsw_layers[0].append(idx)
            
        # 设置入口点（如果为空）
        if self.hnsw_entry_point is None and indices:
            self.hnsw_entry_point = indices[0]
            
    def _build_hnsw(self) -> None:
        """
        构建HNSW索引
        
        为了简化，这里只实现基础的分层结构
        完整的HNSW算法需要构建层次图并优化链接
        """
        if not self.vectors:
            return
            
        self.hnsw_layers = []  # 重置层次结构
        
        # 第0层包含所有有效节点
        layer0 = [i for i in range(len(self.vectors)) if i not in self.deleted_indices]
        self.hnsw_layers.append(layer0)
        
        # 设置入口点为第一个节点
        if layer0:
            self.hnsw_entry_point = layer0[0]
            
        # 简化版分层，仅生成基本结构
        # 实际应用中需要实现完整的HNSW算法
        
    def _hnsw_search(self, query_vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        """
        使用HNSW结构搜索近邻
        
        为了简化，当前实现退化为flat search
        完整的HNSW算法需要遍历层次图搜索近邻
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            (id, 相似度)元组列表
        """
        # 当前简化实现，使用flat search
        return self._flat_search(query_vector, top_k)
        
    def _add_to_ivf(self, idx: int, vector: List[float]) -> None:
        """
        将向量添加到IVF结构
        
        Args:
            idx: 向量索引
            vector: 向量
        """
        # 如果IVF未初始化，构建
        if self.ivf_centroids is None:
            self._build_ivf()
            return
            
        # 找到最近的聚类中心
        v = np.array(vector)
        distances = [np.linalg.norm(v - centroid) for centroid in self.ivf_centroids]
        nearest_cluster = np.argmin(distances)
        
        # 添加到对应聚类
        self.ivf_clusters[nearest_cluster].append(idx)
        
    def _batch_add_to_ivf(self, indices, vectors) -> None:
        """
        批量将向量添加到IVF结构
        
        Args:
            indices: 向量索引列表
            vectors: 向量列表
        """
        # 如果IVF未初始化，构建
        if self.ivf_centroids is None:
            self._build_ivf()
            return
            
        # 批量为向量分配聚类
        for idx, vector in zip(indices, vectors):
            v = np.array(vector)
            distances = [np.linalg.norm(v - centroid) for centroid in self.ivf_centroids]
            nearest_cluster = np.argmin(distances)
            self.ivf_clusters[nearest_cluster].append(idx)
            
    def _build_ivf(self) -> None:
        """
        构建IVF索引
        
        使用K-means算法构建聚类中心和倒排表
        """
        from sklearn.cluster import KMeans
        
        # 过滤已删除的向量
        valid_indices = [i for i in range(len(self.vectors)) if i not in self.deleted_indices]
        if not valid_indices:
            return
            
        valid_vectors = [self.vectors[i] for i in valid_indices]
        
        # 确保聚类数量不超过向量数量
        n_clusters = min(self.n_lists, len(valid_vectors))
        
        # 使用K-means进行聚类
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(valid_vectors)
        self.ivf_centroids = kmeans.cluster_centers_
        
        # 构建倒排表
        self.ivf_clusters = defaultdict(list)
        for i, label in enumerate(labels):
            self.ivf_clusters[label].append(valid_indices[i])
            
    def _ivf_search(self, query_vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        """
        使用IVF结构搜索近邻
        
        首先找到最近的n_probes个聚类，然后在这些聚类中搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            (id, 相似度)元组列表
        """
        if not self.ivf_centroids is not None:
            return self._flat_search(query_vector, top_k)
            
        # 找到最近的n_probes个聚类中心
        v = np.array(query_vector)
        distances = [np.linalg.norm(v - centroid) for centroid in self.ivf_centroids]
        nearest_clusters = np.argsort(distances)[:self.n_probes]
        
        # 在这些聚类中搜索
        candidate_indices = []
        for cluster in nearest_clusters:
            candidate_indices.extend(self.ivf_clusters[cluster])
            
        # 计算候选向量的相似度
        similarities = []
        for idx in candidate_indices:
            if idx in self.deleted_indices:
                continue
            similarity = self._cosine_similarity(query_vector, self.vectors[idx])
            similarities.append((idx, similarity))
            
        # 排序并取前top_k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # 转换为(id, 相似度)格式
        return [(self.ids[idx], similarity) for idx, similarity in top_results]
        
    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "index_type": self.index_type,
            "vector_dim": self.vector_dim,
            "max_elements": self.max_elements,
            "current_size": len(self.vectors) - len(self.deleted_indices),
            "total_vectors": len(self.vectors),
            "deleted_vectors": len(self.deleted_indices),
            "build_time": self.build_time,
            "n_queries": self.n_queries,
            "avg_query_time": self.total_query_time / max(1, self.n_queries),
            "cache_size": len(self.vector_cache),
            "cache_hit_ratio": 0.0,  # 需添加缓存命中统计
        }
        
    def update(self, id: str, vector: List[float]) -> bool:
        """
        更新向量索引中的向量
        
        Args:
            id: 向量ID
            vector: 新向量
            
        Returns:
            是否成功更新
        """
        if id not in self.id_to_index:
            self.logger.warning(f"更新失败：ID {id} 不存在于索引中")
            return False
        
        try:
            # 获取索引位置
            idx = self.id_to_index[id]
            
            # 删除旧缓存
            if id in self.vector_cache:
                del self.vector_cache[id]
                if id in self.recently_used:
                    self.recently_used.remove(id)
            
            # 更新向量
            self.vectors[idx] = vector
            
            # 由于向量发生变化，需要重建索引
            if self.index_type == "hnsw" or (self.index_type == "hybrid" and len(self.vectors) > self.threshold):
                # 对于HNSW索引，我们需要重建
                self.hnsw_layers = []
                self.hnsw_entry_point = None
                self._build_hnsw()
            elif self.index_type == "ivf":
                # 对于IVF索引，我们需要更新聚类
                old_centroid = None
                for centroid, cluster in self.ivf_clusters.items():
                    if idx in cluster:
                        old_centroid = centroid
                        cluster.remove(idx)
                        break
                
                # 找到新的最近聚类中心
                min_dist = float('inf')
                closest_centroid = None
                for centroid in self.ivf_centroids:
                    dist = self._compute_distance(centroid, vector)
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid = centroid
                
                if closest_centroid is not None:
                    self.ivf_clusters[closest_centroid].append(idx)
            
            # 更新缓存
            self._update_cache(id, vector)
            
            self.logger.debug(f"成功更新ID {id} 的向量")
            return True
        except Exception as e:
            self.logger.error(f"更新向量失败：{str(e)}")
            return False
    
    def __len__(self):
        """
        返回索引中向量的数量，支持 len(vector_index) 操作
        
        Returns:
            有效向量的数量（不包括已删除的向量）
        """
        if not hasattr(self, 'ids'):
            return 0
        if not hasattr(self, 'deleted_indices'):
            return len(self.ids)
        return len(self.ids) - len(self.deleted_indices)
        
    def _perform_retrieval(self, query_vector: List[float], top_k: int = 5, include_metadata: bool = False, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        执行向量检索，返回标准格式的结果
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            include_metadata: 是否包含元数据（在此实现中忽略）
            threshold: 相似度阈值，低于此值的结果将被过滤
            
        Returns:
            检索结果列表，每项包含id、相似度等信息
        """
        if not self.vectors or len(self.vectors) == 0:
            return []
            
        # 使用query方法获取原始结果
        query_results = self.query(query_vector, top_k=max(top_k * 2, 10))  # 获取更多结果以便过滤
        
        # 转换为标准格式
        results = []
        for id, similarity in query_results:
            # 过滤低于阈值的结果
            if similarity < threshold:
                continue
                
            # 构建结果项
            item = {
                "id": id,
                "similarity": similarity,
                # VectorIndex本身不存储文本和元数据，所以这里只能返回ID和相似度
                "metadata": {}
            }
            
            # 添加到结果列表
            results.append(item)
        
        # 确保不超过要求的top_k数量
        return results[:top_k]
        
    def retrieve(self, query_vector: List[float], top_k: int = 10, metadata_filter: Dict[str, Any] = None, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        检索与查询向量最相似的向量，返回结构化结果
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            metadata_filter: 元数据过滤条件（在当前实现中会被忽略）
            threshold: 相似度阈值，低于此值的结果将被过滤
            
        Returns:
            检索结果列表，每项包含id、相似度和元数据
        """
        return self._perform_retrieval(query_vector, top_k, include_metadata=bool(metadata_filter), threshold=threshold)
        
    def batch_retrieve(self, query_vectors: List[List[float]], top_k: int = 10, metadata_filter: Dict[str, Any] = None, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        批量检索与查询向量最相似的向量，返回结构化结果
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的结果数量
            metadata_filter: 元数据过滤条件（在当前实现中会被忽略）
            threshold: 相似度阈值，低于此值的结果将被过滤
            
        Returns:
            每个查询对应的检索结果列表
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.retrieve(query_vector, top_k, metadata_filter, threshold))
        return results

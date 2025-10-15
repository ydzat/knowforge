'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-15 15:30:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-15 15:30:00
 * @Description: 高级知识记忆管理系统 - 实现多层次记忆结构、记忆动态管理和上下文感知检索
'''
import os
import time
import uuid
import math
import json
import datetime
import heapq
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from src.utils.logger import setup_logger
from src.utils.exceptions import NoteGenError
from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager, MemoryError

class AdvancedMemoryError(MemoryError):
    """高级记忆管理系统中的异常"""
    pass

class ShortTermMemory:
    """
    短期记忆模块，实现临时缓存，存储最近的交互内容
    """
    def __init__(self, capacity: int = 100):
        """
        初始化短期记忆
        
        Args:
            capacity: 最大容量（条目数）
        """
        self.capacity = capacity
        self.buffer = []  # 环形缓冲区
        self.logger = setup_logger()
        self.logger.info(f"初始化短期记忆，最大容量: {capacity}条")
        
    def add(self, item: Dict[str, Any]) -> None:
        """
        添加新项目到短期记忆
        
        Args:
            item: 包含内容和元数据的字典
        """
        if 'timestamp' not in item:
            item['timestamp'] = time.time()
            
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # 移除最早的项目
        self.buffer.append(item)
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的n条记忆
        
        Args:
            n: 要获取的条目数
            
        Returns:
            最近的n条记忆列表
        """
        return self.buffer[-min(n, len(self.buffer)):] if len(self.buffer) > 0 else []
    
    def clear(self) -> None:
        """清空短期记忆"""
        self.buffer = []
        self.logger.info("短期记忆已清空")
        
    def get_by_filter(self, filter_func) -> List[Dict[str, Any]]:
        """
        通过过滤函数获取满足条件的记忆
        
        Args:
            filter_func: 过滤函数，接收记忆项并返回布尔值
            
        Returns:
            满足条件的记忆列表
        """
        return [item for item in self.buffer if filter_func(item)]
    
    def __len__(self) -> int:
        """获取短期记忆中的条目数量"""
        return len(self.buffer)


class WorkingMemory:
    """
    工作记忆模块，实现优先级队列，存储活跃使用中的知识
    """
    def __init__(self, capacity: int = 500):
        """
        初始化工作记忆
        
        Args:
            capacity: 最大容量（条目数）
        """
        self.capacity = capacity
        self.priority_queue = []  # 优先级队列
        self.item_index = {}  # 快速索引
        self.last_accessed = {}  # 最后访问时间
        self.access_count = {}  # 访问计数
        self.logger = setup_logger()
        self.logger.info(f"初始化工作记忆，最大容量: {capacity}条")
        
    def add(self, item_id: str, item: Dict[str, Any], priority: float) -> None:
        """
        添加或更新工作记忆中的项目
        
        Args:
            item_id: 项目ID
            item: 包含内容和元数据的字典
            priority: 项目优先级
        """
        # 如果已存在，先移除旧项目
        self.remove(item_id)
        
        # 添加新项目到优先级队列
        heapq.heappush(self.priority_queue, (-priority, item_id))
        self.item_index[item_id] = item
        self.last_accessed[item_id] = time.time()
        self.access_count[item_id] = self.access_count.get(item_id, 0)
        
        # 如果超过容量，移除优先级最低的项目
        if len(self.priority_queue) > self.capacity:
            self._remove_lowest_priority_item()
            
    def remove(self, item_id: str) -> bool:
        """
        从工作记忆中移除指定项目
        
        Args:
            item_id: 要移除的项目ID
            
        Returns:
            是否成功移除
        """
        if item_id not in self.item_index:
            return False
            
        # 从索引中移除
        del self.item_index[item_id]
        if item_id in self.last_accessed:
            del self.last_accessed[item_id]
        if item_id in self.access_count:
            del self.access_count[item_id]
            
        # 从优先级队列中移除（惰性删除，实际上是在下次访问时过滤掉）
        # 完整清除队列中的元素代价较高，这里采用惰性删除策略
        return True
    
    def _remove_lowest_priority_item(self) -> Optional[str]:
        """
        移除优先级最低的项目
        
        考虑综合因素：基础优先级、访问频率和最近访问时间
        
        Returns:
            被移除的项目ID，如无项目则返回None
        """
        # 如果没有有效项目，返回None
        if not self.item_index:
            return None
            
        # 获取所有有效项目的ID
        valid_item_ids = list(self.item_index.keys())
        
        # 计算项目的综合分数
        item_scores = []
        current_time = time.time()
        
        for item_id in valid_item_ids:
            # 从优先级队列中找到对应项目的优先级
            base_priority = 0.5  # 默认中等优先级
            for p, i in self.priority_queue:
                if i == item_id:
                    base_priority = -p  # 转换回正常的优先级值
                    break
            
            # 计算访问频率因素 (0.0-0.4)
            access_count = self.access_count.get(item_id, 0)
            access_factor = min(0.4, 0.02 * access_count)
            
            # 计算时间衰减因素 (0.0-0.4)
            # 最近访问的条目获得较高的分数
            time_since_access = current_time - self.last_accessed.get(item_id, 0)
            # 使用指数衰减：12小时内访问过的条目保持较高分数
            time_factor = 0.4 * math.exp(-time_since_access / (12 * 3600))
            
            # 综合得分 (基础优先级 * 0.6 + 访问频率 * 0.2 + 时间衰减 * 0.2)
            # 得分越低，越可能被移除
            total_score = base_priority * 0.6 + access_factor + time_factor
            
            item_scores.append((total_score, item_id))
        
        # 按综合得分排序
        item_scores.sort()
        
        # 移除得分最低的项目
        if item_scores:
            lowest_score, lowest_id = item_scores[0]
            self.remove(lowest_id)
            
            # 重建优先级队列，移除不存在的项目
            new_queue = []
            for p, i in self.priority_queue:
                if i in self.item_index:
                    new_queue.append((p, i))
            
            self.priority_queue = new_queue
            heapq.heapify(self.priority_queue)
                
            return lowest_id
            
        return None
            
    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定ID的项目并更新访问记录
        
        Args:
            item_id: 项目ID
            
        Returns:
            项目内容，不存在则返回None
        """
        if item_id in self.item_index:
            self.last_accessed[item_id] = time.time()
            self.access_count[item_id] = self.access_count.get(item_id, 0) + 1
            return self.item_index[item_id]
        return None
    
    def get_top(self, n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """
        获取优先级最高的n个项目
        
        Args:
            n: 要获取的条目数
            
        Returns:
            (item_id, item)元组的列表，按优先级降序排序
        """
        # 创建有效项目的优先级队列副本
        valid_items = []
        for priority, item_id in self.priority_queue:
            if item_id in self.item_index:
                valid_items.append((priority, item_id))
                
        # 按优先级排序（注意priority是负值，所以这里是按照优先级的绝对值降序）
        valid_items.sort()
        
        # 返回前n个项目
        result = []
        for i, (priority, item_id) in enumerate(valid_items):
            if i >= n:
                break
            if item_id in self.item_index:  # 再次检查项目是否存在
                result.append((item_id, self.item_index[item_id]))
        
        return result
    
    def update_priority(self, item_id: str, new_priority: float) -> bool:
        """
        更新项目优先级
        
        Args:
            item_id: 项目ID
            new_priority: 新优先级值
            
        Returns:
            是否成功更新
        """
        if item_id not in self.item_index:
            return False
            
        # 获取当前项目
        item = self.item_index[item_id]
        
        # 移除旧项目
        self.remove(item_id)
        
        # 使用新优先级重新添加
        self.add(item_id, item, new_priority)
        return True
    
    def get_by_filter(self, filter_func, n: int = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        通过过滤函数获取满足条件的记忆
        
        Args:
            filter_func: 过滤函数，接收(item_id, item)并返回布尔值
            n: 最大返回数量，None表示不限制
            
        Returns:
            满足条件的(item_id, item)元组列表
        """
        filtered = [(item_id, item) 
                  for item_id, item in self.item_index.items() 
                  if filter_func(item_id, item)]
        
        if n is not None:
            return filtered[:n]
        return filtered
    
    def optimize_queue(self) -> None:
        """
        优化优先级队列，移除无效条目并重新平衡
        
        此方法应在队列中可能存在大量无效条目时调用，或者定期执行以维护队列效率
        """
        # 如果队列大小超过有效条目的2倍，则进行优化
        if len(self.priority_queue) > len(self.item_index) * 2:
            # 创建新队列，只包含有效条目
            new_queue = []
            for priority, item_id in self.priority_queue:
                if item_id in self.item_index:
                    new_queue.append((priority, item_id))
            
            # 替换原队列并重新建堆
            self.priority_queue = new_queue
            heapq.heapify(self.priority_queue)
            
            self.logger.debug(f"优化了工作记忆优先级队列，从 {len(self.priority_queue)} 条精简到 {len(self.item_index)} 条")
        
    def recompute_priorities(self) -> None:
        """
        根据访问频率和最近访问时间重新计算所有项目的优先级
        
        定期调用此方法可以确保优先级反映最新的使用模式
        """
        if not self.item_index:
            return
        
        current_time = time.time()
        # 创建项目ID到新优先级的映射
        new_priorities = {}
        
        for item_id in list(self.item_index.keys()):
            # 找到原始优先级
            base_priority = 0.5  # 默认优先级
            for p, i in self.priority_queue:
                if i == item_id:
                    base_priority = -p  # 转换回正常的优先级值
                    break
            
            # 计算访问频率因素
            access_count = self.access_count.get(item_id, 0)
            access_factor = min(0.3, 0.02 * access_count)
            
            # 计算时间衰减因素
            time_since_access = current_time - self.last_accessed.get(item_id, 0)
            time_factor = 0.2 * math.exp(-time_since_access / (12 * 3600))
            
            # 计算新优先级 (原始优先级部分 + 访问频率部分 + 时间部分)
            new_priority = base_priority * 0.5 + access_factor + time_factor
            new_priorities[item_id] = min(new_priority, 1.0)  # 确保优先级不超过1.0
        
        # 重建优先级队列
        self.priority_queue = []
        for item_id, priority in new_priorities.items():
            item = self.item_index[item_id]
            heapq.heappush(self.priority_queue, (-priority, item_id))
        
        self.logger.debug(f"重新计算了 {len(new_priorities)} 个工作记忆项目的优先级")
    
    def __len__(self) -> int:
        """获取工作记忆中的有效条目数量"""
        return len(self.item_index)


class AdvancedMemoryManager:
    """
    高级知识记忆管理系统，实现多层次记忆结构、记忆动态管理和上下文感知检索
    """
    # 默认清理策略
    CLEANUP_STRATEGIES = {
        "oldest": "删除最早添加的记忆",
        "least_used": "删除最少使用的记忆", 
        "relevance": "删除最不相关的记忆",
        "forgetting_curve": "基于遗忘曲线删除记忆"
    }
    
    # 默认检索策略
    RETRIEVAL_STRATEGIES = {
        "semantic": "语义相似度检索",
        "keyword": "关键词匹配检索",
        "temporal": "时间加权检索",
        "context": "上下文感知检索",
        "hybrid": "混合检索策略"
    }
    
    def __init__(
        self, 
        chroma_db_path: str,
        embedder: Optional[Embedder] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "knowforge_memory",
        config: Dict[str, Any] = None
    ):
        """
        初始化高级记忆管理系统
        
        Args:
            chroma_db_path: ChromaDB存储路径
            embedder: 向量化工具实例，如为None则内部创建
            embedding_model: 使用的嵌入模型名称（当embedder为None时使用）
            collection_name: 集合名称，默认为"knowforge_memory"
            config: 配置字典，包含记忆管理相关配置
        """
        self.logger = setup_logger()
        self.config = config or {}
        
        # 初始化长期记忆（基于现有的MemoryManager）
        self.long_term_memory = MemoryManager(
            chroma_db_path=chroma_db_path,
            embedder=embedder,
            embedding_model=embedding_model,
            collection_name=collection_name,
            config=self.config
        )
        
        # 初始化短期记忆和工作记忆
        stm_capacity = self.config.get("short_term_memory_capacity", 100)
        wm_capacity = self.config.get("working_memory_capacity", 500)
        self.short_term_memory = ShortTermMemory(capacity=stm_capacity)
        self.working_memory = WorkingMemory(capacity=wm_capacity)
        
        # 配置参数
        memory_config = self.config.get("advanced_memory", {})
        self.importance_threshold = memory_config.get("importance_threshold", 0.3)
        self.forgetting_base_factor = memory_config.get("forgetting_base_factor", 0.1)
        self.reinforcement_factor = memory_config.get("reinforcement_factor", 1.5)
        
        # 检索权重
        retrieval_weights = memory_config.get("retrieval_weights", {})
        self.semantic_weight = retrieval_weights.get("semantic", 0.4)
        self.keyword_weight = retrieval_weights.get("keyword", 0.2)
        self.temporal_weight = retrieval_weights.get("temporal", 0.15)
        self.relational_weight = retrieval_weights.get("relational", 0.15)
        self.preference_weight = retrieval_weights.get("preference", 0.1)
        
        # 高级功能配置
        self.use_forgetting_curve = memory_config.get("use_forgetting_curve", True)
        self.auto_reinforcement = memory_config.get("auto_reinforcement", True)
        
        self.logger.info("高级记忆管理系统初始化完成")
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        添加知识到记忆系统
        
        Args:
            content: 知识内容
            metadata: 知识元数据
            
        Returns:
            知识ID
        """
        if not content:
            self.logger.warning("尝试添加空内容知识")
            return None
            
        # 准备元数据
        if metadata is None:
            metadata = {}
        
        # 确保基本元数据字段存在
        if "source" not in metadata:
            metadata["source"] = "user_input"
        if "timestamp" not in metadata:
            metadata["timestamp"] = str(time.time())
        if "access_count" not in metadata:
            metadata["access_count"] = "0"
        if "importance" not in metadata:
            # 计算初始重要性分数
            metadata["importance"] = str(self._calculate_initial_importance(content, metadata))
        
        try:
            # 添加到长期记忆
            ids = self.long_term_memory.add_segments([content], [metadata])
            knowledge_id = ids[0] if ids else str(uuid.uuid4())
            
            # 计算优先级并添加到工作记忆
            priority = float(metadata.get("importance", 0.5))
            item = {"content": content, "metadata": metadata}
            self.working_memory.add(knowledge_id, item, priority)
            
            # 添加到短期记忆
            self.short_term_memory.add({"id": knowledge_id, "content": content, "metadata": metadata})
            
            self.logger.info(f"知识已添加到记忆系统，ID: {knowledge_id}")
            return knowledge_id
            
        except Exception as e:
            self.logger.error(f"添加知识失败: {str(e)}")
            raise AdvancedMemoryError(f"添加知识失败: {str(e)}")
    
    def retrieve(self, query: str, context: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关知识
        
        Args:
            query: 查询文本
            context: 上下文文本列表
            top_k: 返回结果数量
            
        Returns:
            相关知识列表，每项包含内容、相似度分数和元数据
        """
        if not query:
            self.logger.warning("检索查询为空")
            return []
            
        # 首先从工作记忆中检索（快速访问路径）
        working_results = self._retrieve_from_working_memory(query, top_k)
        
        # 然后从长期记忆中检索（更全面但较慢）
        long_term_results = self._retrieve_from_long_term_memory(query, context, top_k)
        
        # 合并结果，去重并保持最高相似度
        combined_results = {}
        for result in working_results + long_term_results:
            item_id = result.get("id")
            if item_id not in combined_results or result.get("similarity", 0) > combined_results[item_id].get("similarity", 0):
                combined_results[item_id] = result
                
        # 排序并截取前top_k个结果
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x.get("similarity", 0), 
            reverse=True
        )[:top_k]
        
        # 增加访问计数
        self._update_access_statistics(sorted_results)
        
        return sorted_results
    
    def _retrieve_from_working_memory(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        从工作记忆中检索知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 如果工作记忆为空，直接返回
        if len(self.working_memory) == 0:
            return []
            
        # 计算查询的向量表示
        query_vector = self.long_term_memory.embedder.embed_single(query)
        
        # 对工作记忆中的每个项计算相似度
        results = []
        for item_id, item in self.working_memory.item_index.items():
            content = item.get("content", "")
            # 获取或计算内容的向量表示
            content_vector = self.long_term_memory.embedder.embed_single(content)
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, content_vector)
            
            results.append({
                "id": item_id,
                "content": content,
                "metadata": item.get("metadata", {}),
                "similarity": float(similarity)
            })
        
        # 按相似度降序排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _retrieve_from_long_term_memory(self, query: str, context: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从长期记忆中检索知识
        
        Args:
            query: 查询文本
            context: 上下文文本列表
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 使用混合检索策略
        try:
            return self.long_term_memory.query_similar(
                query_text=query,
                top_k=top_k,
                context_texts=context,
                retrieval_mode="hybrid"
            )
        except Exception as e:
            self.logger.error(f"从长期记忆检索失败: {str(e)}")
            return []
    
    def _update_access_statistics(self, results: List[Dict[str, Any]]) -> None:
        """
        更新访问统计信息
        
        Args:
            results: 检索结果列表
        """
        for result in results:
            item_id = result.get("id")
            
            # 更新工作记忆中的访问统计
            item = self.working_memory.get(item_id)
            if item:
                # 访问工作记忆会自动更新访问计数和时间
                # 根据新的访问信息更新优先级
                metadata = item.get("metadata", {})
                importance = float(metadata.get("importance", 0.5))
                access_count = self.working_memory.access_count.get(item_id, 0)
                # 增加重要性以反映访问频率
                new_importance = importance * (1 + 0.01 * min(access_count, 100))
                metadata["importance"] = str(min(new_importance, 1.0))
                self.working_memory.update_priority(item_id, new_importance)
            
            # 更新长期记忆中的访问统计
            try:
                self.long_term_memory._update_metadata_on_access([item_id])
            except Exception as e:
                self.logger.warning(f"更新长期记忆访问统计失败: {str(e)}")
        
        # 每100次检索后，定期优化工作记忆优先级队列
        self._retrieval_count = getattr(self, '_retrieval_count', 0) + 1
        if self._retrieval_count % 100 == 0:
            self.working_memory.optimize_queue()
            
        # 每500次检索后，重新计算所有项目的优先级
        if self._retrieval_count % 500 == 0:
            self.working_memory.recompute_priorities()
    
    def _calculate_initial_importance(self, content: str, metadata: Dict[str, Any]) -> float:
        """
        计算知识的初始重要性分数
        
        Args:
            content: 知识内容
            metadata: 知识元数据
            
        Returns:
            重要性分数 (0.0-1.0)
        """
        # 基础重要性
        base_importance = 0.5
        
        # 内容因素 (30%)
        content_score = 0.5  # 默认中等重要性
        
        # 内容长度
        content_length = len(content)
        if content_length > 1000:
            content_score += 0.2  # 较长内容可能更重要
        elif content_length < 50:
            content_score -= 0.1  # 较短内容可能不太重要
        
        # 信息密度（简单估计：特殊字符、数字比例）
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        digits = sum(1 for c in content if c.isdigit())
        density_ratio = (special_chars + digits) / max(content_length, 1)
        if density_ratio > 0.2:
            content_score += 0.1  # 高信息密度
        
        # 使用因素 (40%) - 初始添加时为默认值
        usage_score = 0.5
        
        # 外部因素 (30%)
        external_score = 0.5  # 默认中等重要性
        
        # 来源可靠性
        source = metadata.get("source", "unknown")
        if source == "expert_input":
            external_score += 0.3
        elif source == "user_input":
            external_score += 0.1
        elif source == "generated":
            external_score -= 0.1
        
        # 明确的重要性标记
        if "explicit_importance" in metadata:
            explicit_value = float(metadata["explicit_importance"])
            external_score = explicit_value * 0.7 + external_score * 0.3
        
        # 计算总分
        importance = (0.3 * content_score + 0.4 * usage_score + 0.3 * external_score)
        
        # 确保在有效范围内
        return max(0.1, min(importance, 1.0))
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新知识条目
        
        Args:
            knowledge_id: 知识ID
            updates: 要更新的内容和元数据
            
        Returns:
            是否成功更新
        """
        try:
            # 检查是否存在于工作记忆
            item = self.working_memory.get(knowledge_id)
            if item:
                # 更新内容（如果提供）
                if "content" in updates:
                    item["content"] = updates["content"]
                
                # 更新元数据（如果提供）
                if "metadata" in updates:
                    item["metadata"].update(updates["metadata"])
                
                # 如果提供了新的重要性分数，更新优先级
                if "importance" in updates.get("metadata", {}):
                    new_importance = float(updates["metadata"]["importance"])
                    self.working_memory.update_priority(knowledge_id, new_importance)
            
            # 更新长期记忆
            # 目前MemoryManager没有直接的更新方法，暂时先删除后添加
            # 后续可以改进MemoryManager添加update_item方法
            old_data = self.long_term_memory.collection.get(ids=[knowledge_id])
            if old_data and old_data["ids"]:
                # 准备更新后的内容和元数据
                content = updates.get("content", old_data["documents"][0])
                
                old_metadata = old_data["metadatas"][0] if old_data["metadatas"] else {}
                metadata = old_metadata.copy()
                if "metadata" in updates:
                    metadata.update(updates["metadata"])
                
                # 删除旧条目
                self.long_term_memory.collection.delete(ids=[knowledge_id])
                
                # 添加更新后的条目
                self.long_term_memory.collection.add(
                    ids=[knowledge_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                
                self.logger.info(f"知识条目 {knowledge_id} 已更新")
                return True
            else:
                self.logger.warning(f"知识条目 {knowledge_id} 不存在于长期记忆中")
                return False
                
        except Exception as e:
            self.logger.error(f"更新知识条目 {knowledge_id} 失败: {str(e)}")
            return False
    
    def reinforce(self, knowledge_id: str, factor: float = 1.0) -> bool:
        """
        强化指定知识条目
        
        Args:
            knowledge_id: 知识ID
            factor: 强化因子
            
        Returns:
            是否成功强化
        """
        try:
            # 检查是否存在于工作记忆
            item = self.working_memory.get(knowledge_id)
            if item:
                # 获取当前重要性分数
                metadata = item.get("metadata", {})
                current_importance = float(metadata.get("importance", 0.5))
                
                # 计算新的重要性分数
                reinforcement = self.reinforcement_factor * factor
                new_importance = min(current_importance * reinforcement, 1.0)
                
                # 更新元数据
                metadata["importance"] = str(new_importance)
                metadata["last_reinforced"] = str(time.time())
                
                # 更新工作记忆的优先级
                self.working_memory.update_priority(knowledge_id, new_importance)
                
                # 更新长期记忆中的元数据
                return self.update_knowledge(knowledge_id, {"metadata": metadata})
            else:
                self.logger.warning(f"知识条目 {knowledge_id} 不存在于工作记忆中")
                return False
                
        except Exception as e:
            self.logger.error(f"强化知识条目 {knowledge_id} 失败: {str(e)}")
            return False
    
    def forget(self, knowledge_id: str = None, older_than_days: int = None) -> int:
        """
        应用遗忘机制，可选择特定知识或时间范围
        
        Args:
            knowledge_id: 特定知识ID，为None则应用于所有符合条件的知识
            older_than_days: 超过指定天数的知识，为None则使用遗忘曲线
            
        Returns:
            遗忘的条目数
        """
        forgotten_count = 0
        
        if knowledge_id:
            # 遗忘特定知识
            forgotten = self._forget_specific_knowledge(knowledge_id)
            return 1 if forgotten else 0
            
        elif older_than_days is not None:
            # 遗忘特定天数之前的知识
            cutoff_time = time.time() - older_than_days * 86400  # 86400秒 = 1天
            forgotten_count = self._forget_by_time(cutoff_time)
            
        else:
            # 使用遗忘曲线应用于所有知识
            forgotten_count = self._apply_forgetting_curve()
            
        return forgotten_count
    
    def _forget_specific_knowledge(self, knowledge_id: str) -> bool:
        """
        遗忘特定知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            是否成功遗忘
        """
        try:
            # 从工作记忆中移除
            removed_from_working = self.working_memory.remove(knowledge_id)
            
            # 从长期记忆中降低重要性（但不删除）
            item_data = self.long_term_memory.collection.get(ids=[knowledge_id])
            if item_data and item_data["ids"]:
                metadata = item_data["metadatas"][0] if item_data["metadatas"] else {}
                current_importance = float(metadata.get("importance", 0.5))
                
                # 大幅降低重要性
                new_importance = max(current_importance * 0.3, 0.1)
                metadata["importance"] = str(new_importance)
                metadata["forgotten"] = "true"
                metadata["forgotten_time"] = str(time.time())
                
                # 更新长期记忆
                return self.update_knowledge(knowledge_id, {"metadata": metadata})
            
            return removed_from_working
            
        except Exception as e:
            self.logger.error(f"遗忘知识条目 {knowledge_id} 失败: {str(e)}")
            return False
    
    def _forget_by_time(self, cutoff_time: float) -> int:
        """
        遗忘指定时间之前的知识
        
        Args:
            cutoff_time: 截止时间戳
            
        Returns:
            遗忘的条目数
        """
        forgotten_count = 0
        
        try:
            # 从工作记忆中遗忘旧知识
            old_knowledge = self.working_memory.get_by_filter(
                lambda item_id, item: float(item.get("metadata", {}).get("timestamp", time.time())) < cutoff_time
            )
            
            for item_id, _ in old_knowledge:
                if self._forget_specific_knowledge(item_id):
                    forgotten_count += 1
                    
        except Exception as e:
            self.logger.error(f"按时间遗忘知识失败: {str(e)}")
            
        return forgotten_count
    
    def _apply_forgetting_curve(self) -> int:
        """
        应用遗忘曲线机制
        
        Returns:
            遗忘的条目数
        """
        forgotten_count = 0
        current_time = time.time()
        
        try:
            # 获取工作记忆中的所有知识
            all_knowledge = list(self.working_memory.item_index.items())
            
            for item_id, item in all_knowledge:
                metadata = item.get("metadata", {})
                timestamp = float(metadata.get("timestamp", current_time))
                last_accessed = float(metadata.get("last_accessed", timestamp))
                importance = float(metadata.get("importance", 0.5))
                access_count = int(metadata.get("access_count", "0"))
                
                # 计算经过的时间（天）
                days_passed = (current_time - last_accessed) / 86400
                
                # 计算相对强度 (基于重要性和访问次数)
                relative_strength = importance * (1 + 0.2 * min(access_count, 10))
                
                # 计算保留率 R = e^(-t/S)
                retention_rate = math.exp(-days_passed * self.forgetting_base_factor / relative_strength)
                
                # 如果保留率低于阈值，应用遗忘
                if retention_rate < self.importance_threshold:
                    if self._forget_specific_knowledge(item_id):
                        forgotten_count += 1
                
        except Exception as e:
            self.logger.error(f"应用遗忘曲线失败: {str(e)}")
            
        return forgotten_count
    
    def retrieve_with_associations(self, query: str, depth: int = 1) -> Dict[str, Any]:
        """
        检索知识并返回关联网络
        
        Args:
            query: 查询文本
            depth: 关联深度
            
        Returns:
            包含检索结果和关联网络的字典
        """
        # 首先检索核心结果
        core_results = self.retrieve(query, top_k=5)
        if not core_results:
            return {"core": [], "associations": {}}
            
        # 建立关联网络
        associations = {}
        all_results = {result["id"]: result for result in core_results}
        
        # 为每个核心结果查找关联
        for level in range(depth):
            new_ids = []
            # 获取当前层级的所有结果ID
            current_level_ids = list(associations.keys()) if level > 0 else [r["id"] for r in core_results]
            
            for item_id in current_level_ids:
                # 查找与当前项关联的其他知识
                if item_id in all_results:
                    content = all_results[item_id].get("content", all_results[item_id].get("text", ""))
                    related = self.retrieve(content, top_k=3)
                    
                    # 过滤掉已经在网络中的结果
                    related = [r for r in related if r["id"] not in all_results]
                    
                    if related:
                        associations[item_id] = [r["id"] for r in related]
                        # 添加到所有结果中
                        for r in related:
                            all_results[r["id"]] = r
                            new_ids.append(r["id"])
            
            # 如果没有新的关联结果，提前结束
            if not new_ids:
                break
                
        return {
            "core": core_results,
            "associations": associations,
            "all_nodes": all_results
        }
    
    def create_knowledge_graph(self, central_topic: str) -> Dict[str, Any]:
        """
        创建以特定主题为中心的知识图谱
        
        Args:
            central_topic: 中心主题
            
        Returns:
            知识图谱数据
        """
        # 检索中心主题相关的知识
        central_results = self.retrieve(central_topic, top_k=5)
        
        # 获取完整的关联网络
        graph_data = self.retrieve_with_associations(central_topic, depth=2)
        
        # 添加中心节点
        graph_data["central_topic"] = central_topic
        
        return graph_data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆系统统计信息
        
        Returns:
            记忆系统统计信息字典
        """
        stats = {
            "short_term_memory": {
                "capacity": self.short_term_memory.capacity,
                "used": len(self.short_term_memory),
                "usage_percentage": len(self.short_term_memory) / max(1, self.short_term_memory.capacity) * 100
            },
            "working_memory": {
                "capacity": self.working_memory.capacity,
                "used": len(self.working_memory),
                "usage_percentage": len(self.working_memory) / max(1, self.working_memory.capacity) * 100
            },
            "long_term_memory": self.long_term_memory.get_collection_stats()
        }
        
        # 添加配置信息
        stats["config"] = {
            "importance_threshold": self.importance_threshold,
            "forgetting_base_factor": self.forgetting_base_factor,
            "reinforcement_factor": self.reinforcement_factor,
            "use_forgetting_curve": self.use_forgetting_curve,
            "auto_reinforcement": self.auto_reinforcement
        }
        
        return stats
        
    def export_memory(self, file_path: Optional[str] = None, format: str = "json") -> str:
        """
        导出记忆数据

        Args:
            file_path: 导出的文件路径，如果为None则自动生成
            format: 导出格式，目前支持"json"

        Returns:
            导出的数据或文件路径
        """
        if format.lower() == "json":
            # 确定导出路径
            if file_path is None:
                # 自动生成路径
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"workspace/memory_export_{timestamp}.json"
            else:
                export_path = file_path

            try:
                # 导出长期记忆
                ltm_path = self.long_term_memory.export_to_json(export_path)
                
                # 添加工作记忆和短期记忆信息
                with open(ltm_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 添加工作记忆信息
                data["working_memory"] = {
                    "capacity": self.working_memory.capacity,
                    "items": [{"id": item_id, **item} for item_id, item in self.working_memory.item_index.items()]
                }
                
                # 添加短期记忆信息
                data["short_term_memory"] = {
                    "capacity": self.short_term_memory.capacity,
                    "buffer": self.short_term_memory.buffer
                }
                
                # 保存增强的导出文件
                with open(ltm_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"记忆系统已导出到: {ltm_path}")
                return ltm_path
                
            except Exception as e:
                self.logger.error(f"导出记忆系统失败: {str(e)}")
                raise AdvancedMemoryError(f"导出记忆系统失败: {str(e)}")
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def import_memory(self, data: str, format: str = "json") -> int:
        """
        导入记忆数据
        
        Args:
            data: JSON字符串或文件路径
            format: 导入格式，目前支持"json"
            
        Returns:
            导入的条目数
        """
        if format.lower() == "json":
            try:
                # 判断是文件路径还是JSON字符串
                if os.path.isfile(data):
                    with open(data, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                else:
                    memory_data = json.loads(data)
                
                # 导入长期记忆
                imported_count = 0
                if "entries" in memory_data:
                    entries = memory_data["entries"]
                    for entry in entries:
                        # 添加到长期记忆
                        self.long_term_memory.collection.add(
                            ids=[entry["id"]],
                            documents=[entry["text"]],
                            metadatas=[entry["metadata"]]
                        )
                        
                        # 添加到工作记忆
                        importance = float(entry["metadata"].get("importance", 0.5))
                        item = {"content": entry["text"], "metadata": entry["metadata"]}
                        self.working_memory.add(entry["id"], item, importance)
                        
                        imported_count += 1
                
                # 导入工作记忆
                if "working_memory" in memory_data and "items" in memory_data["working_memory"]:
                    wm_items = memory_data["working_memory"]["items"]
                    for item_data in wm_items:
                        if "id" in item_data:
                            item_id = item_data["id"]
                            # 如果该项已经作为长期记忆条目导入，则跳过
                            if item_id not in self.working_memory.item_index:
                                content = item_data.get("content", "")
                                metadata = item_data.get("metadata", {})
                                importance = float(metadata.get("importance", 0.5))
                                self.working_memory.add(item_id, {"content": content, "metadata": metadata}, importance)
                                imported_count += 1
                
                self.logger.info(f"成功导入 {imported_count} 条记忆")
                return imported_count
                
            except Exception as e:
                self.logger.error(f"导入记忆失败: {str(e)}")
                raise AdvancedMemoryError(f"导入记忆失败: {str(e)}")
        else:
            raise ValueError(f"不支持的导入格式: {format}")
    
    def enhance_ocr_results(self, ocr_text: str, context: str = None) -> Dict[str, Any]:
        """
        使用记忆系统增强OCR结果
        
        Args:
            ocr_text: OCR识别的文本
            context: 上下文信息
            
        Returns:
            增强后的结果
        """
        if not ocr_text:
            return {"original": "", "enhanced": "", "confidence": 0.0}
        
        # 检索相关知识
        context_list = [context] if context else []
        relevant_knowledge = self.retrieve(ocr_text, context_list, top_k=3)
        
        # 如果没有找到相关知识，返回原始文本
        if not relevant_knowledge:
            return {
                "original": ocr_text,
                "enhanced": ocr_text,
                "confidence": 0.5,
                "references": []
            }
        
        # 构建参考知识列表
        references = []
        for item in relevant_knowledge:
            references.append({
                "id": item["id"],
                "content": item.get("content", item.get("text", "")),
                "similarity": item["similarity"]
            })
        
        # 使用LLM增强OCR结果
        try:
            from src.note_generator.llm_caller import LLMCaller
            
            # 初始化LLM调用器
            llm_caller = LLMCaller()
            
            # 构建提示以改进OCR结果
            prompt = self._build_ocr_correction_prompt(ocr_text, references)
            
            # 调用LLM进行增强
            enhanced_text = llm_caller.call_model(prompt)
            
            # 后处理LLM结果，去除可能的格式标记
            enhanced_text = self._postprocess_llm_result(enhanced_text)
            
            # 计算增强后的置信度（基于初始置信度和参考知识的相似度）
            base_confidence = 0.5
            similarity_boost = sum(item["similarity"] for item in relevant_knowledge) / (2 * len(relevant_knowledge))
            llm_boost = 0.2  # LLM处理提供额外20%的置信度提升
            final_confidence = min(base_confidence + similarity_boost + llm_boost, 0.98)
            
            return {
                "original": ocr_text,
                "enhanced": enhanced_text,
                "confidence": final_confidence,
                "references": references
            }
        except Exception as e:
            # 如果LLM处理失败，回退到原始结果
            logger.warning(f"LLM增强OCR结果失败: {str(e)}，使用原始文本")
            return {
                "original": ocr_text,
                "enhanced": ocr_text, 
                "confidence": 0.5 + sum(item["similarity"] for item in relevant_knowledge) / (2 * len(relevant_knowledge)),
                "references": references
            }
    
    def index_document_content(self, document_id: str, content_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        索引文档内容到记忆系统
        
        Args:
            document_id: 文档ID
            content_blocks: 内容块列表，每项包含内容和元数据
            
        Returns:
            添加的知识ID列表
        """
        added_ids = []
        
        for block in content_blocks:
            content = block.get("content", "")
            if not content:
                continue
                
            # 准备元数据
            metadata = block.get("metadata", {}).copy()
            metadata["source"] = f"document:{document_id}"
            metadata["content_type"] = block.get("type", "text")
            metadata["document_id"] = document_id
            
            # 添加到记忆系统
            knowledge_id = self.add_knowledge(content, metadata)
            if knowledge_id:
                added_ids.append(knowledge_id)
        
        return added_ids
    
    def retrieve_for_document(self, document_id: str, section_text: str) -> List[Dict[str, Any]]:
        """
        为文档特定部分检索相关知识
        
        Args:
            document_id: 文档ID
            section_text: 文档部分的文本
            
        Returns:
            相关知识列表
        """
        if not section_text:
            return []
            
        # 检索相关知识
        results = self.retrieve(section_text, top_k=5)
        
        # 过滤出来自其他文档的知识（避免循环引用）
        filtered_results = []
        for result in results:
            metadata = result.get("metadata", {})
            result_doc_id = metadata.get("document_id", "")
            if result_doc_id != document_id:
                filtered_results.append(result)
                
        return filtered_results
    
    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
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
    
    def _build_ocr_correction_prompt(self, ocr_text: str, references: List[Dict[str, Any]]) -> str:
        """
        构建用于OCR文本校正的LLM提示
        
        Args:
            ocr_text: OCR识别的原始文本
            references: 相关知识引用列表
            
        Returns:
            LLM提示文本
        """
        # 提取参考知识内容
        reference_texts = []
        for i, ref in enumerate(references):
            ref_text = ref.get("content", "")
            similarity = ref.get("similarity", 0)
            if ref_text:
                reference_texts.append(f"参考知识 {i+1} (相似度: {similarity:.2f}):\n{ref_text}")
        
        reference_context = "\n\n".join(reference_texts)
        
        # 创建提示
        prompt = f"""
你是一个专业的OCR文本校正专家，请使用知识库中的相关参考信息来校正OCR识别文本中的错误。

### OCR识别的文本（可能包含错误）：
{ocr_text}

### 知识库中的相关参考信息：
{reference_context}

### 任务：
1. 对比OCR文本与相关知识，修正OCR中的拼写错误、格式问题和语法错误
2. 使用参考知识中的专业术语和格式规范来改进OCR结果
3. 修复可能被错误识别的词语、数字和符号
4. 保持文本的原始含义和结构，不要添加没有依据的内容
5. 提高文本的准确性和可读性

### 直接返回纯文本结果，不要包含任何解释、注释或标记。
"""
        return prompt
        
    def _postprocess_llm_result(self, llm_text: str) -> str:
        """
        对LLM返回的结果进行后处理
        
        Args:
            llm_text: LLM返回的文本
            
        Returns:
            处理后的文本
        """
        # 去除可能的前后空白字符
        text = llm_text.strip()
        
        # 去除可能的代码块标记
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            
        # 去除可能的引号
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
            
        return text

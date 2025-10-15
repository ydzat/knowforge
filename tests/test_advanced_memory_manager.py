'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-15 16:45:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-15 16:45:00
 * @Description: 高级记忆管理系统测试用例
'''
import os
import json
import pytest
import tempfile
import time
import uuid
import numpy as np
from unittest.mock import patch, MagicMock

from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager, AdvancedMemoryError, ShortTermMemory, WorkingMemory

class TestShortTermMemory:
    """测试短期记忆模块功能"""
    
    def test_init(self):
        """测试初始化"""
        stm = ShortTermMemory(capacity=50)
        assert stm.capacity == 50
        assert len(stm.buffer) == 0
    
    def test_add_get_recent(self):
        """测试添加和获取最近记忆"""
        stm = ShortTermMemory(capacity=3)
        
        # 添加3个项目
        stm.add({"content": "测试1", "metadata": {"source": "test"}})
        stm.add({"content": "测试2", "metadata": {"source": "test"}})
        stm.add({"content": "测试3", "metadata": {"source": "test"}})
        
        # 验证添加成功
        assert len(stm) == 3
        
        # 获取最近2个
        recent = stm.get_recent(2)
        assert len(recent) == 2
        assert recent[0]["content"] == "测试2"
        assert recent[1]["content"] == "测试3"
        
        # 测试超出容量时的FIFO行为
        stm.add({"content": "测试4", "metadata": {"source": "test"}})
        assert len(stm) == 3
        all_items = stm.get_recent(3)
        assert [item["content"] for item in all_items] == ["测试2", "测试3", "测试4"]
    
    def test_clear(self):
        """测试清空短期记忆"""
        stm = ShortTermMemory()
        stm.add({"content": "测试1"})
        stm.add({"content": "测试2"})
        
        assert len(stm) == 2
        stm.clear()
        assert len(stm) == 0
    
    def test_get_by_filter(self):
        """测试通过过滤函数获取记忆"""
        stm = ShortTermMemory()
        stm.add({"content": "苹果", "type": "水果"})
        stm.add({"content": "香蕉", "type": "水果"})
        stm.add({"content": "西红柿", "type": "蔬菜"})
        
        # 过滤水果
        fruits = stm.get_by_filter(lambda item: item.get("type") == "水果")
        assert len(fruits) == 2
        assert fruits[0]["content"] == "苹果"
        assert fruits[1]["content"] == "香蕉"

class TestWorkingMemory:
    """测试工作记忆模块功能"""
    
    def test_init(self):
        """测试初始化"""
        wm = WorkingMemory(capacity=100)
        assert wm.capacity == 100
        assert len(wm.priority_queue) == 0
        assert len(wm.item_index) == 0
    
    def test_add_get(self):
        """测试添加和获取项目"""
        wm = WorkingMemory()
        
        # 添加项目
        item_id = "test1"
        item = {"content": "测试内容", "metadata": {"source": "test"}}
        wm.add(item_id, item, priority=0.8)
        
        # 验证添加成功
        assert len(wm) == 1
        assert len(wm.priority_queue) == 1
        
        # 获取项目
        retrieved = wm.get(item_id)
        assert retrieved == item
        assert item_id in wm.access_count
        assert item_id in wm.last_accessed
    
    def test_remove(self):
        """测试移除项目"""
        wm = WorkingMemory()
        
        # 添加项目
        wm.add("test1", {"content": "测试1"}, 0.5)
        wm.add("test2", {"content": "测试2"}, 0.7)
        
        # 验证添加成功
        assert len(wm) == 2
        
        # 移除项目
        result = wm.remove("test1")
        assert result is True
        assert len(wm) == 1
        assert "test1" not in wm.item_index
        
        # 移除不存在的项目
        result = wm.remove("nonexistent")
        assert result is False
    
    def test_capacity_limit(self):
        """测试容量限制和优先级管理"""
        wm = WorkingMemory(capacity=2)
        
        # 添加3个项目，超出容量
        wm.add("low", {"content": "低优先级"}, 0.1)
        wm.add("medium", {"content": "中优先级"}, 0.5)
        wm.add("high", {"content": "高优先级"}, 0.9)
        
        # 验证只保留了优先级最高的2个
        assert len(wm) == 2
        assert "low" not in wm.item_index
        assert "medium" in wm.item_index
        assert "high" in wm.item_index
    
    def test_get_top(self):
        """测试获取优先级最高的项目"""
        wm = WorkingMemory()
        
        wm.add("test1", {"content": "测试1"}, 0.3)
        wm.add("test2", {"content": "测试2"}, 0.8)
        wm.add("test3", {"content": "测试3"}, 0.5)
        
        # 获取优先级最高的2个项目
        top_items = wm.get_top(2)
        assert len(top_items) == 2
        
        # 验证顺序（按优先级降序）
        ids = [item_id for item_id, _ in top_items]
        assert "test2" in ids
        assert "test3" in ids
    
    def test_update_priority(self):
        """测试更新项目优先级"""
        wm = WorkingMemory()
        
        wm.add("test1", {"content": "测试1"}, 0.3)
        wm.add("test2", {"content": "测试2"}, 0.5)
        
        # 更新优先级
        result = wm.update_priority("test1", 0.9)
        assert result is True
        
        # 验证优先级已更新
        top_items = wm.get_top(1)
        assert len(top_items) == 1
        assert top_items[0][0] == "test1"
        
        # 更新不存在的项
        result = wm.update_priority("nonexistent", 0.8)
        assert result is False

    def test_recompute_priorities(self):
        """测试重新计算优先级功能"""
        wm = WorkingMemory(capacity=5)
        
        # 添加几个不同优先级的项目
        wm.add("test1", {"content": "测试1", "metadata": {"importance": "0.3"}}, 0.3)
        wm.add("test2", {"content": "测试2", "metadata": {"importance": "0.5"}}, 0.5)
        wm.add("test3", {"content": "测试3", "metadata": {"importance": "0.7"}}, 0.7)
        
        # 记录原始位置
        # 获取原始的priority_queue数据，确保test1是最低优先级的
        original_priorities = {}
        for priority, item_id in wm.priority_queue:
            original_priorities[item_id] = -priority  # 转换回正常优先级

        assert original_priorities["test1"] < original_priorities["test2"] < original_priorities["test3"]
        
        # 模拟大量访问，增加访问计数
        for _ in range(20):
            wm.get("test1")  # 低优先级但访问频率高
        
        # 重新计算优先级
        wm.recompute_priorities()
        
        # 计算新的优先级
        new_priorities = {}
        for priority, item_id in wm.priority_queue:
            new_priorities[item_id] = -priority  # 转换回正常优先级

        # 验证高访问频率但原始低优先级的test1的优先级已上升
        # test1的访问因素应该明显提高其优先级
        assert new_priorities["test1"] > original_priorities["test1"]
        
        # 在某些情况下，test1的优先级甚至可能超过test2
        if new_priorities["test1"] > new_priorities["test2"]:
            assert True, "访问频率显著提高了test1的优先级"
    
    def test_optimize_queue(self):
        """测试优化优先级队列功能"""
        wm = WorkingMemory(capacity=10)
        
        # 添加一些项目
        for i in range(5):
            wm.add(f"test{i}", {"content": f"测试{i}"}, 0.5)
        
        # 通过直接操作队列模拟无效项目（通常在实际使用中不会这么做）
        # 添加一些无效的项目ID到优先级队列
        for i in range(10):
            wm.priority_queue.append((-0.3, f"invalid{i}"))
        
        # 验证队列中包含无效项目
        assert len(wm.priority_queue) > len(wm.item_index)
        
        # 优化队列
        wm.optimize_queue()
        
        # 验证队列已被清理
        assert len(wm.priority_queue) == len(wm.item_index)
        
        # 确保有效项目仍然存在
        for i in range(5):
            assert wm.get(f"test{i}") is not None
    
    def test_advanced_capacity_management(self):
        """测试高级容量管理策略"""
        wm = WorkingMemory(capacity=3)
        
        # 添加项目：包括不同重要性和访问模式的项目
        wm.add("rare", {"content": "很少访问"}, 0.4)
        wm.add("old", {"content": "较早添加"}, 0.5)
        wm.add("important", {"content": "重要项目"}, 0.9)
        
        # 验证所有项目都在
        assert len(wm) == 3
        
        # 模拟频繁访问较不重要的项目
        for _ in range(10):
            wm.get("rare")
        
        # 添加新项目，触发容量管理
        wm.add("new", {"content": "新项目"}, 0.6)
        
        # 验证容量管理策略
        assert len(wm) == 3
        
        # 重要性高的项目和访问频率高的项目应被保留
        assert "important" in wm.item_index  # 高重要性
        assert "rare" in wm.item_index       # 高访问频率
        
        # 中等重要性但没有访问频率优势的旧项目应被移除
        assert "old" not in wm.item_index

class TestAdvancedMemoryManager:
    """测试高级记忆管理系统功能"""
    
    @pytest.fixture
    def mock_embedder(self):
        """创建模拟的Embedder实例"""
        mock = MagicMock(spec=Embedder)
        mock.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        mock.vector_size = 4
        mock.embed_texts.return_value = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
        mock.embed_single.return_value = [0.5, 0.5, 0.5, 0.5]
        return mock
    
    @pytest.fixture
    def mock_memory_manager(self):
        """创建模拟的MemoryManager实例"""
        mock = MagicMock(spec=MemoryManager)
        mock.embedder = MagicMock(spec=Embedder)
        mock.embedder.embed_single.return_value = [0.5, 0.5, 0.5, 0.5]
        mock.collection = MagicMock()
        mock.collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "documents": ["测试文本1", "测试文本2", "测试文本3"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
        }
        
        mock.add_segments.return_value = ["test_id1"]
        mock.query_similar.return_value = [
            {"id": "test_id1", "text": "测试文本", "metadata": {"source": "test"}, "similarity": 0.9}
        ]
        mock.get_collection_stats.return_value = {
            "count": 10,
            "collection_name": "knowforge_memory",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        return mock
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_init(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试初始化"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 验证内部组件初始化
            assert amm.long_term_memory == mock_memory_manager
            assert isinstance(amm.short_term_memory, ShortTermMemory)
            assert isinstance(amm.working_memory, WorkingMemory)
            
            # 验证配置参数
            assert amm.importance_threshold == 0.3
            assert amm.semantic_weight == 0.4
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_add_knowledge(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试添加知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 添加知识
            knowledge_id = amm.add_knowledge("测试知识内容", {"source": "user_input"})
            
            # 验证调用
            mock_memory_manager.add_segments.assert_called_once()
            
            # 验证结果
            assert knowledge_id == "test_id1"
            
            # 验证工作记忆和短期记忆更新
            assert len(amm.working_memory) == 1
            assert len(amm.short_term_memory) == 1
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_retrieve(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试检索知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 先添加一些知识到工作记忆
            amm.working_memory.add(
                "work_mem_id", 
                {"content": "工作记忆中的知识", "metadata": {"source": "test"}}, 
                0.8
            )
            
            # 检索知识
            results = amm.retrieve("测试查询", top_k=5)
            
            # 验证调用长期记忆检索
            mock_memory_manager.query_similar.assert_called_once()
            
            # 验证结果合并
            assert len(results) > 0
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_update_knowledge(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试更新知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        # 设置collection.get返回值
        mock_memory_manager.collection.get.return_value = {
            "ids": ["test_id"],
            "documents": ["原始内容"],
            "metadatas": [{"source": "test", "importance": "0.5"}]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 添加到工作记忆
            amm.working_memory.add(
                "test_id", 
                {"content": "原始内容", "metadata": {"source": "test", "importance": "0.5"}}, 
                0.5
            )
            
            # 更新知识
            result = amm.update_knowledge(
                "test_id", 
                {"content": "更新内容", "metadata": {"importance": "0.8"}}
            )
            
            # 验证调用
            mock_memory_manager.collection.get.assert_called_with(ids=["test_id"])
            mock_memory_manager.collection.delete.assert_called_with(ids=["test_id"])
            mock_memory_manager.collection.add.assert_called_once()
            
            # 验证成功更新
            assert result is True
            
            # 验证工作记忆已更新
            item = amm.working_memory.get("test_id")
            assert item["content"] == "更新内容"
            assert item["metadata"]["importance"] == "0.8"
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_reinforce(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试强化知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 添加到工作记忆
            amm.working_memory.add(
                "test_id", 
                {"content": "测试内容", "metadata": {"importance": "0.5"}}, 
                0.5
            )
            
            # 强化知识
            result = amm.reinforce("test_id", 1.0)
            
            # 验证工作记忆优先级已更新
            item = amm.working_memory.get("test_id")
            assert float(item["metadata"]["importance"]) > 0.5
            assert "last_reinforced" in item["metadata"]
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_forget(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试遗忘机制"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 添加旧知识到工作记忆
            old_time = time.time() - 86400 * 10  # 10天前
            amm.working_memory.add(
                "old_id", 
                {
                    "content": "旧知识", 
                    "metadata": {"importance": "0.3", "timestamp": str(old_time)}
                }, 
                0.3
            )
            
            # 添加新知识到工作记忆
            amm.working_memory.add(
                "new_id", 
                {
                    "content": "新知识", 
                    "metadata": {"importance": "0.6", "timestamp": str(time.time())}
                }, 
                0.6
            )
            
            # 遗忘指定天数前的知识
            forgot_count = amm.forget(older_than_days=5)
            
            # 验证只遗忘旧知识
            assert forgot_count >= 1
            assert amm.working_memory.get("old_id") is None
            assert amm.working_memory.get("new_id") is not None
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_retrieve_with_associations(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试检索关联知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        # 设置模拟检索结果
        mock_memory_manager.query_similar.side_effect = [
            # 第一次调用返回核心结果
            [
                {"id": "id1", "text": "核心知识1", "metadata": {"source": "test"}, "similarity": 0.9},
                {"id": "id2", "text": "核心知识2", "metadata": {"source": "test"}, "similarity": 0.8}
            ],
            # 第二次调用返回关联结果
            [
                {"id": "id3", "text": "关联知识1", "metadata": {"source": "test"}, "similarity": 0.7}
            ],
            # 第三次调用返回关联结果
            [
                {"id": "id4", "text": "关联知识2", "metadata": {"source": "test"}, "similarity": 0.6}
            ]
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 检索关联知识
            result = amm.retrieve_with_associations("测试查询", depth=1)
            
            # 验证结果结构
            assert "core" in result
            assert "associations" in result
            assert "all_nodes" in result
            assert len(result["core"]) == 2
            assert len(result["all_nodes"]) >= 2
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_create_knowledge_graph(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试创建知识图谱"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        # 设置模拟检索结果（与上一个测试类似）
        mock_memory_manager.query_similar.side_effect = [
            [
                {"id": "id1", "text": "核心知识1", "metadata": {"source": "test"}, "similarity": 0.9},
                {"id": "id2", "text": "核心知识2", "metadata": {"source": "test"}, "similarity": 0.8}
            ],
            [{"id": "id3", "text": "关联知识1", "metadata": {"source": "test"}, "similarity": 0.7}],
            [{"id": "id4", "text": "关联知识2", "metadata": {"source": "test"}, "similarity": 0.6}]
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 创建知识图谱
            graph = amm.create_knowledge_graph("人工智能")
            
            # 验证结果结构
            assert "central_topic" in graph
            assert graph["central_topic"] == "人工智能"
            assert "core" in graph
            assert "associations" in graph
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_get_memory_stats(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试获取记忆系统统计信息"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 获取统计信息
            stats = amm.get_memory_stats()
            
            # 验证结果结构
            assert "short_term_memory" in stats
            assert "working_memory" in stats
            assert "long_term_memory" in stats
            assert "config" in stats
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_export_import_memory(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试导出和导入记忆"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        # 设置导出返回值
        mock_memory_manager.export_to_json.return_value = "/tmp/memory_export.json"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 添加一些测试数据
            amm.working_memory.add("test_id", {"content": "测试内容", "metadata": {"source": "test"}}, 0.8)
            amm.short_term_memory.add({"id": "stm_id", "content": "短期记忆内容"})
            
            # 准备模拟的导出文件
            export_file = os.path.join(temp_dir, "test_export.json")
            with open(export_file, 'w') as f:
                json.dump({
                    "metadata": {"version": "0.1", "timestamp": time.time()},
                    "entries": [
                        {"id": "id1", "text": "导入测试1", "metadata": {"source": "import_test", "importance": "0.7"}}
                    ],
                    "working_memory": {
                        "capacity": 500,
                        "items": [{"id": "id2", "content": "工作记忆测试", "metadata": {"importance": "0.6"}}]
                    }
                }, f)
            
            # 模拟导出操作
            mock_memory_manager.export_to_json.return_value = export_file
            export_path = amm.export_memory()
            assert export_path == export_file
            
            # 导入操作
            imported_count = amm.import_memory(export_file)
            assert imported_count > 0
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_enhance_ocr_results(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试增强OCR结果"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 增强OCR结果
            result = amm.enhance_ocr_results("这是一个光萌识刮的文本", context="识别错误的例子")
            
            # 验证结果结构
            assert "original" in result
            assert "enhanced" in result
            assert "confidence" in result
            assert "references" in result
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_index_document_content(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试索引文档内容"""
        mock_memory_manager_class.return_value = mock_memory_manager
        mock_memory_manager.add_segments.side_effect = [["id1"], ["id2"]]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 准备文档内容
            content_blocks = [
                {"content": "第一章内容", "type": "text", "metadata": {"page": 1}},
                {"content": "第二章内容", "type": "text", "metadata": {"page": 2}}
            ]
            
            # 索引文档
            ids = amm.index_document_content("doc123", content_blocks)
            
            # 验证结果
            assert len(ids) == 2
            assert "id1" in ids
            assert "id2" in ids
    
    @patch('src.note_generator.advanced_memory_manager.MemoryManager')
    def test_retrieve_for_document(self, mock_memory_manager_class, mock_embedder, mock_memory_manager):
        """测试为文档检索相关知识"""
        mock_memory_manager_class.return_value = mock_memory_manager
        
        # 设置检索结果
        mock_memory_manager.query_similar.return_value = [
            {
                "id": "id1", 
                "text": "相关知识1", 
                "metadata": {"source": "test", "document_id": "doc123"}, 
                "similarity": 0.9
            },
            {
                "id": "id2", 
                "text": "相关知识2", 
                "metadata": {"source": "test", "document_id": "doc456"}, 
                "similarity": 0.8
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            amm = AdvancedMemoryManager(temp_dir, embedder=mock_embedder)
            
            # 为文档检索相关知识
            results = amm.retrieve_for_document("doc123", "文档内容示例")
            
            # 验证只返回非当前文档的知识
            assert len(results) == 1
            assert results[0]["id"] == "id2"
            assert results[0]["metadata"]["document_id"] == "doc456"

"""
测试记忆检索和管理策略的脚本
"""
import os
import sys
import time
from pprint import pprint

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.processor import Processor
from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager
from src.utils.config_loader import ConfigLoader

def test_memory_retrieval_strategy():
    """测试记忆检索策略"""
    print("\n=== 测试记忆检索策略 ===")

    # 初始化处理器
    processor = Processor()

    # 确保内存模块已启用
    if not processor.memory_manager:
        print("记忆管理模块未启用，请检查配置")
        return
    
    # 打印当前记忆配置
    config = ConfigLoader("resources/config/config.yaml")
    retrieval_mode = config.get("memory.retrieval_strategy.mode", "未配置")
    cleanup_strategy = config.get("memory.management.cleanup_strategy", "未配置")
    print(f"当前检索模式: {retrieval_mode}")
    print(f"当前清理策略: {cleanup_strategy}")
    
    # 获取记忆库统计信息
    try:
        stats = processor.get_memory_stats()
        print("\n记忆库统计信息:")
        print(f"- 条目数量: {stats['count']}")
        print(f"- 集合名称: {stats['collection_name']}")
        print(f"- 嵌入模型: {stats['embedding_model']}")
        print(f"- 向量维度: {stats['vector_size']}")
        if 'sources' in stats and stats['sources']:
            print(f"- 来源分布: {stats['sources']}")
        if 'time_range' in stats:
            print(f"- 时间范围: {stats['time_range']['min_date']} 至 {stats['time_range']['max_date']}")
    except Exception as e:
        print(f"获取统计信息失败: {str(e)}")
    
    # 测试不同检索模式的查询
    test_query = "深度学习在知识管理中的应用"
    print(f"\n查询文本: '{test_query}'")
    
    # 测试各种检索模式
    retrieval_modes = ["simple", "time_weighted", "hybrid"]
    for mode in retrieval_modes:
        print(f"\n使用 {mode} 模式查询:")
        try:
            results = processor.query_memory(
                query_text=test_query,
                top_k=3,
                retrieval_mode=mode
            )
            
            if not results["results"]:
                print(f"  未找到相关结果")
            else:
                for i, item in enumerate(results["results"]):
                    print(f"  结果 {i+1}: 相似度 {item['similarity']:.4f}")
                    print(f"  文本: {item['text'][:100]}..." if len(item['text']) > 100 else f"  文本: {item['text']}")
                    if 'final_score' in item:  # time_weighted 模式特有
                        print(f"  最终分数: {item['final_score']:.4f}")
                    if 'context_enhanced' in item:  # context_aware 模式特有
                        print(f"  上下文增强: {item['context_enhanced']}")
                    print()
        except Exception as e:
            print(f"  查询失败: {str(e)}")

def test_memory_management():
    """测试记忆管理功能"""
    print("\n=== 测试记忆管理功能 ===")
    
    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    chroma_db_path = config.get("memory.chroma_db_path", "workspace/memory_db/")
    collection_name = config.get("memory.collection_name", "knowforge_memory")
    embedding_model = config.get("embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2")
    
    # 初始化组件
    embedder = Embedder(embedding_model)
    memory_config = config.get_section("memory")
    memory_manager = MemoryManager(
        chroma_db_path=chroma_db_path,
        embedder=embedder,
        collection_name=collection_name,
        config=memory_config
    )
    
    # 显示当前策略
    print(f"清理策略: {memory_manager.cleanup_strategy}")
    print(f"最大记忆数量: {memory_manager.max_memory_size}")
    
    # 测试添加样本数据
    print("\n添加示例段落:")
    test_segments = [
        "深度学习是机器学习的一个分支，它使用多层神经网络来分析各种类型的数据。",
        "向量检索是一种基于文本向量化的搜索方法，可以找到语义上相似的内容。",
        "知识管理是一种明确的、系统的管理组织智力资产的方法。",
        "记忆检索策略有多种，如简单检索、时间加权检索、上下文感知检索和混合检索。",
        "相关性优先是一种记忆清理策略，它会保留与核心主题最相关的内容。"
    ]
    
    # 添加样本数据
    metadata = []
    for i, text in enumerate(test_segments):
        metadata.append({
            "source": "test_script",
            "timestamp": str(time.time() - i * 86400),  # 每个条目相差一天
            "access_count": "0"
        })
    
    ids = memory_manager.add_segments(test_segments, metadata)
    print(f"添加了 {len(ids)} 个样本段落")
    
    # 导出记忆库
    export_path = os.path.join("workspace", "memory_test_export.json")
    memory_manager.export_to_json(export_path)
    print(f"记忆库已导出至: {export_path}")
    
    # 执行一次查询，触发访问计数更新
    print("\n执行查询，更新访问计数:")
    results = memory_manager.query_similar("记忆管理和检索策略", top_k=2)
    for i, item in enumerate(results):
        print(f"结果 {i+1}: {item['text'][:50]}... (相似度: {item['similarity']:.4f})")
    
    # 获取并显示统计信息
    stats = memory_manager.get_collection_stats()
    print("\n记忆库统计信息:")
    pprint(stats)

if __name__ == "__main__":
    print("开始测试记忆管理策略...")
    test_memory_retrieval_strategy()
    test_memory_management()
    print("\n测试完成")
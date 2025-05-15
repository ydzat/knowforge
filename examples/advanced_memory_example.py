'''
 * @Author: @ydzat, GitHub Copilot
 * @Date: 2025-05-16 11:00:00
 * @LastEditors: @ydzat, GitHub Copilot
 * @LastEditTime: 2025-05-16 11:00:00
 * @Description: 高级记忆管理系统使用示例
'''
import os
import sys
import time
import json
import random
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import ConfigLoader
from src.note_generator.embedder import Embedder
from src.note_generator.advanced_memory_manager import AdvancedMemoryManager

def print_divider(title: str = None):
    """打印分隔线"""
    width = 60
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def example_1_basic_usage():
    """基本使用示例"""
    print_divider("基本使用示例")
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    
    # 初始化嵌入模型
    embedder = Embedder()
    
    # 初始化高级记忆管理系统
    memory_manager = AdvancedMemoryManager(
        chroma_db_path="workspace/memory_db/",
        embedder=embedder,
        config=config_loader._config
    )
    
    print(f"高级记忆管理系统已初始化, 使用模型: {embedder.model_name}")
    
    # 添加知识
    print("\n添加知识...")
    
    knowledge_ids = []
    
    # 添加第一条知识
    id1 = memory_manager.add_knowledge(
        "人工智能(Artificial Intelligence)是研究如何使计算机具有人类智能的一门学科。",
        {"source": "user_input", "topic": "AI", "importance": "0.8"}
    )
    knowledge_ids.append(id1)
    print(f"已添加知识1, ID: {id1}")
    
    # 添加第二条知识
    id2 = memory_manager.add_knowledge(
        "机器学习是人工智能的一个分支，它专注于开发能够从数据中学习的算法和技术。",
        {"source": "user_input", "topic": "Machine Learning", "importance": "0.7"}
    )
    knowledge_ids.append(id2)
    print(f"已添加知识2, ID: {id2}")
    
    # 添加第三条知识
    id3 = memory_manager.add_knowledge(
        "深度学习是机器学习的一部分，它使用神经网络模型来处理和学习复杂的模式。",
        {"source": "user_input", "topic": "Deep Learning", "importance": "0.75"}
    )
    knowledge_ids.append(id3)
    print(f"已添加知识3, ID: {id3}")
    
    # 检索知识
    print("\n检索知识...")
    results = memory_manager.retrieve("什么是机器学习？", top_k=2)
    
    print(f"找到 {len(results)} 条相关知识:")
    for idx, result in enumerate(results):
        print(f"  [{idx+1}] 相似度: {result['similarity']:.4f}")
        print(f"      内容: {result['content'][:60]}..." if len(result['content']) > 60 else f"      内容: {result['content']}")
    
    # 获取记忆系统统计信息
    print("\n记忆系统统计信息:")
    stats = memory_manager.get_memory_stats()
    
    print(f"  短期记忆: {stats['short_term_memory']['used']}/{stats['short_term_memory']['capacity']} 条 "
          f"({stats['short_term_memory']['usage_percentage']:.1f}% 已用)")
    
    print(f"  工作记忆: {stats['working_memory']['used']}/{stats['working_memory']['capacity']} 条 "
          f"({stats['working_memory']['usage_percentage']:.1f}% 已用)")
    
    print(f"  长期记忆: {stats['long_term_memory']['count']} 条")
    
    return knowledge_ids

def example_2_memory_reinforcement(knowledge_ids: List[str]):
    """记忆强化示例"""
    print_divider("记忆强化示例")
    
    # 初始化系统
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    embedder = Embedder()
    memory_manager = AdvancedMemoryManager(
        chroma_db_path="workspace/memory_db/",
        embedder=embedder,
        config=config_loader._config
    )
    
    if not knowledge_ids or len(knowledge_ids) < 3:
        print("没有足够的知识条目进行示例")
        return
    
    # 模拟多次访问第一条知识
    print("\n模拟多次访问知识...")
    for _ in range(5):
        memory_manager.retrieve("人工智能是什么？", top_k=1)
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" 完成")
    
    # 显式强化第二条知识
    print(f"\n显式强化知识 (ID: {knowledge_ids[1]})")
    memory_manager.reinforce(knowledge_ids[1], factor=2.0)
    
    # 检查三条知识的优先级
    print("\n检查知识优先级:")
    for i, knowledge_id in enumerate(knowledge_ids):
        item = memory_manager.working_memory.get(knowledge_id)
        if item:
            importance = float(item["metadata"].get("importance", "0.0"))
            access_count = memory_manager.working_memory.access_count.get(knowledge_id, 0)
            print(f"  知识 {i+1} (ID: {knowledge_id[:6]}...) - 重要性: {importance:.4f}, 访问次数: {access_count}")

def example_3_memory_forgetting():
    """记忆遗忘示例"""
    print_divider("记忆遗忘示例")
    
    # 初始化系统
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    embedder = Embedder()
    memory_manager = AdvancedMemoryManager(
        chroma_db_path="workspace/memory_db/",
        embedder=embedder,
        config=config_loader._config
    )
    
    # 添加一些知识条目，模拟不同时间添加
    print("\n添加测试知识条目...")
    
    # 模拟老知识（10天前）
    old_time = time.time() - 86400 * 10
    old_id = memory_manager.add_knowledge(
        "这是一条将被遗忘的旧知识。",
        {"source": "test", "timestamp": str(old_time), "importance": "0.2"}
    )
    print(f"已添加旧知识 (ID: {old_id})")
    
    # 添加新知识（现在）
    new_id = memory_manager.add_knowledge(
        "这是一条重要的新知识。",
        {"source": "test", "importance": "0.8"}
    )
    print(f"已添加新知识 (ID: {new_id})")
    
    # 应用遗忘机制
    print("\n应用遗忘机制...")
    forgotten_count = memory_manager.forget(older_than_days=5)
    print(f"已遗忘 {forgotten_count} 条知识")
    
    # 检查知识状态
    print("\n检查知识状态:")
    old_item = memory_manager.working_memory.get(old_id)
    new_item = memory_manager.working_memory.get(new_id)
    
    print(f"  旧知识 (ID: {old_id}) - {'存在于工作记忆' if old_item else '已从工作记忆中移除'}")
    print(f"  新知识 (ID: {new_id}) - {'存在于工作记忆' if new_item else '已从工作记忆中移除'}")

def example_4_association_network():
    """关联网络示例"""
    print_divider("关联网络示例")
    
    # 初始化系统
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    embedder = Embedder()
    memory_manager = AdvancedMemoryManager(
        chroma_db_path="workspace/memory_db/",
        embedder=embedder,
        config=config_loader._config
    )
    
    # 添加一组相关知识
    print("\n添加一组相关知识...")
    
    topics = [
        ("Python", "Python是一种高级编程语言，以简洁的语法和强大的库生态系统闻名。"),
        ("JavaScript", "JavaScript是一种前端编程语言，主要用于网页开发和交互功能。"),
        ("TypeScript", "TypeScript是JavaScript的超集，添加了静态类型定义，提高了代码的可维护性。"),
        ("React", "React是一个JavaScript库，用于构建用户界面，尤其是单页应用程序。"),
        ("Vue", "Vue是一个渐进式JavaScript框架，用于构建用户界面，易于集成到项目中。"),
        ("Node.js", "Node.js是一个基于V8引擎的JavaScript运行环境，使JavaScript可以在服务器端运行。"),
        ("Flask", "Flask是一个使用Python编写的轻量级Web应用框架，易于上手和扩展。"),
        ("Django", "Django是一个高级Python Web框架，鼓励快速开发和干净的设计。")
    ]
    
    for topic, description in topics:
        knowledge_id = memory_manager.add_knowledge(
            description,
            {"source": "example", "topic": topic, "category": "programming"}
        )
        print(f"已添加知识 '{topic}' (ID: {knowledge_id})")
    
    # 创建知识图谱
    print("\n创建知识图谱...")
    graph = memory_manager.create_knowledge_graph("Python")
    
    # 打印关联信息
    print("\n知识关联网络:")
    print(f"  中心主题: {graph['central_topic']}")
    print(f"  核心知识点: {len(graph['core'])}")
    print(f"  关联关系: {len(graph['associations'])}")
    
    # 打印核心知识点
    print("\n核心知识点:")
    for idx, item in enumerate(graph["core"]):
        print(f"  [{idx+1}] {item.get('metadata', {}).get('topic', 'Unknown')}: {item.get('content', '')[:60]}..." if len(item.get('content', '')) > 60 else f"  [{idx+1}] {item.get('metadata', {}).get('topic', 'Unknown')}: {item.get('content', '')}")

def example_5_ocr_enhancement():
    """OCR增强示例"""
    print_divider("OCR增强示例")
    
    # 初始化系统
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "resources", "config", "config.yaml")
    config_loader = ConfigLoader(config_path)
    embedder = Embedder()
    memory_manager = AdvancedMemoryManager(
        chroma_db_path="workspace/memory_db/",
        embedder=embedder,
        config=config_loader._config
    )
    
    # 添加一些领域知识
    print("\n添加领域知识到记忆系统...")
    
    memory_manager.add_knowledge(
        "神经网络是一种模拟人脑结构和功能的计算模型，由多层神经元组成，用于解决复杂的模式识别问题。",
        {"source": "domain_knowledge", "topic": "Neural Networks"}
    )
    
    memory_manager.add_knowledge(
        "卷积神经网络(CNN)是一种专门处理网格状数据(如图像)的神经网络，包含卷积层、池化层和全连接层。",
        {"source": "domain_knowledge", "topic": "CNN"}
    )
    
    memory_manager.add_knowledge(
        "循环神经网络(RNN)是处理序列数据的神经网络，通过内部状态可以记忆之前的信息，适合处理时间序列数据。",
        {"source": "domain_knowledge", "topic": "RNN"}
    )
    
    # 模拟OCR结果
    ocr_text = "神经网洛是一种模拟人脇结构和功能的计算模式，由多次神经员组成，用于解块的模识别问题。"
    print(f"\nOCR原始文本: \"{ocr_text}\"")
    
    # 使用记忆系统增强OCR结果
    enhanced_result = memory_manager.enhance_ocr_results(ocr_text)
    
    # 打印结果
    print("\n增强结果:")
    print(f"  置信度: {enhanced_result['confidence']:.4f}")
    print(f"  参考知识: {len(enhanced_result['references'])} 项")
    
    if enhanced_result['references']:
        print("\n参考知识:")
        for idx, ref in enumerate(enhanced_result['references']):
            print(f"  [{idx+1}] 相似度: {ref['similarity']:.4f}")
            print(f"      {ref['content'][:60]}..." if len(ref['content']) > 60 else f"      {ref['content']}")

def run_all_examples():
    """运行所有示例"""
    # 运行基本示例
    knowledge_ids = example_1_basic_usage()
    
    # 运行记忆强化示例
    example_2_memory_reinforcement(knowledge_ids)
    
    # 运行记忆遗忘示例
    example_3_memory_forgetting()
    
    # 运行关联网络示例
    example_4_association_network()
    
    # 运行OCR增强示例
    example_5_ocr_enhancement()
    
    print_divider("所有示例完成")
    print("高级记忆管理系统演示完成!\n")

if __name__ == "__main__":
    run_all_examples()

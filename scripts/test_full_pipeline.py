'''
 * @Author: @ydzat
 * @Date: 2025-05-14 16:45:00
 * @LastEditors: @ydzat
 * @LastEditTime: 2025-05-14 16:45:00
 * @Description: 完整流程测试，包含embedding和memory管理功能
'''
import os
import sys
import time
import logging
import argparse
from pprint import pprint

# 添加src到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.note_generator.processor import Processor
from src.note_generator.embedder import Embedder
from src.note_generator.memory_manager import MemoryManager
from src.note_generator.input_handler import InputHandler
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger, get_logger

# 配置日志
setup_logger()
logger = get_logger('KnowForge')

def test_embedding_functionality():
    """测试向量化功能"""
    print("\n=== 测试向量化功能 ===")

    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    embedding_config = config.get_section("embedding")
    model_name = config.get("embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2")

    # 初始化向量化组件
    try:
        print(f"正在加载嵌入模型: {model_name}...")
        embedder = Embedder(model_name=model_name)
        print(f"模型加载成功，向量维度: {embedder.vector_size}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

    # 测试向量化多段文本
    test_texts = [
        "KnowForge是一款智能笔记生成器",
        "向量化技术可以将文本转换为数值表示",
        "记忆管理模块可以存储和检索相关信息"
    ]

    print("\n向量化多段文本:")
    try:
        start_time = time.time()
        embeddings = embedder.embed_texts(test_texts)
        duration = time.time() - start_time
        print(f"向量化完成，耗时: {duration:.4f}秒")
        print(f"生成了 {len(embeddings)} 个向量，每个向量维度: {len(embeddings[0])}")
        
        # 计算文本相似度示例
        similarity = Embedder.cosine_similarity(embeddings[0], embeddings[1])
        print(f"文本1和文本2的余弦相似度: {similarity:.4f}")
        
        similarity = Embedder.cosine_similarity(embeddings[0], embeddings[2])
        print(f"文本1和文本3的余弦相似度: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"向量化失败: {str(e)}")
        return False

def test_memory_store_retrieve():
    """测试记忆存储和检索"""
    print("\n=== 测试记忆存储和检索 ===")

    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    chroma_db_path = config.get("memory.chroma_db_path", "workspace/memory_db/")
    collection_name = config.get("memory.collection_name", "test_collection")
    model_name = config.get("embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2")

    # 初始化组件
    try:
        embedder = Embedder(model_name=model_name)
        memory_manager = MemoryManager(
            chroma_db_path=chroma_db_path,
            embedder=embedder,
            collection_name=collection_name
        )
        print("记忆管理器初始化成功")
    except Exception as e:
        print(f"记忆管理器初始化失败: {str(e)}")
        return False

    # 测试存储和检索
    test_segments = [
        "人工智能是计算机科学的一个重要分支，致力于开发能模拟人类智能的系统。",
        "机器学习是人工智能的核心技术之一，使计算机能够从数据中学习。",
        "深度学习是一种基于神经网络的机器学习方法，通过多层次的学习提取特征。",
        "自然语言处理技术让计算机能够理解和生成人类语言。",
        "计算机视觉专注于让计算机能够从图像或视频中获取高级理解。"
    ]

    print(f"\n添加 {len(test_segments)} 个测试片段到记忆库...")
    try:
        # 添加内容到记忆库
        ids = memory_manager.add_segments(test_segments)
        print(f"成功添加 {len(ids)} 个片段")

        # 获取库统计信息
        stats = memory_manager.get_collection_stats()
        print(f"记忆库中共有 {stats['count']} 条记录")

        # 测试检索
        print("\n测试检索功能:")
        query = "人工智能和机器学习的关系"
        print(f"查询: '{query}'")
        
        # 测试不同的检索模式
        for mode in ["simple", "hybrid"]:
            print(f"\n使用 {mode} 检索模式:")
            try:
                if mode == "simple":
                    results = memory_manager.query_similar(query, top_k=3)
                else:
                    results = memory_manager._hybrid_retrieval(query, top_k=3)
                
                for i, item in enumerate(results):
                    print(f"结果 {i+1} (相似度: {item['similarity']:.4f}):")
                    print(f"  {item['text']}")
            except Exception as e:
                print(f"  {mode}模式检索失败: {str(e)}")

        return True
    except Exception as e:
        print(f"记忆存储或检索测试失败: {str(e)}")
        return False

def test_document_processing():
    """测试实际文档处理流程"""
    print("\n=== 测试实际文档处理流程 ===")

    # 检查是否有测试文档
    input_dir = "input"
    workspace_dir = "workspace"
    
    # 确保工作目录存在
    os.makedirs(workspace_dir, exist_ok=True)
    
    # 加载配置
    config = ConfigLoader("resources/config/config.yaml")
    
    # 查找可用的测试文档
    test_docs = []
    for subdir, category in [
        (os.path.join(input_dir, "pdf"), "PDF"),
        (os.path.join(input_dir, "codes"), "代码文件")
    ]:
        if os.path.exists(subdir):
            files = [f for f in os.listdir(subdir) if not f.startswith('.')]
            if files:
                test_docs.append((os.path.join(subdir, files[0]), category))
    
    if not test_docs:
        print("未找到测试文档，请确保input目录中有PDF或代码文件")
        return False
    
    # 初始化处理器
    try:
        processor = Processor()
        print("初始化处理器成功")
    except Exception as e:
        print(f"初始化处理器失败: {str(e)}")
        return False
    
    # 依次处理测试文档
    for test_path, doc_type in test_docs:
        print(f"\n处理 {doc_type} 文档: {os.path.basename(test_path)}")
        
        try:
            # 提取文本 - 使用process_file方法处理单个文件
            print("1. 提取文本...")
            input_handler = InputHandler(input_dir, workspace_dir, config)
            text = input_handler.process_file(test_path)
            if not text:
                print("  未能提取到文本")
                continue
                
            texts = [text]
            print(f"  成功提取文本，长度: {len(text)} 字符")
            print(f"  文本示例: {text[:100]}..." if len(text) > 100 else f"  文本示例: {text}")
            
            # 拆分文本
            print("2. 拆分文本...")
            segments = processor.splitter.split_text(texts)
            print(f"  文本被拆分为 {len(segments)} 个片段")
            
            # 向量化和记忆存储
            if processor.memory_manager:
                print("3. 向量化并存储到记忆库...")
                ids = processor.memory_manager.add_segments(segments)
                print(f"  成功添加 {len(ids)} 个片段到记忆库")
                
                # 测试检索
                print("4. 从记忆库检索相关内容...")
                # 使用文档名作为查询
                query = os.path.basename(test_path).split('.')[0].replace('_', ' ')
                results = processor.memory_manager.query_similar(query, top_k=2)
                print(f"  查询 '{query}' 返回 {len(results)} 个结果")
                if results:
                    print(f"  最相关片段 (相似度: {results[0]['similarity']:.4f}):")
                    print(f"  {results[0]['text'][:150]}..." if len(results[0]['text']) > 150 else f"  {results[0]['text']}")
            else:
                print("记忆管理器未启用，跳过向量化和检索测试")
            
            print(f"{doc_type}文档处理测试完成")
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False
    
    return True

def main():
    """主函数，运行各项测试"""
    parser = argparse.ArgumentParser(description='KnowForge全流程测试工具')
    parser.add_argument('--embedding-only', action='store_true', help='仅测试向量化功能')
    parser.add_argument('--memory-only', action='store_true', help='仅测试记忆管理功能')
    parser.add_argument('--document-only', action='store_true', help='仅测试文档处理流程')
    args = parser.parse_args()
    
    print("===== KnowForge 全流程测试开始 =====")
    start_time = time.time()
    
    success_count = 0
    total_tests = 0
    
    # 根据参数决定运行哪些测试
    run_all = not (args.embedding_only or args.memory_only or args.document_only)
    
    if run_all or args.embedding_only:
        total_tests += 1
        if test_embedding_functionality():
            success_count += 1
    
    if run_all or args.memory_only:
        total_tests += 1
        if test_memory_store_retrieve():
            success_count += 1
    
    if run_all or args.document_only:
        total_tests += 1
        if test_document_processing():
            success_count += 1
    
    duration = time.time() - start_time
    print(f"\n===== 测试完成 ({success_count}/{total_tests} 通过) =====")
    print(f"总耗时: {duration:.2f}秒")
    
    if success_count == total_tests:
        print("全部测试通过！")
        return 0
    else:
        print(f"有 {total_tests - success_count} 项测试未通过，请检查日志")
        return 1

if __name__ == "__main__":
    sys.exit(main())
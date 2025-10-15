"""
此文件包含向量索引的清空方法，作为对现有实现的补充。
这个方法应该被添加到VectorIndex类中。
"""

def clear(self):
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
        self.vector_cache = {}
        self.recently_used = []
        
        # 重置统计
        self.n_queries = 0
        self.total_query_time = 0
        
        # 如果有索引文件路径，保存空索引
        if hasattr(self, 'index_path') and self.index_path:
            self.save(self.index_path)
            
        self.logger.info("向量索引已清空")
        return True
    except Exception as e:
        self.logger.error(f"清空向量索引失败: {str(e)}")
        return False

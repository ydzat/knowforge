#!/usr/bin/env python3
"""
警告监控器模块
负责收集和整合处理过程中的警告信息
"""

class WarningMonitor:
    """
    警告监控器类
    
    收集和整合各种处理过程中的警告，支持对不同类型的警告进行分类和统计
    """
    
    def __init__(self):
        """初始化警告监控器"""
        self._warning_counts = {
            "not_enough_image_data": 0,
            "convert_image_error": 0, 
            "extract_region_error": 0,
            "other_warnings": 0
        }
        self._warnings_log = []
        self._enabled = True
    
    def enable(self):
        """启用警告监控"""
        self._enabled = True
    
    def disable(self):
        """禁用警告监控"""
        self._enabled = False
    
    def add_warning(self, warning_type, message):
        """
        添加警告
        
        Args:
            warning_type: 警告类型，可以是预定义类型或自定义类型
            message: 警告消息
        """
        if not self._enabled:
            return
            
        # 记录警告消息
        self._warnings_log.append({
            "type": warning_type,
            "message": message
        })
        
        # 更新计数
        if warning_type in self._warning_counts:
            self._warning_counts[warning_type] += 1
        else:
            # 如果不是预定义类型，归为其他警告
            self._warning_counts["other_warnings"] += 1
            
    def add_not_enough_image_data_warning(self, message="not enough image data"):
        """添加图像数据不足警告"""
        self.add_warning("not_enough_image_data", message)
        
    def add_convert_image_error(self, message="转换图像数据时出错"):
        """添加图像转换错误警告"""
        self.add_warning("convert_image_error", message)
        
    def add_extract_region_error(self, message="从区域提取图像时出错"):
        """添加区域提取错误警告"""
        self.add_warning("extract_region_error", message)
        
    def get_warning_counts(self):
        """
        获取警告计数
        
        Returns:
            警告计数字典
        """
        return self._warning_counts.copy()
        
    def get_warning_logs(self):
        """
        获取警告日志
        
        Returns:
            警告日志列表
        """
        return self._warnings_log.copy()
        
    def get_total_warnings(self):
        """
        获取总警告数
        
        Returns:
            警告总数
        """
        return sum(self._warning_counts.values())
        
    def reset(self):
        """重置警告计数和日志"""
        self._warning_counts = {
            "not_enough_image_data": 0,
            "convert_image_error": 0, 
            "extract_region_error": 0,
            "other_warnings": 0
        }
        self._warnings_log = []

# 全局警告监控器实例
warning_monitor = WarningMonitor()

# 导出
__all__ = ["WarningMonitor", "warning_monitor"]

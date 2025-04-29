"""
日志管理系统模块，统一管理项目日志
"""
import os
import logging
from typing import Optional


def setup_logger(log_dir: str = "output/logs", 
                log_level: int = logging.INFO, 
                log_name: str = "note_gen.log") -> logging.Logger:
    """
    设置和配置日志系统
    
    Args:
        log_dir: 日志文件存储目录
        log_level: 日志级别
        log_name: 日志文件名
    
    Returns:
        配置好的logger实例
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    # 创建logger
    logger = logging.getLogger("knowforge")
    logger.setLevel(log_level)
    
    # 清除现有handlers，确保每次调用都创建新的handlers
    # 这对测试很重要，因为测试可能使用不同的临时目录
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式器
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 为处理器设置格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 为logger添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块专用的logger实例
    
    Args:
        module_name: 模块名称
    
    Returns:
        附带模块名的logger实例
    """
    return logging.getLogger(f"knowforge.{module_name}")
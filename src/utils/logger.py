"""
安全的日志管理系统模块
提供Logloom集成和基本日志功能的回退方案
"""
import os
import sys
import logging
import yaml
from typing import Dict, Any, Optional

# 基本日志级别映射
BASIC_LEVEL_MAPPING = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 全局标志，防止递归
_INITIALIZING = False
_LANGUAGE_LOADING = False
_CONFIG_LOADING = False

# 全局日志记录器实例
_loggers: Dict[str, Any] = {}

# 导入Logloom
try:
    # 尝试导入logloom库
    from logloom import LogLevel, Logger as LogloomLogger
    from logloom import initialize, set_language, get_text, format_text
    LOGLOOM_AVAILABLE = True
    HAS_INITIALIZED = False
    
    # Logloom级别映射
    LOGLOOM_LEVEL_MAPPING = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARN,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.FATAL
    }

except ImportError:
    LOGLOOM_AVAILABLE = False
    HAS_INITIALIZED = False
    print("警告: Logloom日志系统未安装，将使用基础日志系统")
    
    # 在Logloom不可用时提供基础日志系统
    class Logger(logging.Logger):
        def __init__(self, name):
            super().__init__(name)
            self.default_module = name
            self.setLevel(logging.INFO)
            
        def set_level(self, level):
            """使用基础日志系统设置级别"""
            if isinstance(level, str):
                level = BASIC_LEVEL_MAPPING.get(level.upper(), logging.INFO)
            self.setLevel(level)
            
        def set_file(self, file_path):
            """设置日志文件"""
            handler = logging.FileHandler(file_path)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)
            
        def set_max_file_size(self, size):
            """设置最大文件大小 - 基础系统不需要实现"""
            pass

def _load_logloom_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载Logloom配置文件
    
    Args:
        config_path: 配置文件路径，如果不提供则使用默认路径
        
    Returns:
        配置字典
    """
    global _CONFIG_LOADING
    
    # 防止递归调用
    if _CONFIG_LOADING:
        return {}
        
    _CONFIG_LOADING = True
    
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "resources", "config", "logloom_config.yaml"
    )
    
    config_path = config_path or default_path
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        _CONFIG_LOADING = False
        return config.get('logloom', {})
    except Exception as e:
        _CONFIG_LOADING = False
        print(f"加载配置文件失败: {str(e)}")
        return {}

def get_logger(name: str = "knowforge") -> Any:
    """获取日志记录器实例"""
    if name not in _loggers:
        if LOGLOOM_AVAILABLE:
            # 使用Logloom原生的Logger类，不再使用自定义包装
            _loggers[name] = LogloomLogger(name)
        else:
            # 使用基础日志系统作为回退
            logging.setLoggerClass(Logger)
            basic_logger = logging.getLogger(name)
            
            if not basic_logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                basic_logger.addHandler(console_handler)
            
            _loggers[name] = basic_logger
    
    return _loggers[name]

def get_module_logger(module_name: str) -> Any:
    """
    获取模块级别Logger
    
    Args:
        module_name: 模块名称
        
    Returns:
        带模块名前缀的日志记录器
    """
    logger_name = f"knowforge.{module_name}"
    return get_logger(logger_name)

def setup_logger(log_dir: str = "output/logs",
                log_level: str = "INFO",
                log_name: str = "knowforge.log") -> Any:
    """
    初始化日志系统
    
    Args:
        log_dir: 日志目录路径
        log_level: 日志级别
        log_name: 日志文件名
    
    Returns:
        配置好的日志记录器
    """
    global HAS_INITIALIZED
    global _INITIALIZING
    
    # 防止递归调用
    if _INITIALIZING:
        return get_logger()
        
    _INITIALIZING = True
    
    # 获取主日志记录器
    logger = get_logger()
    
    # 确保日志目录存在
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name)
    except Exception as e:
        print(f"创建日志目录失败: {str(e)}")
        log_path = None
    
    if LOGLOOM_AVAILABLE and not HAS_INITIALIZED:
        try:
            # 使用配置文件路径初始化logloom
            config_file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "resources", "config", "logloom_config.yaml"
            )
            
            if os.path.exists(config_file_path):
                try:
                    # 初始化Logloom
                    initialize(config_file_path)
                    HAS_INITIALIZED = True
                    
                    # 获取配置中设置的语言
                    config = _load_logloom_config(config_file_path)
                    lang = config.get('language', 'zh')
                    safe_set_language(lang)
                    
                    # 初始化日志记录成功
                    logger.info(safe_get_text("welcome"))
                    
                except Exception as e:
                    print(f"初始化Logloom失败: {str(e)}")
            else:
                print(f"Logloom配置文件不存在: {config_file_path}")
                
        except Exception as e:
            print(f"配置Logloom失败: {str(e)}")
            
    elif not LOGLOOM_AVAILABLE:
        # 基本日志系统回退
        basic_logger = logging.getLogger("knowforge")
        level = BASIC_LEVEL_MAPPING.get(log_level.upper(), logging.INFO)
        basic_logger.setLevel(level)
        
        # 添加文件处理器
        if log_path and not any(isinstance(h, logging.FileHandler) for h in basic_logger.handlers):
            try:
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                basic_logger.addHandler(file_handler)
            except Exception as e:
                print(f"添加文件处理器失败: {str(e)}")
        
        logger = basic_logger
    
    _INITIALIZING = False
    return logger

# 为了兼容性提供这些函数
def safe_set_language(lang_code):
    """安全地设置语言，避免递归调用"""
    global _LANGUAGE_LOADING
    
    if _LANGUAGE_LOADING:
        return
        
    _LANGUAGE_LOADING = True
    
    if LOGLOOM_AVAILABLE:
        try:
            set_language(lang_code)
        except Exception as e:
            print(f"设置语言失败: {str(e)}")
    
    _LANGUAGE_LOADING = False

def safe_get_text(key, **kwargs):
    """安全的文本获取，避免递归错误"""
    if not LOGLOOM_AVAILABLE:
        return key
        
    try:
        return get_text(key, **kwargs)
    except Exception as e:
        print(f"获取文本失败: {str(e)}")
        return key

def safe_format_text(text, **kwargs):
    """安全的文本格式化，避免递归错误"""
    if not LOGLOOM_AVAILABLE:
        return text
        
    try:
        return format_text(text, **kwargs)
    except Exception as e:
        print(f"格式化文本失败: {str(e)}")
        return text

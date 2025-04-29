"""
配置加载器模块，用于加载和管理配置
"""
import os
import re
import yaml
from typing import Any, Optional, Dict
from dotenv import load_dotenv
from src.utils.exceptions import ConfigError


class ConfigLoader:
    """配置加载与管理类"""
    
    def __init__(self, config_path: str):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载环境变量
        load_dotenv()
        
        # 加载YAML配置
        try:
            self._config = self._load_yaml(config_path)
            # 处理配置中的环境变量引用
            self._process_env_vars(self._config)
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {str(e)}")
    
    def _load_yaml(self, path: str) -> dict:
        """
        加载YAML配置文件
        
        Args:
            path: 文件路径
            
        Returns:
            加载的配置字典
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML format error: {str(e)}")
    
    def _process_env_vars(self, config: Dict) -> None:
        """
        递归处理配置中的环境变量引用
        格式: ${ENV_VAR} 或 ${ENV_VAR:default_value}
        
        Args:
            config: 配置字典
        """
        if not isinstance(config, dict):
            return
            
        for key, value in config.items():
            if isinstance(value, str):
                # 查找环境变量引用格式 ${ENV_VAR} 或 ${ENV_VAR:default}
                env_matches = re.finditer(r'\${([^{}:]+)(?::([^{}]*))?}', value)
                for match in env_matches:
                    env_var = match.group(1)
                    default = match.group(2) if match.group(2) is not None else ""
                    env_value = os.getenv(env_var, default)
                    
                    # 替换环境变量
                    placeholder = match.group(0)  # 完整匹配，如 ${ENV_VAR}
                    value = value.replace(placeholder, env_value)
                
                config[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                self._process_env_vars(value)
            elif isinstance(value, list):
                # 处理列表中的每个项
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._process_env_vars(item)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过 system.language 这样的路径访问配置项
        
        Args:
            key_path: 点分隔的配置路径
            default: 默认值，当配置项不存在时返回
            
        Returns:
            配置值或默认值
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, {})
        
        return value if value != {} else default
    
    def get_env(self, env_var: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取环境变量
        
        Args:
            env_var: 环境变量名
            default: 默认值，当环境变量不存在时返回
            
        Returns:
            环境变量值或默认值
        """
        return os.getenv(env_var, default)
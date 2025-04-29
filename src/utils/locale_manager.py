"""
多语言支持管理模块，用于加载和管理多语言文本资源
"""
import yaml
import os
from typing import Dict, Optional, Any
from src.utils.exceptions import LocaleError


class LocaleManager:
    """多语言支持管理类"""
    
    # 定义内置的错误消息，用于初始化阶段
    _BUILTIN_MESSAGES = {
        "zh": {
            "locale.error.load": "无法加载语言文件: {error}",
            "locale.error.not_found": "语言文件不存在: {path}",
            "locale.error.yaml_format": "YAML格式错误: {error}",
            "locale.error.unsupported": "不支持的语言: {language}",
        },
        "en": {
            "locale.error.load": "Failed to load locale file: {error}",
            "locale.error.not_found": "Locale file not found: {path}",
            "locale.error.yaml_format": "YAML format error: {error}",
            "locale.error.unsupported": "Unsupported language: {language}",
        }
    }
    
    def __init__(self, locale_path: str, language: str = "zh"):
        """
        初始化多语言管理器
        
        Args:
            locale_path: 语言文件路径
            language: 默认语言代码，默认为中文(zh)
        """
        self.language = language
        self.messages = self._BUILTIN_MESSAGES  # 先加载内置消息作为备份
        
        try:
            custom_messages = self._load_locale(locale_path)
            
            # 合并自定义消息与内置消息
            for lang, msgs in custom_messages.items():
                if lang in self.messages:
                    self.messages[lang].update(msgs)
                else:
                    self.messages[lang] = msgs
                    
        except Exception as e:
            error_msg = self._get_builtin("locale.error.load").format(error=str(e))
            raise LocaleError(error_msg)
    
    def _get_builtin(self, key: str) -> str:
        """获取内置错误消息"""
        lang = self.language if self.language in self._BUILTIN_MESSAGES else "en"
        return self._BUILTIN_MESSAGES[lang].get(key, self._BUILTIN_MESSAGES["en"][key])
    
    def _load_locale(self, path: str) -> Dict[str, Dict[str, Any]]:
        """
        加载语言文件
        
        Args:
            path: 文件路径
            
        Returns:
            加载的语言资源字典
        """
        try:
            # 如果传入的是目录路径，则加载目录下所有yaml文件
            if os.path.isdir(path):
                result = {}
                for filename in os.listdir(path):
                    if filename.endswith('.yaml'):
                        lang_code = filename.split('.')[0]
                        file_path = os.path.join(path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result[lang_code] = yaml.safe_load(f)
                return result
            # 单个文件
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
                    
        except FileNotFoundError:
            error_msg = self._get_builtin("locale.error.not_found").format(path=path)
            raise LocaleError(error_msg)
        except yaml.YAMLError as e:
            error_msg = self._get_builtin("locale.error.yaml_format").format(error=str(e))
            raise LocaleError(error_msg)
    
    def get(self, key_path: str, default: str = "") -> str:
        """
        获取指定路径的语言文本
        
        Args:
            key_path: 点分隔的文本路径，如"system.start_message"
            default: 默认值，当文本不存在时返回
            
        Returns:
            语言文本或默认值
        """
        keys = key_path.split('.')
        value = self.messages.get(self.language, {})
        
        for key in keys:
            if not isinstance(value, dict):
                return default if not isinstance(value, str) else value
            value = value.get(key, {})
        
        return default if not isinstance(value, str) else value
    
    def set_language(self, language: str) -> None:
        """
        切换当前语言
        
        Args:
            language: 语言代码，如"zh", "en"
        """
        if language not in self.messages:
            error_msg = self._get_builtin("locale.error.unsupported").format(language=language)
            raise LocaleError(error_msg)
        self.language = language
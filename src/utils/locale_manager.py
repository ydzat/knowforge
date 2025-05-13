"""
多语言支持管理模块，用于加载和管理多语言文本资源
"""
import yaml
import os
import sys
from typing import Dict, Optional, Any, Set
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
            "locale.error.recursive": "检测到递归调用获取文本: {key}",
            "locale.warn.missing_key": "未找到语言键: {key}",
            "locale.info.loaded": "已加载语言文件: {path}",
            "locale.debug.get_text": "获取文本: {key}",
            "locale.error.format": "格式化文本失败: {key}, 错误: {error}",
        },
        "en": {
            "locale.error.load": "Failed to load locale file: {error}",
            "locale.error.not_found": "Locale file not found: {path}",
            "locale.error.yaml_format": "YAML format error: {error}",
            "locale.error.unsupported": "Unsupported language: {language}",
            "locale.error.recursive": "Recursive text lookup detected: {key}",
            "locale.warn.missing_key": "Language key not found: {key}",
            "locale.info.loaded": "Loaded language file: {path}",
            "locale.debug.get_text": "Getting text: {key}",
            "locale.error.format": "Failed to format text: {key}, error: {error}",
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
        self.messages = self._BUILTIN_MESSAGES.copy()  # 先加载内置消息作为备份
        self._recursion_guard: Set[str] = set()  # 用于防止递归调用
        
        try:
            custom_messages = self._load_locale(locale_path)
            
            # 合并自定义消息与内置消息
            for lang, msgs in custom_messages.items():
                if lang in self.messages:
                    self.messages[lang].update(msgs)
                else:
                    self.messages[lang] = msgs
                    
            # 记录成功加载
            log_message = self._get_builtin("locale.info.loaded", {"path": locale_path})
            sys.stdout.write(f"[LocaleManager] {log_message}\n")
                    
        except Exception as e:
            error_msg = self._get_builtin("locale.error.load", {"error": str(e)})
            # 输出错误到控制台，避免使用logger造成递归
            sys.stderr.write(f"[LocaleManager] ERROR: {error_msg}\n")
            raise LocaleError(error_msg)
    
    def _get_builtin(self, key: str, params: Dict[str, Any] = None) -> str:
        """
        获取内置错误消息
        
        Args:
            key: 消息键
            params: 格式化参数
        
        Returns:
            格式化后的内置消息
        """
        lang = self.language if self.language in self._BUILTIN_MESSAGES else "en"
        text = self._BUILTIN_MESSAGES[lang].get(key, key)
        
        # 如果指定语言中没有这个键，尝试从英文中获取
        if text == key and lang != "en":
            text = self._BUILTIN_MESSAGES["en"].get(key, key)
            
        # 格式化文本
        if params and text != key:
            try:
                return text.format(**params)
            except Exception:
                return text
                
        return text
    
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
            error_msg = self._get_builtin("locale.error.not_found", {"path": path})
            raise LocaleError(error_msg)
        except yaml.YAMLError as e:
            error_msg = self._get_builtin("locale.error.yaml_format", {"error": str(e)})
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
        # 防止递归调用
        if key_path in self._recursion_guard:
            error_msg = self._get_builtin("locale.error.recursive", {"key": key_path})
            sys.stderr.write(f"[LocaleManager] WARNING: {error_msg}\n")
            return f"[RECURSIVE] {key_path}"
            
        try:
            self._recursion_guard.add(key_path)
            
            keys = key_path.split('.')
            value = self.messages.get(self.language, {})
            
            for key in keys:
                if not isinstance(value, dict):
                    # 如果当前值不是字典，就不能继续查找了
                    return default if not isinstance(value, str) else value
                value = value.get(key, {})
            
            # 如果最终值不是字符串，或者没有找到值，返回默认值
            if not isinstance(value, str) or value == {}:
                # 如果在当前语言中没找到，尝试英文
                if self.language != "en":
                    value_en = self.messages.get("en", {})
                    for key in keys:
                        if not isinstance(value_en, dict):
                            break
                        value_en = value_en.get(key, {})
                    
                    if isinstance(value_en, str) and value_en != {}:
                        return value_en
                
                # 如果在英文中也没找到或没有替代值，返回默认值
                return default if default else key_path
            
            return value
        finally:
            # 无论成功与否，都从递归保护中移除
            self._recursion_guard.discard(key_path)
    
    def format(self, key_path: str, params: Dict[str, Any], default: str = "") -> str:
        """
        获取并格式化指定路径的语言文本
        
        Args:
            key_path: 点分隔的文本路径
            params: 格式化参数
            default: 默认值，当文本不存在时返回
            
        Returns:
            格式化后的语言文本
        """
        text = self.get(key_path, default)
        
        # 如果没有参数或文本是默认值，直接返回
        if not params or text == default:
            return text
            
        try:
            return text.format(**params)
        except KeyError as e:
            # 参数错误，输出警告并返回未格式化文本
            error_msg = self._get_builtin("locale.error.format", {"key": key_path, "error": str(e)})
            sys.stderr.write(f"[LocaleManager] WARNING: {error_msg}\n")
            return text
        except Exception as e:
            # 其他错误，返回未格式化文本
            return text
    
    def set_language(self, language: str) -> None:
        """
        切换当前语言
        
        Args:
            language: 语言代码，如"zh", "en"
        """
        if language not in self.messages:
            error_msg = self._get_builtin("locale.error.unsupported", {"language": language})
            raise LocaleError(error_msg)
        self.language = language

def safe_get_text(key: str, params: Dict[str, Any] = None, default: str = None, lang: str = None) -> str:
    """
    安全地获取本地化文本，避免循环依赖
    
    Args:
        key: 文本键
        params: 格式化参数
        default: 默认文本
        lang: 语言代码
    
    Returns:
        本地化文本
    """
    # 使用全局LocaleManager实例或创建临时实例
    try:
        from src.utils.locale_manager import LocaleManager
        locale_manager = LocaleManager("resources/locales")
        return locale_manager.format(key, params or {}, default or key)
    except Exception:
        # 如果LocaleManager不可用，使用内置文本
        builtin_texts = {
            "zh": {
                "welcome": "欢迎使用KnowForge - AI驱动的学习笔记生成器",
                "output.timestamp_label": "生成时间",
                "output.source_label": "来源",
                "output.toc": "目录",
                "output.footer": "由KnowForge v{version}生成"
            },
            "en": {
                "welcome": "Welcome to KnowForge - AI-powered Learning Notes Generator",
                "output.timestamp_label": "Generated at",
                "output.source_label": "Source",
                "output.toc": "Table of Contents",
                "output.footer": "Generated by KnowForge v{version}"
            }
        }
        lang = lang or "zh"
        text = builtin_texts.get(lang, {}).get(key, key)
        if params:
            try:
                return text.format(**params)
            except Exception:
                return text
        return text

def safe_format_text(key: str, *args, **kwargs) -> str:
    """
    安全地格式化本地化文本，避免循环依赖
    
    Args:
        key: 文本键
        *args: 位置参数
        **kwargs: 关键字参数
            lang: 可选的语言代码
    
    Returns:
        格式化后的本地化文本
    """
    lang = kwargs.pop('lang', None)
    params = kwargs if kwargs else (args[0] if args else None)
    return safe_get_text(key, params, None, lang)

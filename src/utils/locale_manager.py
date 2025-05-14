"""
多语言支持管理模块（Logloom封装）
此模块已完全迁移到Logloom，作为适配层存在
"""
import os
import sys
from typing import Dict, Any, Optional, List

# 直接导入logloom全部国际化相关API
from logloom import (
    initialize, 
    set_language, 
    get_text, 
    format_text, 
    register_locale_file, 
    register_locale_directory, 
    get_supported_languages,
    get_language_keys
)


class LocaleManager:
    """
    多语言支持管理类
    
    此类现为Logloom的轻量级封装，完全依赖Logloom实现
    单纯作为现有代码到Logloom API的适配层，降低迁移成本
    
    项目中新代码应直接使用Logloom API，不应使用此类
    """
    
    def __init__(self, locale_path: str, language: str = "zh"):
        """
        初始化多语言管理器，使用Logloom实现
        
        Args:
            locale_path: 语言文件路径，可以是目录或单个文件
            language: 默认语言代码，默认为中文(zh)
        """
        self.language = language
        
        # 查找配置文件
        config_path = None
        
        if os.path.isdir(locale_path):
            # 假设目录结构是 resources/locales，尝试找到 resources/config/logloom_config.yaml
            config_parent_dir = os.path.dirname(os.path.dirname(locale_path))
            config_path = os.path.join(config_parent_dir, "config", "logloom_config.yaml")
            
            # 初始化Logloom
            if os.path.exists(config_path):
                initialize(config_path)
                sys.stdout.write(f"[LocaleManager] Logloom已初始化，配置文件：{config_path}\n")
                
                # 注册语言资源目录，确保所有翻译文件被加载
                register_locale_directory(locale_path)
                sys.stdout.write(f"[LocaleManager] 已注册语言资源目录：{locale_path}\n")
            else:
                # 使用默认初始化
                initialize()
                sys.stdout.write("[LocaleManager] Logloom已使用默认配置初始化\n")
                
                # 手动注册目录
                register_locale_directory(locale_path)
                sys.stdout.write(f"[LocaleManager] 已注册语言资源目录：{locale_path}\n")
        else:
            # 单个文件情况
            initialize()
            sys.stdout.write("[LocaleManager] Logloom已使用默认配置初始化\n")
            
            # 如果是单个文件，注册该文件
            if os.path.exists(locale_path):
                register_locale_file(locale_path)
                sys.stdout.write(f"[LocaleManager] 已注册语言资源文件：{locale_path}\n")
        
        # 打印支持的语言
        supported_langs = get_supported_languages()
        sys.stdout.write(f"[LocaleManager] 支持的语言: {', '.join(supported_langs)}\n")
        
        # 设置语言
        set_language(language)
        sys.stdout.write(f"[LocaleManager] 当前语言设置为: {language}\n")
        
        # 缓存键名映射，用于兼容两种键名格式
        self._key_cache = {}
        self._prefixed_keys = {}
        self._load_keys()
    
    def _load_keys(self):
        """加载并缓存当前语言的所有键"""
        # 获取所有键名
        keys = get_language_keys(self.language)
        if not keys:
            return
            
        # 建立映射关系
        for key in keys:
            # 如果键名是"lang.section.name"格式
            if key.startswith(f"{self.language}."):
                # 截取掉语言前缀部分
                simple_key = key[len(self.language)+1:]
                self._key_cache[simple_key] = key
                # 也保存最后一部分作为简化键
                parts = simple_key.split('.')
                if len(parts) > 1:
                    self._key_cache[parts[-1]] = key
            # 如果键名是直接的文本键如"welcome"
            else:
                self._key_cache[key] = key
                # 添加有语言前缀的键
                self._prefixed_keys[f"{self.language}.{key}"] = key
    
    def _resolve_key(self, key_path: str) -> str:
        """将简单键转换为完整键"""
        if key_path in self._key_cache:
            return self._key_cache[key_path]
            
        # 检查带语言前缀的键
        prefixed_key = f"{self.language}.{key_path}"
        if prefixed_key in self._prefixed_keys:
            return prefixed_key
            
        # 如果没有找到映射，尝试刷新键缓存
        self._load_keys()
        
        # 再次尝试查找
        if key_path in self._key_cache:
            return self._key_cache[key_path]
            
        # 如果仍然找不到，返回原始键名
        return key_path
    
    def get(self, key_path: str, default: str = "") -> str:
        """
        获取指定路径的语言文本
        
        Args:
            key_path: 点分隔的文本路径，如"system.error"或简单键名如"welcome"
            default: 默认值，当文本不存在时返回
            
        Returns:
            语言文本或默认值
        """
        # 解析键名，支持简单键"welcome"和完整键"system.error"
        resolved_key = self._resolve_key(key_path)
        
        # 使用解析后的键获取文本
        text = get_text(resolved_key)
        
        # 如果Logloom返回了原始key，表示未找到翻译
        if text == resolved_key:
            # 如果返回的是解析后的键，再尝试直接用原始键
            if resolved_key != key_path:
                text = get_text(key_path)
                if text != key_path:
                    return text
            return default if default else key_path
        
        return text
    
    def get_text(self, key_path: str, default: str = "") -> str:
        """
        获取指定路径的语言文本 (Logloom API兼容方法)
        
        Args:
            key_path: 点分隔的文本路径，如"system.start_message"
            default: 默认值，当文本不存在时返回
            
        Returns:
            语言文本或默认值
        """
        return self.get(key_path, default)
    
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
        # 解析键名
        resolved_key = self._resolve_key(key_path)
        
        # 首先尝试获取文本
        text = get_text(resolved_key)
        
        # 如果返回原始key，表示未找到翻译
        if text == resolved_key:
            # 如果返回的是解析后的键，再尝试直接用原始键
            if resolved_key != key_path:
                text = get_text(key_path)
                if text != key_path:
                    # 如果找到了翻译，使用格式化参数
                    if params:
                        try:
                            return format_text(key_path, **params)
                        except Exception as e:
                            sys.stderr.write(f"[LocaleManager] 格式化文本失败: {key_path}, 错误: {str(e)}\n")
                            try:
                                return text.format(**params)
                            except Exception:
                                return text
                    return text
            
            # 如果真的找不到，使用默认值
            text = default if default else key_path
            
            # 如果有默认值且有参数，尝试格式化默认值
            if default and params:
                try:
                    return default.format(**params)
                except Exception as e:
                    sys.stderr.write(f"[LocaleManager] 格式化默认值失败: {str(e)}\n")
            
        # 如果没有参数，直接返回文本
        if not params:
            return text
            
        # 使用Logloom的format_text格式化文本
        try:
            return format_text(resolved_key, **params)
        except Exception as e:
            sys.stderr.write(f"[LocaleManager] 格式化文本失败: {resolved_key}, 错误: {str(e)}\n")
            
            # 尝试直接格式化文本
            try:
                return text.format(**params)
            except Exception:
                return text
    
    def format_text(self, key_path: str, **kwargs) -> str:
        """
        获取并格式化指定路径的语言文本 (Logloom API方法)
        
        Args:
            key_path: 点分隔的文本路径
            **kwargs: 格式化参数
            
        Returns:
            格式化后的语言文本
        """
        return self.format(key_path, kwargs)
    
    def set_language(self, language: str) -> None:
        """
        切换当前语言
        
        Args:
            language: 语言代码，如"zh", "en"
        """
        if self.language != language:
            self.language = language
            set_language(language)
            # 切换语言后刷新键缓存
            self._key_cache = {}
            self._prefixed_keys = {}
            self._load_keys()
    
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            语言代码列表
        """
        return get_supported_languages()
    
    def get_language_keys(self, lang_code: str = None) -> List[str]:
        """
        获取指定语言的所有翻译键
        解析后返回不带语言前缀的简单键名
        
        Args:
            lang_code: 语言代码，如不指定则使用当前语言
            
        Returns:
            翻译键列表
        """
        lang = lang_code or self.language
        keys = get_language_keys(lang)
        
        # 处理返回的键，移除语言前缀
        simple_keys = []
        prefix = f"{lang}."
        for key in keys:
            if key.startswith(prefix):
                simple_keys.append(key[len(prefix):])
            else:
                simple_keys.append(key)
                
        return simple_keys


# 提供直接的函数访问，与Logloom保持一致
def safe_get_text(key: str, params: Dict[str, Any] = None, default: str = None, lang: str = None) -> str:
    """
    安全地获取本地化文本
    
    此函数现在直接使用Logloom API
    
    Args:
        key: 文本键
        params: 格式化参数
        default: 默认文本
        lang: 语言代码
    
    Returns:
        本地化文本
    """
    # 使用LocaleManager获取文本（确保键名正确解析）
    # 获取单例实例
    locale_manager = _get_locale_manager_instance()
    
    # 临时切换语言（如果需要）
    original_lang = None
    if lang and lang != locale_manager.language:
        original_lang = locale_manager.language
        locale_manager.set_language(lang)
        
    try:
        # 使用LocaleManager获取文本
        text = locale_manager.get(key, default or "")
        
        # 处理格式化
        if params and text:
            try:
                if isinstance(params, dict):
                    return locale_manager.format(key, params)
                else:
                    return locale_manager.format(key, {"value": params})
            except Exception as e:
                sys.stderr.write(f"[LocaleManager] 格式化文本失败: {str(e)}\n")
                return text
        return text
    finally:
        # 恢复原始语言
        if original_lang:
            locale_manager.set_language(original_lang)


def safe_format_text(key: str, *args, **kwargs) -> str:
    """
    安全地格式化本地化文本
    
    此函数现在直接使用Logloom API
    
    Args:
        key: 文本键
        *args: 位置参数
        **kwargs: 关键字参数
            lang: 可选的语言代码
    
    Returns:
        格式化后的本地化文本
    """
    # 提取并移除lang参数，如果存在的话
    lang = kwargs.pop('lang', None) if kwargs else None
    
    # 获取LocaleManager单例
    locale_manager = _get_locale_manager_instance()
    
    # 临时切换语言（如果需要）
    original_lang = None
    if lang and lang != locale_manager.language:
        original_lang = locale_manager.language
        locale_manager.set_language(lang)
    
    try:
        # 使用LocaleManager格式化文本
        if args:
            # 将位置参数转为字典
            if len(args) == 1 and isinstance(args[0], dict):
                return locale_manager.format(key, args[0])
            else:
                # 创建包含位置参数的字典，用arg0, arg1等作为键
                params = {f"arg{i}": arg for i, arg in enumerate(args)}
                params.update(kwargs)
                return locale_manager.format(key, params)
        else:
            return locale_manager.format(key, kwargs)
    finally:
        # 恢复原始语言
        if original_lang:
            locale_manager.set_language(original_lang)


# 单例模式辅助函数
_LOCALE_MANAGER_INSTANCE = None

def _get_locale_manager_instance():
    """获取或创建LocaleManager单例实例"""
    global _LOCALE_MANAGER_INSTANCE
    if _LOCALE_MANAGER_INSTANCE is None:
        # 默认使用resources/locales目录
        locale_path = os.path.join("resources", "locales")
        if os.path.exists(locale_path):
            _LOCALE_MANAGER_INSTANCE = LocaleManager(locale_path)
        else:
            # 如果找不到默认路径，使用当前目录作为备用
            _LOCALE_MANAGER_INSTANCE = LocaleManager(".")
    return _LOCALE_MANAGER_INSTANCE

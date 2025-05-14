"""
LocaleManager单元测试（基于Logloom实现）
"""
import os
import pytest
from src.utils.locale_manager import LocaleManager, safe_get_text, safe_format_text


@pytest.fixture
def locale_manager():
    """创建LocaleManager测试实例"""
    locale_path = os.path.join("resources", "locales")
    return LocaleManager(locale_path, 'zh')


class TestLocaleManager:
    """LocaleManager类单元测试（Logloom封装版）"""
    
    def test_locale_manager_initialization(self, locale_manager):
        """测试LocaleManager初始化"""
        # 验证语言设置正确
        assert locale_manager.language == 'zh'
        
        # 验证支持的语言列表非空
        langs = locale_manager.get_supported_languages()
        assert isinstance(langs, list)
        assert len(langs) > 0
    
    def test_get_method(self, locale_manager):
        """测试get方法"""
        # 测试获取已存在的键 - 这里只验证不是空值，不检查具体内容
        text = locale_manager.get('welcome')
        assert isinstance(text, str)
        assert text != ''  # 应该返回非空文本
        
        # 尝试获取system.error这种多级键
        system_error = locale_manager.get('system.error')
        assert isinstance(system_error, str)
        assert system_error != ''
        
        # 测试获取不存在的键（应该返回默认值或键名）
        text_non_existent = locale_manager.get('non.existent.key', 'default text')
        assert text_non_existent == 'default text'
        
        # 不提供默认值时应该返回键名
        text_no_default = locale_manager.get('another.non.existent')
        assert text_no_default == 'another.non.existent'
    
    def test_get_text_method(self, locale_manager):
        """测试get_text方法（Logloom API兼容）"""
        # get_text应该和get方法行为一致
        assert locale_manager.get_text('welcome') == locale_manager.get('welcome')
        assert locale_manager.get_text('non.existent', 'default') == 'default'
    
    def test_format_method(self, locale_manager):
        """测试format方法"""
        # 测试格式化已存在的键
        params = {'message': 'test message'}
        formatted = locale_manager.format('system.error', params)
        assert isinstance(formatted, str)
        assert formatted != ''  # 应该返回非空文本
        
        # 测试格式化不存在的键（应该返回默认值并格式化）
        default = 'Default text with {param}'
        params = {'param': 'value'}
        formatted_non_existent = locale_manager.format('non.existent', params, default)
        assert formatted_non_existent == 'Default text with value'
        
        # 无参数调用
        text = locale_manager.format('welcome', {})
        assert text == locale_manager.get('welcome')
    
    def test_format_text_method(self, locale_manager):
        """测试format_text方法（Logloom API兼容）"""
        # format_text应该和format方法行为一致，但接受kwargs
        result1 = locale_manager.format('system.error', {'message': 'test message'})
        result2 = locale_manager.format_text('system.error', message='test message')
        assert result1 == result2
    
    def test_set_language(self, locale_manager):
        """测试语言切换"""
        # 保存中文文本
        zh_welcome = locale_manager.get('welcome')
        
        # 如果支持英语，测试语言切换
        langs = locale_manager.get_supported_languages()
        if 'en' in langs:
            # 切换到英文
            locale_manager.set_language('en')
            assert locale_manager.language == 'en'
            
            # 获取英文文本
            en_welcome = locale_manager.get('welcome')
            
            # 如果英文翻译存在，检查两种语言的文本不同
            if en_welcome and en_welcome != 'welcome':
                assert zh_welcome != en_welcome
            
            # 切回中文
            locale_manager.set_language('zh')
            assert locale_manager.language == 'zh'
    
    def test_key_resolution(self, locale_manager):
        """测试键名解析功能"""
        # 缓存键名映射
        locale_manager._load_keys()
        
        # 尝试解析各种格式的键
        # 简单键，没有点号
        simple_key = 'welcome'
        resolved_simple = locale_manager._resolve_key(simple_key)
        assert isinstance(resolved_simple, str)
        
        # 带点的多级键
        nested_key = 'system.error'
        resolved_nested = locale_manager._resolve_key(nested_key)
        assert isinstance(resolved_nested, str)
        
        # 不存在的键应该返回原始键名
        non_existent = 'non.existent.key'
        resolved_non_existent = locale_manager._resolve_key(non_existent)
        assert resolved_non_existent == non_existent
    
    def test_get_language_keys(self, locale_manager):
        """测试获取语言键列表"""
        # 获取当前语言的所有键
        keys = locale_manager.get_language_keys()
        assert isinstance(keys, list)
        # 至少应该有一些键
        assert len(keys) > 0


class TestSafeFunctions:
    """测试safe_get_text和safe_format_text函数"""
    
    def test_safe_get_text(self):
        """测试safe_get_text函数"""
        # 测试基本功能
        text = safe_get_text('welcome')
        assert isinstance(text, str)
        assert text != ''  # 应该返回非空文本
        
        # 测试默认值
        text_with_default = safe_get_text('non.existent.key', default='default text')
        assert text_with_default == 'default text'
        
        # 测试带参数
        params = {'message': 'test message'}
        formatted = safe_get_text('system.error', params)
        assert isinstance(formatted, str)
    
    def test_safe_format_text(self):
        """测试safe_format_text函数"""
        # 测试基本格式化
        formatted = safe_format_text('system.error', message='test message')
        assert isinstance(formatted, str)
        
        # 测试位置参数格式
        formatted2 = safe_format_text('system.error', {'message': 'test message'})
        assert isinstance(formatted2, str)
        
        # 测试默认键名回退
        non_existent = safe_format_text('non.existent.key', value='test')
        assert isinstance(non_existent, str)
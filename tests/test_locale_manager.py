"""
LocaleManager单元测试
"""
import os
import pytest
import tempfile
import yaml
from src.utils.locale_manager import LocaleManager
from src.utils.exceptions import LocaleError


class TestLocaleManager:
    """LocaleManager类单元测试"""
    
    def test_load_valid_locale(self):
        """测试加载有效的语言文件"""
        # 创建临时语言文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({
                'zh': {
                    'system': {
                        'start_message': '程序启动',
                        'error_occurred': '发生错误'
                    }
                },
                'en': {
                    'system': {
                        'start_message': 'Program started',
                        'error_occurred': 'Error occurred'
                    }
                }
            }, temp)
        
        try:
            # 测试加载语言
            locale = LocaleManager(temp.name, 'zh')
            assert locale.get('system.start_message') == '程序启动'
            assert locale.get('system.error_occurred') == '发生错误'
            
            # 切换语言
            locale.set_language('en')
            assert locale.get('system.start_message') == 'Program started'
            assert locale.get('system.error_occurred') == 'Error occurred'
            
            # 测试默认值
            assert locale.get('nonexistent') == ''
            assert locale.get('nonexistent', 'default') == 'default'
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_load_invalid_yaml(self):
        """测试加载无效的YAML语言文件"""
        # 创建格式错误的YAML临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            temp.write('invalid: yaml: content:')
        
        try:
            # 应该抛出LocaleError
            with pytest.raises(LocaleError):
                LocaleManager(temp.name)
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的语言文件"""
        with pytest.raises(LocaleError):
            LocaleManager('nonexistent_file.yaml')
    
    def test_get_nested_text(self):
        """测试获取嵌套的语言文本"""
        # 创建嵌套的语言文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({
                'zh': {
                    'level1': {
                        'level2': {
                            'message': '深层消息'
                        }
                    }
                }
            }, temp)
        
        try:
            locale = LocaleManager(temp.name, 'zh')
            assert locale.get('level1.level2.message') == '深层消息'
            assert locale.get('level1.level2.nonexistent') == ''
            assert locale.get('level1.nonexistent.message', '默认值') == '默认值'
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_set_invalid_language(self):
        """测试设置不支持的语言"""
        # 创建只有中文的语言文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({
                'zh': {
                    'message': '消息'
                }
            }, temp)
        
        try:
            locale = LocaleManager(temp.name, 'zh')
            
            # 尝试设置不存在的语言
            with pytest.raises(LocaleError):
                locale.set_language('fr')  # 法语不在文件中
        finally:
            # 清理临时文件
            os.unlink(temp.name)
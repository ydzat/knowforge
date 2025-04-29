"""
ConfigLoader单元测试
"""
import os
import pytest
import tempfile
import yaml
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import ConfigError


class TestConfigLoader:
    """ConfigLoader类单元测试"""
    
    def test_load_valid_config(self):
        """测试加载有效的配置文件"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({
                'system': {
                    'language': 'zh',
                    'workspace_dir': 'test_workspace/'
                }
            }, temp)
        
        try:
            # 测试加载配置
            config = ConfigLoader(temp.name)
            assert config.get('system.language') == 'zh'
            assert config.get('system.workspace_dir') == 'test_workspace/'
            assert config.get('nonexistent') is None
            assert config.get('nonexistent', 'default') == 'default'
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_load_invalid_yaml(self):
        """测试加载无效的YAML文件"""
        # 创建格式错误的YAML临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            temp.write('invalid: yaml: content:')
        
        try:
            # 应该抛出ConfigError
            with pytest.raises(ConfigError):
                ConfigLoader(temp.name)
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的配置文件"""
        with pytest.raises(ConfigError):
            ConfigLoader('nonexistent_file.yaml')
    
    def test_get_config_nested(self):
        """测试获取嵌套配置项"""
        # 创建嵌套配置的临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({
                'level1': {
                    'level2': {
                        'level3': 'deep_value'
                    }
                }
            }, temp)
        
        try:
            config = ConfigLoader(temp.name)
            assert config.get('level1.level2.level3') == 'deep_value'
            assert config.get('level1.level2.nonexistent') is None
            assert config.get('level1.nonexistent.level3') is None
        finally:
            # 清理临时文件
            os.unlink(temp.name)
    
    def test_get_env(self, monkeypatch):
        """测试获取环境变量"""
        # 设置测试环境变量
        monkeypatch.setenv('TEST_ENV_VAR', 'test_value')
        
        # 创建任意配置文件（不影响环境变量测试）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
            yaml.dump({'dummy': 'content'}, temp)
        
        try:
            config = ConfigLoader(temp.name)
            assert config.get_env('TEST_ENV_VAR') == 'test_value'
            assert config.get_env('NONEXISTENT_ENV_VAR') is None
            assert config.get_env('NONEXISTENT_ENV_VAR', 'default') == 'default'
        finally:
            # 清理临时文件
            os.unlink(temp.name)
"""
异常类模块单元测试
"""
import pytest
from src.utils.exceptions import (
    NoteGenError, 
    ConfigError, 
    InputError, 
    APIError, 
    MemoryError, 
    OutputError, 
    LocaleError
)


class TestExceptions:
    """异常类单元测试"""
    
    def test_base_exception(self):
        """测试基础异常类"""
        # 创建基础异常
        error_message = "测试基础异常"
        error = NoteGenError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, Exception)
    
    def test_config_error(self):
        """测试配置相关异常"""
        # 创建配置异常
        error_message = "配置错误"
        error = ConfigError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
        assert isinstance(error, Exception)
    
    def test_input_error(self):
        """测试输入相关异常"""
        # 创建输入异常
        error_message = "输入错误"
        error = InputError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
    
    def test_api_error(self):
        """测试API相关异常"""
        # 创建API异常
        error_message = "API错误"
        error = APIError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
    
    def test_memory_error(self):
        """测试记忆相关异常"""
        # 创建记忆异常
        error_message = "记忆错误"
        error = MemoryError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
    
    def test_output_error(self):
        """测试输出相关异常"""
        # 创建输出异常
        error_message = "输出错误"
        error = OutputError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
    
    def test_locale_error(self):
        """测试多语言支持相关异常"""
        # 创建语言异常
        error_message = "语言错误"
        error = LocaleError(error_message)
        
        # 验证异常消息
        assert str(error) == error_message
        # 验证异常继承关系
        assert isinstance(error, NoteGenError)
    
    def test_exception_handling(self):
        """测试异常捕获和处理"""
        try:
            # 故意抛出异常
            raise ConfigError("配置文件缺失")
        except NoteGenError as e:
            # 验证可以捕获子类异常
            assert "配置文件缺失" in str(e)
        
        # 验证不同类型异常可以区分
        with pytest.raises(ConfigError):
            raise ConfigError("配置错误")
            
        with pytest.raises(InputError):
            raise InputError("输入错误")
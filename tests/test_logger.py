"""
Logger模块单元测试
"""
import os
import pytest
import logging
import tempfile
import shutil
from src.utils.logger import setup_logger, get_module_logger


class TestLogger:
    """Logger模块单元测试"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """创建临时日志目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_setup_logger(self, temp_log_dir):
        """测试设置Logger的基本功能"""
        # 配置日志系统
        logger = setup_logger(log_dir=temp_log_dir, log_name="test.log")
        
        # 检查logger是否正确配置
        assert logger.name == "knowforge"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # 文件处理器和控制台处理器
        
        # 检查日志文件是否创建
        log_file = os.path.join(temp_log_dir, "test.log")
        assert os.path.exists(log_file)
        
        # 写入日志
        test_message = "测试日志消息"
        logger.info(test_message)
        
        # 验证日志是否写入文件
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            assert test_message in log_content
    
    def test_log_level(self, temp_log_dir):
        """测试日志级别过滤"""
        # 使用DEBUG级别的日志器
        logger = setup_logger(
            log_dir=temp_log_dir, 
            log_level=logging.DEBUG, 
            log_name="debug.log"
        )
        
        # 写入各级别日志
        logger.debug("DEBUG消息")
        logger.info("INFO消息")
        logger.warning("WARNING消息")
        logger.error("ERROR消息")
        
        # 验证所有日志都写入了文件
        log_file = os.path.join(temp_log_dir, "debug.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            assert "DEBUG消息" in log_content
            assert "INFO消息" in log_content
            assert "WARNING消息" in log_content
            assert "ERROR消息" in log_content
        
        # 使用INFO级别的日志器
        logger = setup_logger(
            log_dir=temp_log_dir, 
            log_level=logging.INFO, 
            log_name="info.log"
        )
        
        # 写入各级别日志
        logger.debug("另一个DEBUG消息")
        logger.info("另一个INFO消息")
        
        # 验证只有INFO及以上级别被记录
        log_file = os.path.join(temp_log_dir, "info.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            assert "另一个DEBUG消息" not in log_content  # DEBUG不应被记录
            assert "另一个INFO消息" in log_content      # INFO应被记录
    
    def test_get_module_logger(self):
        """测试获取模块级别Logger"""
        module_name = "test_module"
        logger = get_module_logger(module_name)
        
        # 验证logger名称
        assert logger.name == f"knowforge.{module_name}"
        
    def test_log_dir_creation(self):
        """测试自动创建日志目录"""
        # 使用不存在的目录路径
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = os.path.join(temp_dir, "logs", "nested")
            
            # 确保目录不存在
            assert not os.path.exists(nonexistent_dir)
            
            # 设置logger应该自动创建目录
            logger = setup_logger(log_dir=nonexistent_dir)
            
            # 验证目录被创建
            assert os.path.exists(nonexistent_dir)
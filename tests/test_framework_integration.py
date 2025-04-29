"""
基础框架集成测试 - 验证迭代1完成的组件能够正确协同工作
"""
import os
import pytest
import tempfile
import yaml
from typer.testing import CliRunner
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager
from src.utils.logger import setup_logger
from src.cli.cli_main import cli
from src import __version__


class TestFrameworkIntegration:
    """测试基础框架组件集成"""
    
    @pytest.fixture
    def runner(self):
        """返回CLI测试运行器"""
        return CliRunner()
    
    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置文件目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建配置文件
            os.makedirs(os.path.join(temp_dir, "config"))
            config_path = os.path.join(temp_dir, "config", "config.yaml")
            
            # 创建配置内容
            config = {
                "system": {
                    "language": "zh",
                    "workspace_dir": "workspace/",
                    "output_dir": "output/"
                }
            }
            
            # 写入配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            # 创建语言文件目录
            os.makedirs(os.path.join(temp_dir, "locales"))
            locale_path = os.path.join(temp_dir, "locales", "locale.yaml")
            
            # 创建语言文件内容
            locale_content = {
                "zh": {
                    "system": {
                        "start_message": "程序启动",
                        "completed": "处理完成"
                    }
                },
                "en": {
                    "system": {
                        "start_message": "Program started",
                        "completed": "Processing completed"
                    }
                }
            }
            
            # 写入语言文件
            with open(locale_path, 'w', encoding='utf-8') as f:
                yaml.dump(locale_content, f)
                
            yield temp_dir
    
    def test_config_locale_integration(self, temp_config_dir):
        """测试配置和本地化模块集成"""
        config_path = os.path.join(temp_config_dir, "config", "config.yaml")
        locale_path = os.path.join(temp_config_dir, "locales", "locale.yaml")
        
        # 测试配置加载
        config = ConfigLoader(config_path)
        assert config.get("system.language") == "zh"
        
        # 测试语言加载（使用配置中的语言）
        language = config.get("system.language")
        locale = LocaleManager(locale_path, language)
        
        # 测试获取本地化文本
        assert locale.get("system.start_message") == "程序启动"
        
        # 测试语言切换
        locale.set_language("en")
        assert locale.get("system.start_message") == "Program started"
    
    def test_logger_integration(self):
        """测试日志系统集成"""
        with tempfile.TemporaryDirectory() as temp_log_dir:
            # 设置日志
            logger = setup_logger(log_dir=temp_log_dir, log_name="integration_test.log")
            
            # 写入日志
            test_message = "测试集成日志"
            logger.info(test_message)
            
            # 验证日志写入
            log_file = os.path.join(temp_log_dir, "integration_test.log")
            assert os.path.exists(log_file)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert test_message in content
    
    def test_cli_version_command(self, runner):
        """测试CLI版本命令"""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert f"KnowForge v{__version__}" in result.stdout
    
    @pytest.mark.skip(reason="依赖LLM，跳过单元测试，仅在真实环境下验证")
    def test_cli_generate_command(self, runner):
        pass
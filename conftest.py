"""
pytest配置文件
"""
import pytest

# 告诉pytest只在tests目录中收集测试
collect_ignore = ["setup.py", "input"]
pytest_plugins = []

def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line("norecursedirs", "input")
    
def pytest_collection_modifyitems(config, items):
    """修改收集的测试项"""
    # 这里可以进一步处理收集到的测试项
    pass

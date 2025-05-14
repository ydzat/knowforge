"""
测试Logloom与LocaleManager集成
"""
import pytest
from src.utils.locale_manager import LocaleManager, safe_get_text, safe_format_text
from src.utils.logger import get_logger

# 这个测试假设Logloom可用，如果不可用就跳过
try:
    from logloom import get_text, format_text, set_language
    LOGLOOM_AVAILABLE = True
except ImportError:
    LOGLOOM_AVAILABLE = False
    
pytestmark = pytest.mark.skipif(not LOGLOOM_AVAILABLE, reason="Logloom不可用")

def test_locale_manager_init():
    """测试LocaleManager初始化并正确连接到Logloom"""
    # 使用正确的配置路径
    locale_path = "resources/locales"
    locale_manager = LocaleManager(locale_path, "zh")
    
    # 验证基本获取文本功能
    welcome_text = locale_manager.get("welcome")
    assert welcome_text != "welcome"  # 应该返回翻译，而不是键名
    assert isinstance(welcome_text, str)
    assert len(welcome_text) > 0
    
def test_locale_manager_format():
    """测试LocaleManager格式化功能"""
    locale_manager = LocaleManager("resources/locales", "zh")
    
    # 测试格式化功能
    version = "1.0.0"
    formatted = locale_manager.format("output.footer", {"version": version})
    assert version in formatted
    
    # 测试format_text方法
    formatted2 = locale_manager.format_text("output.footer", version=version)
    assert version in formatted2
    assert formatted == formatted2  # 两个方法应该返回相同结果
    
def test_safe_functions():
    """测试safe_get_text和safe_format_text函数"""
    # 测试safe_get_text
    text = safe_get_text("welcome")
    assert text != "welcome"  # 应该返回翻译
    assert isinstance(text, str)
    
    # 测试safe_format_text
    version = "1.0.0"
    formatted = safe_format_text("output.footer", version=version)
    assert version in formatted
    
def test_language_switching():
    """测试语言切换功能"""
    locale_manager = LocaleManager("resources/locales", "zh")
    
    # 保存中文文本
    zh_welcome = locale_manager.get("welcome")
    
    # 切换到英文
    locale_manager.set_language("en")
    en_welcome = locale_manager.get("welcome")
    
    # 文本应该不同
    assert zh_welcome != en_welcome
    
    # 直接使用Logloom API获取相同文本，应该匹配
    set_language("en")
    logloom_en = get_text("welcome")
    assert en_welcome == logloom_en
    
    # 恢复中文
    set_language("zh")
    
def test_logger_integration():
    """测试LocaleManager与logger的集成"""
    # 获取logger
    logger = get_logger("test_integration")
    
    # 使用LocaleManager获取文本
    locale_manager = LocaleManager("resources/locales", "zh")
    welcome_text = locale_manager.get("welcome")
    
    # 记录一条包含本地化文本的日志
    logger.info(welcome_text)
    
    # 这个测试主要是确认不会有异常抛出，表明集成正常
    assert True
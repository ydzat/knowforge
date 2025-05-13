#!/usr/bin/env python
'''
Author: @ydzat
Date: 2025-04-28 17:51:57
LastEditors: @ydzat
LastEditTime: 2025-05-13 13:30:00
Description: 
'''

"""
KnowForge CLI Entry File
Usage: python gen_notes.py [command] [options]
"""
import typer
from src.cli.cli_main import cli
from src import __version__
from src.utils.logger import setup_logger
from src.utils.locale_manager import LocaleManager
from src.utils.config_loader import ConfigLoader
from logloom import set_language, get_text, format_text, initialize

# 初始化Logloom
try:
    initialize("./resources/config/logloom_config.yaml")
except Exception as e:
    # 如果配置加载失败，将在setup_logger中处理
    pass

logger = setup_logger()

def main():
    """
    KnowForge - AI-driven study note generator
    """
    # 尝试加载配置和语言资源
    try:
        config = ConfigLoader("resources/config/config.yaml")
        language = config.get("system.language", "zh")
        locale = LocaleManager(f"resources/locales/{language}.yaml", language)
        
        # 设置Logloom语言
        set_language(language)
        
        # 使用Logloom的国际化功能
        startup_message = format_text("system.start") + f" v{__version__}"
    except Exception as e:
        # 如果配置加载失败，使用硬编码默认值
        startup_message = f"KnowForge v{__version__} 启动"
        logger.error("配置加载失败: {}", str(e))
    
    logger.info(startup_message)
    logger.info(get_text("welcome"))
    #typer.run(cli)
    cli()

if __name__ == "__main__":
    main()
"""
命令行接口主模块，处理命令行参数并调用对应功能
"""
import os
import typer
from typing import List, Optional
from src import __version__
from src.utils.logger import setup_logger
from src.utils.locale_manager import LocaleManager
from src.utils.config_loader import ConfigLoader
from src.utils.exceptions import NoteGenError
from src.note_generator.processor import Processor

# 创建CLI应用
cli = typer.Typer(help="KnowForge - AI-powered study note generator")
logger = setup_logger()

# 尝试加载配置和语言资源
try:
    config = ConfigLoader("resources/config/config.yaml")
    language = config.get("system.language", "zh")
    locale = LocaleManager(f"resources/locales/{language}.yaml", language)
except Exception as e:
    # 如果配置加载失败，使用硬编码默认值
    locale = None
    logger.error("Failed to load configuration: {0}".format(str(e)))


@cli.callback()
def callback():
    """
    KnowForge - AI-powered study note generator
    """
    if locale:
        typer.echo(locale.get("cli.welcome"))
        typer.echo(locale.get("cli.version").format(version=__version__))
    else:
        typer.echo("Welcome to KnowForge - AI-powered study note generator")
        typer.echo(f"Version: {__version__}")


@cli.command()
def generate(
    input_dir: str = typer.Option("input/", help="Input directory"),
    output_dir: str = typer.Option("output/", help="Output directory"),
    config_path: str = typer.Option("resources/config/config.yaml", help="Config file path"),
    formats: str = typer.Option("markdown", help="Output formats, comma separated")
):
    """
    Generate notes
    """
    if locale:
        typer.echo(locale.get("system.processing"))
    else:
        typer.echo("Processing...")
    
    # 添加输入目录提示
    if locale:
        typer.echo(locale.get("system.reading_input").format(input_dir=input_dir))
    else:
        typer.echo(f"Reading input from {input_dir}")
    
    # 添加输出格式提示
    if locale:
        typer.echo(locale.get("system.generating_format").format(formats=formats))
    else:
        typer.echo(f"Generating note formats: {formats}")
    
    try:
        # 初始化处理器
        processor = Processor(input_dir, output_dir, config_path)
        
        # 解析输出格式
        output_formats = [fmt.strip().lower() for fmt in formats.split(',')]
        valid_formats = ["markdown", "ipynb", "pdf"]
        
        # 验证输出格式
        invalid_formats = [fmt for fmt in output_formats if fmt not in valid_formats]
        if invalid_formats:
            if locale:
                typer.echo(locale.get("cli.warning_unsupported_format").format(formats=', '.join(invalid_formats)))
            else:
                typer.echo(f"Warning: Unsupported output formats: {', '.join(invalid_formats)}")
            output_formats = [fmt for fmt in output_formats if fmt in valid_formats]
            
        if not output_formats:
            if locale:
                typer.echo(locale.get("cli.error_no_valid_format"))
            else:
                typer.echo("Error: No valid output formats provided, using default: markdown")
            output_formats = ["markdown"]
        
        # 运行处理流程
        output_paths = processor.run_full_pipeline(output_formats)
        
        if not output_paths:
            if locale:
                typer.echo(locale.get("system.no_input_files"))
            else:
                typer.echo("No valid input files found")
        else:
            # 显示输出文件路径
            if locale:
                typer.echo("\n" + locale.get("system.generated_notes"))
            else:
                typer.echo("\nGenerated note files:")
                
            for fmt, path in output_paths.items():
                if locale:
                    typer.echo(locale.get("system.format_output").format(format=fmt.upper(), path=path))
                else:
                    typer.echo(f"- {fmt.upper()}: {path}")
        
        # 不管是否生成了文件，都显示完成消息
        if locale:
            typer.echo(locale.get("system.completed"))
        else:
            typer.echo("Processing completed")
    except NoteGenError as e:
        if locale:
            logger.error(locale.get("system.error_prefix").format(error=str(e)))
            typer.echo(locale.get("system.error_occurred").format(error=str(e)))
        else:
            logger.error("Known error: {0}".format(str(e)))
            typer.echo(f"Error occurred: {str(e)}")
    except Exception as e:
        if locale:
            logger.exception(locale.get("system.unexpected_error"))
            typer.echo(locale.get("system.unexpected_error_occurred").format(error=str(e)))
        else:
            logger.exception("Unexpected error: {0}".format(str(e)))
            typer.echo(f"An unexpected error occurred: {str(e)}")


@cli.command()
def process_file(
    file_path: str = typer.Argument(..., help="File path to process"),
    output_format: str = typer.Option("markdown", help="Output format (markdown/ipynb/pdf)"),
    output_dir: str = typer.Option("output/", help="Output directory"),
    config_path: str = typer.Option("resources/config/config.yaml", help="Config file path")
):
    """
    Process a single file
    """
    if locale:
        typer.echo(locale.get("system.processing"))
    else:
        typer.echo("Processing...")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            if locale:
                typer.echo(locale.get("cli.file_not_exist").format(file_path=file_path))
            else:
                typer.echo(f"Error: File does not exist: {file_path}")
            return
        
        # 检查输出格式是否有效
        if output_format.lower() not in ["markdown", "ipynb", "pdf"]:
            if locale:
                typer.echo(locale.get("cli.unsupported_output_format").format(format=output_format))
                typer.echo(locale.get("cli.supported_formats"))
            else:
                typer.echo(f"Error: Unsupported output format: {output_format}")
                typer.echo("Supported formats: markdown, ipynb, pdf")
            return
        
        # 初始化处理器
        processor = Processor(
            input_dir=os.path.dirname(file_path), 
            output_dir=output_dir, 
            config_path=config_path
        )
        
        # 处理文件
        output_path = processor.generate_note(file_path, output_format.lower())
        
        # 显示结果
        if locale:
            typer.echo("\n" + locale.get("cli.note_generated").format(path=output_path))
        else:
            typer.echo(f"\nNote generated: {output_path}")
        
        if locale:
            typer.echo(locale.get("system.completed"))
        else:
            typer.echo("Processing completed")
    except NoteGenError as e:
        if locale:
            logger.error(locale.get("system.error_prefix").format(error=str(e)))
            typer.echo(locale.get("system.error_occurred"))
        else:
            logger.error("Known error: {0}".format(str(e)))
            typer.echo(f"Error occurred: {str(e)}")
    except Exception as e:
        if locale:
            logger.exception(locale.get("system.unexpected_error"))
            typer.echo(locale.get("system.unexpected_error"))
        else:
            logger.exception("Unexpected error: {0}".format(str(e)))
            typer.echo(f"An unexpected error occurred: {str(e)}")


@cli.command()
def version():
    """
    Show version information
    """
    if locale:
        typer.echo(locale.get("cli.version_info").format(version=__version__))
    else:
        typer.echo(f"KnowForge v{__version__}")
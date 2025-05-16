#!/usr/bin/env python
"""
资源下载脚本

此脚本用于下载KnowForge需要的外部资源，供离线使用。
包括Bootstrap, MathJax, highlight.js等
"""
import os
import sys
import argparse
import yaml
import requests
from tqdm import tqdm

# 添加项目根目录到路径，以便导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 默认资源列表
DEFAULT_RESOURCES = {
    "bootstrap_css": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
    "bootstrap_js": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js",
    "mathjax_js": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
    "highlight_js": "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js",
}

# 默认代码高亮主题
DEFAULT_CODE_THEMES = [
    "default",
    "github", 
    "a11y-dark",
    "androidstudio", 
    "monokai", 
    "atom-one-dark", 
    "atom-one-light",
    "vs2015",
    "xcode",
    "solarized-dark",
    "solarized-light",
]

def download_file(url, destination):
    """下载文件到指定位置，显示进度条"""
    print(f"下载 {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求出错，抛出异常
        
        total_size = int(response.headers.get('content-length', 0))
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        print(f"已保存到 {destination}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def load_config():
    """加载配置文件，获取资源URL"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "resources", "config", "output_config.yaml"
    )
    
    resources = DEFAULT_RESOURCES.copy()
    code_themes = DEFAULT_CODE_THEMES.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config and 'html' in config and 'resources' in config['html']:
                for key in resources:
                    if key in config['html']['resources']:
                        resources[key] = config['html']['resources'][key]
                
            print(f"已从配置文件加载资源URL")
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
    else:
        print(f"配置文件 {config_path} 不存在，使用默认URL")
    
    return resources, code_themes

def download_resources(output_dir=None, code_themes=None):
    """下载所有资源到指定目录"""
    resources, default_themes = load_config()
    
    if code_themes is None:
        code_themes = default_themes
    
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources", "assets"
        )
    
    print(f"资源将被下载到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载基础资源
    results = []
    for name, url in resources.items():
        file_name = os.path.basename(url.split('?')[0])  # 移除URL中的查询参数
        if name == "bootstrap_css":
            file_name = "bootstrap.min.css"
        elif name == "bootstrap_js":
            file_name = "bootstrap.bundle.min.js"
        elif name == "mathjax_js":
            file_name = "tex-mml-chtml.js"
        elif name == "highlight_js":
            file_name = "highlight.min.js"
        
        destination = os.path.join(output_dir, file_name)
        success = download_file(url, destination)
        results.append((name, success))
    
    # 下载代码高亮主题
    os.makedirs(os.path.join(output_dir, "styles"), exist_ok=True)
    
    for theme in code_themes:
        theme_url = f"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/{theme}.min.css"
        destination = os.path.join(output_dir, "styles", f"{theme}.min.css")
        success = download_file(theme_url, destination)
        results.append((f"highlight_theme_{theme}", success))
    
    # 输出结果摘要
    print("\n下载结果摘要:")
    success_count = 0
    for name, success in results:
        status = "成功" if success else "失败"
        print(f"  - {name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {len(results)} 个文件, {success_count} 成功, {len(results) - success_count} 失败")
    
    # 生成使用说明
    print("\n使用说明:")
    print("在output_config.yaml中设置:")
    print("html:")
    print("  resources:")
    print("    use_cdn: false")
    print("    local_resource_dir: resources/assets")

def main():
    parser = argparse.ArgumentParser(description="KnowForge资源下载工具")
    parser.add_argument("--output", "-o", type=str, help="输出目录路径")
    parser.add_argument("--themes", "-t", type=str, help="要下载的代码高亮主题列表，以逗号分隔")
    parser.add_argument("--list-themes", action="store_true", help="列出所有可用的代码高亮主题")
    
    args = parser.parse_args()
    
    if args.list_themes:
        print("可用的代码高亮主题:")
        for theme in DEFAULT_CODE_THEMES:
            print(f"  - {theme}")
        return
    
    output_dir = args.output
    
    themes = None
    if args.themes:
        themes = [t.strip() for t in args.themes.split(',')]
    
    download_resources(output_dir, themes)

if __name__ == "__main__":
    main()

'''
Author: @ydzat
Date: 2025-04-29 00:52:26
LastEditors: @ydzat
LastEditTime: 2025-04-29 01:09:27
Description: 
'''
import subprocess
import os
import time  # 新增

# 自动化 LLM 集成流程验收脚本
# 运行主流程并检查输出和日志，确保 LLM 拆分等功能在真实环境下可用

def main():
    # 设置超时时间（秒）
    timeout = 120  # 2分钟超时，根据需要调整
    
    print(f"开始执行KnowForge主流程，超时时间: {timeout}秒...")
    start_time = time.time()
    
    try:
        # 运行主流程，添加超时限制
        result = subprocess.run(
                  ["python", "gen_notes.py", "generate", "--input-dir", "input/", "--output-dir", "output/"],
                  capture_output=True, text=True, timeout=timeout
              )
        print("=== CLI输出 ===")
        print(result.stdout)
        print(result.stderr)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        print(f"执行完成，耗时: {execution_time:.2f}秒")

        # 检查输出文件
        notes_path = os.path.join("output", "markdown", "notes.md")
        if os.path.exists(notes_path):
            print(f"[OK] 输出文件存在: {notes_path}")
        else:
            print(f"[FAIL] 未生成输出文件: {notes_path}")

        # 检查日志内容
        log_path = os.path.join("output", "logs", "note_gen.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            if "LLM成功拆分文本" in log_content or "LLM成功拆分文本" in result.stdout:
                print("[OK] 日志中包含 LLM 拆分成功 相关信息")
            else:
                print("[WARN] 日志中未检测到 LLM 拆分成功 相关信息，请人工检查")
        else:
            print(f"[FAIL] 未生成日志文件: {log_path}")

        # 检查输出内容片段
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                notes_content = f.read()
            if len(notes_content.strip()) > 0:
                print("[OK] 输出笔记内容非空")
            else:
                print("[FAIL] 输出笔记内容为空")
                
    except subprocess.TimeoutExpired:
        print(f"[ERROR] 命令执行超时 (>{timeout}秒)，LLM调用可能出现延迟或故障")
        print("请检查网络连接或LLM服务状态，或增加脚本超时时间")
    except Exception as e:
        print(f"[ERROR] 执行过程中出现异常: {str(e)}")

if __name__ == "__main__":
    main()
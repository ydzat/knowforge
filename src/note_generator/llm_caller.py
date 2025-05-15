"""
LLM调用模块，负责与大语言模型API交互
"""
import os
import time
import json
import requests
from typing import Dict, Any, Optional, List, Union
import logging
from src.utils.logger import get_module_logger
from src.utils.exceptions import NoteGenError

logger = get_module_logger("llm_caller")

class LLMError(NoteGenError):
    """LLM调用过程中的异常"""
    pass

class LLMCaller:
    """LLM调用类，负责与不同LLM API交互"""
    
    def __init__(self, model: str = "deepseek-chat", api_key: str = None, base_url: str = None):
        """
        初始化LLM调用器
        
        Args:
            model: 模型名称，如'deepseek-chat'
            api_key: API密钥，如果为None则尝试从环境变量获取
            base_url: API基础URL，默认使用DeepSeek的API
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or "https://api.deepseek.com"
        
        # 确定提供商
        if "deepseek" in self.model.lower() or "deepseek" in self.base_url.lower():
            self.provider = "deepseek"
        elif "openai" in self.model.lower() or "openai" in self.base_url.lower():
            self.provider = "openai"
        else:
            self.provider = "unknown"
        
        # API请求默认参数
        self.default_params = {
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.95
        }
        
        # 重试设置
        self.max_retries = 3
        self.retry_delay = 2
        
        logger.info(f"LLM调用器初始化完成，提供商: {self.provider}, 模型: {self.model}")
    
    def call_model(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """
        调用LLM模型
        
        Args:
            prompt: 提示文本
            params: 调用参数，如temperature等
            
        Returns:
            模型生成的文本
            
        Raises:
            LLMError: 调用过程中的错误
        """
        if not self.api_key:
            raise LLMError("未提供API密钥")
        
        # 合并默认参数与自定义参数
        call_params = self.default_params.copy()
        if params:
            call_params.update(params)
        
        # 根据提供商选择不同的调用方法
        if self.provider == "deepseek":
            return self._call_deepseek(prompt, call_params)
        elif self.provider == "openai":
            return self._call_openai(prompt, call_params)
        else:
            raise LLMError(f"不支持的LLM提供商: {self.provider}")
    
    def _call_deepseek(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        调用DeepSeek API
        
        Args:
            prompt: 提示文本
            params: 调用参数
            
        Returns:
            生成的文本
            
        Raises:
            LLMError: 调用过程中的错误
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": params.get("temperature", 0.3),
            "max_tokens": params.get("max_tokens", 2000),
            "top_p": params.get("top_p", 0.95)
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"调用DeepSeek API, 模型: {self.model}, 尝试次数: {attempt + 1}")
                start_time = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"API调用成功，耗时: {duration:.2f}秒")
                    
                    # 提取生成的文本
                    if 'choices' in result and len(result['choices']) > 0:
                        message = result['choices'][0].get('message', {})
                        text = message.get('content', '').strip()
                        return text
                    else:
                        raise LLMError("API响应格式不正确，缺少'choices'字段")
                else:
                    error_msg = f"API调用失败，状态码: {response.status_code}, 响应: {response.text}"
                    logger.error(error_msg)
                    
                    # 根据状态码决定是否重试
                    if response.status_code in [429, 500, 502, 503, 504]:
                        if attempt < self.max_retries - 1:
                            logger.info(f"将在 {self.retry_delay} 秒后重试")
                            time.sleep(self.retry_delay)
                            self.retry_delay *= 2  # 指数退避
                            continue
                    
                    raise LLMError(error_msg)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"API请求异常: {str(e)}"
                logger.error(error_msg)
                
                if attempt < self.max_retries - 1:
                    logger.info(f"将在 {self.retry_delay} 秒后重试")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # 指数退避
                    continue
                
                raise LLMError(error_msg)
        
        raise LLMError(f"在 {self.max_retries} 次尝试后调用API失败")
    
    def _call_openai(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        调用OpenAI API
        
        Args:
            prompt: 提示文本
            params: 调用参数
            
        Returns:
            生成的文本
            
        Raises:
            LLMError: 调用过程中的错误
        """
        try:
            import openai
            openai.api_key = self.api_key
            
            # 如果提供了自定义base_url，则使用它
            if "openai.com" not in self.base_url:
                openai.base_url = self.base_url
            
            logger.info(f"调用OpenAI API, 模型: {self.model}")
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=params.get("temperature", 0.3),
                max_tokens=params.get("max_tokens", 2000),
                top_p=params.get("top_p", 0.95)
            )
            
            duration = time.time() - start_time
            logger.info(f"API调用成功，耗时: {duration:.2f}秒")
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            raise LLMError("使用OpenAI API需要安装openai包: pip install openai")
        
        except Exception as e:
            raise LLMError(f"OpenAI API调用失败: {str(e)}")

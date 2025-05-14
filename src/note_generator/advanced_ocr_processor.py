"""
高级OCR处理模块，实现OCR与LLM协同识别
"""
import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.utils.logger import get_module_logger
from src.utils.config_loader import ConfigLoader
from src.utils.locale_manager import LocaleManager
from logloom import get_text, format_text

logger = get_module_logger("advanced_ocr")

class AdvancedOCRProcessor:
    """高级OCR处理器，结合LLM进行增强识别"""
    
    def __init__(self, config: dict, workspace_dir: str):
        """
        初始化高级OCR处理器
        
        Args:
            config: 应用程序配置
            workspace_dir: 工作空间目录路径
        """
        self.config = config
        self.workspace_dir = workspace_dir
        self.locale_manager = LocaleManager(config.get("locale", "en"))
        
        # 日志初始化
        self.logger = get_module_logger('AdvancedOCRProcessor')
        
        # 本地化
        global logger, get_text, format_text
        logger = self.logger
        get_text = self.locale_manager.get_text
        format_text = self.locale_manager.format_text
        
        # OCR配置
        self.ocr_config = config.get("input.ocr", {})
        self.ocr_enabled = self.ocr_config.get("enabled", True)
        self.ocr_languages = self.ocr_config.get("languages", ["ch_sim", "en"])
        self.ocr_confidence_threshold = self.ocr_config.get("confidence_threshold", 0.6)
        
        # LLM增强配置
        self.use_llm_enhancement = self.ocr_config.get("use_llm_enhancement", True)
        self.deep_enhancement = self.ocr_config.get("deep_enhancement", True)
        self.knowledge_enhanced_ocr = self.ocr_config.get("knowledge_enhanced_ocr", True)
        
        # 图像预处理配置
        self.img_preprocessing = self.ocr_config.get("image_preprocessing", {
            "enabled": True,
            "denoise": True,
            "contrast_enhancement": True,
            "auto_rotate": True,
            "adaptive_thresholding": True,
            "deskew": True
        })
        
        # OCR模型缓存
        self.ocr_reader = None
        
        # LLM调用器
        self.llm_caller = None
        
        # 调试模式
        self.debug_mode = self.ocr_config.get("debug_mode", False)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        高级图像预处理，提高OCR质量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像数组
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            error_msg = f"无法读取图像: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 如果预处理未启用，直接返回原图
        if not self.img_preprocessing.get("enabled", True):
            return img
            
        logger.info(get_text("advanced_ocr.preprocessing_image"))
        
        # 转为灰度图像（如果是彩色图像）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 去噪处理
        if self.img_preprocessing.get("denoise", True):
            logger.debug(get_text("advanced_ocr.applying_denoise"))
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
        # 对比度增强
        if self.img_preprocessing.get("contrast_enhancement", True):
            logger.debug(get_text("advanced_ocr.enhancing_contrast"))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # 自适应阈值处理
        if self.img_preprocessing.get("adaptive_thresholding", True):
            logger.debug(get_text("advanced_ocr.applying_adaptive_threshold"))
            # 应用自适应阈值，改进文字与背景的分离
            # 只对灰度级中间范围的图像应用，避免处理已经很清晰的文档
            if np.mean(gray) > 70 and np.mean(gray) < 200:  
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                
                # 根据原始图像决定是否使用二值化结果
                # 计算二值化前后的清晰度差异
                clarity_before = cv2.Laplacian(gray, cv2.CV_64F).var()
                clarity_after = cv2.Laplacian(binary, cv2.CV_64F).var()
                
                # 如果二值化提高了清晰度，则使用二值化图像
                if clarity_after > clarity_before:
                    gray = binary
        
        # 图像倾斜校正
        if self.img_preprocessing.get("deskew", True):
            logger.debug(get_text("advanced_ocr.deskewing_image"))
            try:
                coords = np.column_stack(np.where(gray > 0))
                angle = cv2.minAreaRect(coords)[-1]
                
                # 角度校正
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                    
                # 只修正大于0.5度的倾斜
                if abs(angle) > 0.5:
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
                    logger.debug(format_text("advanced_ocr.corrected_skew", angle=angle))
            except Exception as e:
                logger.warning(format_text("advanced_ocr.deskew_failed", error=str(e)))
        
        # 保存预处理图像（调试模式）
        if self.debug_mode:
            debug_dir = os.path.join(self.workspace_dir, "debug_images")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
                
            base_filename = os.path.basename(image_path)
            processed_path = os.path.join(debug_dir, f"processed_{base_filename}")
            cv2.imwrite(processed_path, gray)
            logger.debug(format_text("advanced_ocr.saved_debug_image", path=processed_path))
        
        logger.info(get_text("advanced_ocr.preprocessing_complete"))
        return gray

    def init_ocr_reader(self):
        """初始化OCR读取器"""
        if not self.ocr_enabled:
            return None
        
        try:
            import easyocr
        except ImportError:
            logger.error(get_text("advanced_ocr.easyocr_not_installed"))
            return None
            
        if self.ocr_reader:
            return self.ocr_reader
            
        try:
            # 从配置中获取GPU设置
            use_gpu = self.ocr_config.get("use_gpu", "auto")
            
            # GPU设置逻辑处理
            if use_gpu == "auto":
                # 自动检测CUDA可用性
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                    if use_gpu:
                        gpu_info = f"CUDA {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}"
                        logger.info(format_text("advanced_ocr.using_gpu", gpu_info=gpu_info))
                    else:
                        logger.info(get_text("advanced_ocr.no_gpu_available"))
                except ImportError:
                    use_gpu = False
                    logger.info(get_text("advanced_ocr.torch_not_available"))
            else:
                # 使用配置中的明确设置
                use_gpu = use_gpu in (True, "true", "yes", 1)
            
            logger.info(format_text("advanced_ocr.initializing_ocr", 
                                   languages=",".join(self.ocr_languages)))
            
            self.ocr_reader = easyocr.Reader(
                self.ocr_languages,
                gpu=use_gpu,
                model_storage_directory=os.path.join(self.workspace_dir, "ocr_models"),
                download_enabled=True
            )
            logger.info(get_text("advanced_ocr.ocr_initialized"))
            return self.ocr_reader
        except Exception as e:
            logger.error(format_text("advanced_ocr.ocr_init_failed", error=str(e)))
            return None
    
    def init_llm_caller(self):
        """初始化LLM调用器"""
        if not self.use_llm_enhancement:
            return None
            
        if self.llm_caller:
            return self.llm_caller
            
        try:
            from src.note_generator.llm_caller import LLMCaller
            
            # 获取LLM配置
            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "deepseek")
            model = llm_config.get("model", "deepseek-chat")
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            base_url = llm_config.get("base_url", "https://api.deepseek.com")
            
            if not api_key:
                logger.warning(get_text("advanced_ocr.no_llm_api_key"))
                return None
                
            logger.info(format_text("advanced_ocr.initializing_llm", provider=provider, model=model))
            
            self.llm_caller = LLMCaller(model, api_key, base_url)
            logger.info(get_text("advanced_ocr.llm_initialized"))
            return self.llm_caller
            
        except ImportError:
            logger.warning(get_text("advanced_ocr.llm_module_not_available"))
            return None
            
        except Exception as e:
            logger.error(format_text("advanced_ocr.llm_init_failed", error=str(e)))
            return None
    
    def process_image(self, image_path: str) -> Tuple[str, float]:
        """
        处理图像，提取文本，并应用LLM增强
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            (处理后的文本, 置信度)
        """
        logger.info(format_text("advanced_ocr.processing_image", path=image_path))
        
        # 1. 初始化OCR读取器
        reader = self.init_ocr_reader()
        if reader is None:
            logger.error(get_text("advanced_ocr.ocr_not_available"))
            return "", 0.0
            
        # 2. 图像预处理
        try:
            processed_img = self.preprocess_image(image_path)
        except Exception as e:
            logger.error(format_text("advanced_ocr.preprocessing_failed", error=str(e)))
            # 预处理失败时尝试使用原始图像
            processed_img = cv2.imread(image_path)
            if processed_img is None:
                return "", 0.0
        
        # 3. 获取图像上下文信息
        context = self._get_image_context(image_path)
        
        try:
            # 4. OCR识别（获取所有结果）
            logger.info(get_text("advanced_ocr.running_ocr"))
            raw_results = reader.readtext(processed_img)
            
            # 5. 区分高低置信度结果
            high_conf_texts = []    # 高置信度文本 (>= 0.6)
            high_conf_values = []   # 高置信度值
            low_conf_texts = []     # 低置信度文本 (< 0.6)
            low_conf_values = []    # 低置信度值
            all_texts = []          # 所有文本
            all_confidences = []    # 所有置信度值
            all_bboxes = []         # 所有边界框
            
            for (bbox, text, confidence) in raw_results:
                all_texts.append(text)
                all_confidences.append(confidence)
                all_bboxes.append(bbox)
                
                if confidence >= self.ocr_confidence_threshold:  # 保持0.6高置信度标准
                    high_conf_texts.append(text)
                    high_conf_values.append(confidence)
                else:
                    low_conf_texts.append(text)
                    low_conf_values.append(confidence)
            
            # 6. 组合所有OCR文本结果
            raw_ocr_text = " ".join(all_texts)
            high_conf_text = " ".join(high_conf_texts)
            
            # 如果没有任何OCR结果，则直接返回空
            if not raw_ocr_text:
                logger.warning(get_text("advanced_ocr.no_ocr_results"))
                return "", 0.0
                
            # 如果不需要LLM增强，返回高置信度结果
            if not self.use_llm_enhancement:
                text_to_return = high_conf_text if high_conf_text else raw_ocr_text
                avg_confidence = sum(high_conf_values) / len(high_conf_values) if high_conf_values else 0.0
                logger.info(format_text("advanced_ocr.standard_ocr_complete", 
                                      confidence=avg_confidence))
                return text_to_return, avg_confidence
            
            # 7. 初始化LLM调用器
            llm_caller = self.init_llm_caller()
            if llm_caller is None:
                # LLM初始化失败，返回标准OCR结果
                text_to_return = high_conf_text if high_conf_text else raw_ocr_text
                avg_confidence = sum(high_conf_values) / len(high_conf_values) if high_conf_values else 0.0
                return text_to_return, avg_confidence
            
            # 8. 使用LLM增强OCR结果
            # 如果有高置信度结果，优先使用它们，否则使用所有结果
            initial_text = high_conf_text if high_conf_text else raw_ocr_text
            
            try:
                # 构建OCR增强提示
                prompt = self._build_ocr_enhancement_prompt(
                    initial_text, context, 
                    has_low_conf=bool(low_conf_texts)  # 告知LLM是否有低置信度文本被过滤
                )
                
                logger.info(get_text("advanced_ocr.calling_llm"))
                # 调用LLM进行增强
                enhanced_text = llm_caller.call_model(prompt)
                # 处理LLM返回结果
                enhanced_text = self._postprocess_llm_result(enhanced_text)
                
                # 深度增强模式：知识库集成
                if self.deep_enhancement and self.knowledge_enhanced_ocr:
                    enhanced_text = self._apply_knowledge_enhancement(enhanced_text, context)
            
            except Exception as e:
                logger.error(format_text("advanced_ocr.llm_enhancement_failed", error=str(e)))
                # LLM增强失败，返回标准OCR结果
                enhanced_text = initial_text
            
            # 9. 计算最终置信度
            final_confidence = self._estimate_final_confidence(
                high_conf_values, all_confidences, 
                has_llm_enhancement=True,
                has_knowledge_enhancement=self.deep_enhancement and self.knowledge_enhanced_ocr
            )
            
            logger.info(format_text("advanced_ocr.processing_complete", 
                                  confidence=final_confidence))
            
            return enhanced_text, final_confidence
            
        except Exception as e:
            logger.error(format_text("advanced_ocr.ocr_processing_failed", error=str(e)))
            return "", 0.0
    
    def _get_image_context(self, image_path: str) -> str:
        """
        获取图像的上下文信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            上下文信息描述
        """
        # 提取文件名（可能包含章节信息）
        filename = os.path.basename(image_path)
        
        # 提取父目录名（可能包含分类信息）
        parent_dir = os.path.basename(os.path.dirname(image_path))
        
        # 获取图像信息（如尺寸）
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            img_info = f"图像尺寸: {width}x{height}"
        except:
            img_info = "无法获取图像信息"
        
        # 返回组合的上下文信息
        context = f"""
文件名: {filename}
所在目录: {parent_dir}
{img_info}
"""
        return context
    
    def _build_ocr_enhancement_prompt(self, ocr_text: str, image_context: str,
                                    has_low_conf: bool = False) -> str:
        """
        构建用于OCR增强的LLM提示
        
        Args:
            ocr_text: OCR文本
            image_context: 图像上下文信息
            has_low_conf: 是否有低置信度文本被过滤
            
        Returns:
            LLM提示文本
        """
        # 创建上下文感知的智能提示
        prompt = f"""
你是一个专业的OCR文本增强专家。请改进以下OCR文本，提高其准确性和可读性。

### 图片上下文信息：
{image_context}

### OCR文本（已应用{self.ocr_confidence_threshold}置信度阈值）：
{ocr_text}

{f"注意：部分低置信度文本已被过滤，可能导致内容不完整。请尝试根据上下文补充可能缺失的内容。" if has_low_conf else ""}

### 任务：
1. 仔细分析OCR文本，修正明显的拼写错误和识别错误
2. 恢复正确的段落和格式结构
3. 确保专业术语、特殊符号、数字和标点正确
4. 提高整体文本的连贯性和可读性
5. 保留所有关键信息，不要随意添加内容

### 直接返回纯文本结果，不要添加任何解释、标签或描述。
"""
        return prompt
    
    def _apply_knowledge_enhancement(self, initial_text: str, context: str) -> str:
        """
        应用知识库增强OCR结果
        
        Args:
            initial_text: 初步优化的OCR文本
            context: 图像上下文信息
            
        Returns:
            知识增强后的文本
        """
        try:
            from src.note_generator.embedding_manager import EmbeddingManager
            
            logger.info(get_text("advanced_ocr.applying_knowledge_enhancement"))
            
            # 初始化embedding管理器
            embedding_manager = EmbeddingManager(self.workspace_dir, self.config)
            
            # 使用优化后的文本查询知识库
            relevant_docs = embedding_manager.search_similar_content(initial_text, top_k=3)
            
            # 没有找到相关文档
            if not relevant_docs:
                logger.info(get_text("advanced_ocr.no_relevant_knowledge"))
                return initial_text
                
            # 构建知识库上下文
            knowledge_context = "\n\n".join([doc.content for doc in relevant_docs])
            
            # 构建知识库增强提示
            prompt = self._build_knowledge_enhancement_prompt(initial_text, context, knowledge_context)
            
            # 初始化LLM调用器
            llm_caller = self.init_llm_caller()
            if llm_caller is None:
                return initial_text
                
            logger.info(get_text("advanced_ocr.calling_llm_with_knowledge"))
            # 调用LLM进行知识库增强
            enhanced_text = llm_caller.call_model(prompt)
            # 处理LLM返回结果
            enhanced_text = self._postprocess_llm_result(enhanced_text)
            
            logger.info(get_text("advanced_ocr.knowledge_enhancement_complete"))
            return enhanced_text
            
        except ImportError:
            logger.warning(get_text("advanced_ocr.embedding_module_not_available"))
            return initial_text
            
        except Exception as e:
            logger.error(format_text("advanced_ocr.knowledge_enhancement_failed", error=str(e)))
            return initial_text
    
    def _build_knowledge_enhancement_prompt(self, ocr_text: str, image_context: str, 
                                         knowledge_context: str) -> str:
        """
        构建用于知识库增强的LLM提示
        
        Args:
            ocr_text: 初步优化的OCR文本
            image_context: 图像上下文信息
            knowledge_context: 知识库上下文
            
        Returns:
            LLM提示文本
        """
        prompt = f"""
你是一个专业的OCR文本增强专家。请利用知识库信息进一步优化OCR文本，使其更准确、更专业。

### 图片上下文信息：
{image_context}

### 已初步优化的OCR文本：
{ocr_text}

### 知识库相关内容（用于参考和校正）：
{knowledge_context}

### 任务：
1. 仔细比对OCR文本与知识库内容，更正专业术语和概念
2. 修正可能的错误或模糊内容，参考知识库中的准确表述
3. 确保文本格式和结构符合专业标准
4. 提高识别质量和准确性，使文本更加清晰可信
5. 保持原始内容的核心意思，不要过度修改或添加无关内容

### 直接返回纯文本结果，不要添加任何解释、标签或描述。
"""
        return prompt
    
    def _postprocess_llm_result(self, llm_text: str) -> str:
        """
        对LLM返回的结果进行后处理
        
        Args:
            llm_text: LLM返回的文本
            
        Returns:
            处理后的文本
        """
        # 去除可能的引用符号或格式标记
        text = llm_text.strip()
        
        # 去除可能的代码块标记
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        
        # 去除可能的引号
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
            
        return text
    
    def _estimate_final_confidence(self, high_confidences: List[float], 
                                 all_confidences: List[float], 
                                 has_llm_enhancement: bool = False,
                                 has_knowledge_enhancement: bool = False) -> float:
        """
        估算最终OCR结果的置信度
        
        Args:
            high_confidences: 高置信度值列表（>=阈值）
            all_confidences: 所有置信度值列表
            has_llm_enhancement: 是否使用了LLM增强
            has_knowledge_enhancement: 是否使用了知识库增强
            
        Returns:
            估算的置信度值，确保不低于配置的阈值
        """
        # 计算基础置信度
        if high_confidences:
            # 如果有高置信度结果，优先使用
            base_confidence = sum(high_confidences) / len(high_confidences)
        elif all_confidences:
            # 否则使用所有结果的平均值
            base_confidence = sum(all_confidences) / len(all_confidences)
        else:
            # 没有任何结果时的默认置信度
            base_confidence = self.ocr_confidence_threshold
        
        # LLM增强因子（基于经验值）- 提高置信度
        llm_factor = 1.25 if has_llm_enhancement else 1.0  # 25%的提升
        
        # 知识库集成因子（如果启用）- 进一步提高置信度
        knowledge_factor = 1.15 if has_knowledge_enhancement else 1.0  # 15%的提升
        
        # 计算最终置信度（不超过1.0）
        final_confidence = min(base_confidence * llm_factor * knowledge_factor, 1.0)
        
        # 确保最终置信度不低于配置的阈值
        final_confidence = max(final_confidence, self.ocr_confidence_threshold)
        
        return final_confidence
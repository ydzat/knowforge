#!/usr/bin/env python3
"""
测试从PDF中提取图像
"""
import os
import sys
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

pdf_path = "/home/ydzat/workspace/knowforge/input/pdf/ml2_25-part01-intro.pdf"

def test_image_extraction():
    print(f"测试PDF图像提取: {pdf_path}")
    print(f"PyMuPDF版本: {fitz.__version__}")
    
    try:
        # 打开PDF
        doc = fitz.open(pdf_path)
        print(f"PDF页数: {len(doc)}")
        
        # 统计所有图像
        total_images = 0
        extracted_images = 0
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            # 获取页面上所有图像引用
            image_list = page.get_images(full=True)
            page_images = len(image_list)
            total_images += page_images
            
            print(f"第{page_index+1}页: 发现{page_images}个图像")
            
            # 尝试提取每个图像
            for img_index, img_info in enumerate(image_list):
                if not img_info:
                    continue
                    
                try:
                    xref = img_info[0]  # 图像引用号
                    if not xref:
                        continue
                    
                    print(f"  处理图像 {page_index+1}-{img_index+1} (xref={xref})")
                    
                    # 尝试提取图像
                    try:
                        # 方法1: 使用tobytes
                        pix = fitz.Pixmap(doc, xref)
                        if pix and pix.width > 0 and pix.height > 0:
                            # 转换为RGB格式（如果是CMYK）
                            if pix.n - pix.alpha > 3:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                                
                            # 转换为numpy数组
                            img_data = pix.tobytes()
                            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                            img_array = np.array(img)
                            
                            # 保存图像用于检查
                            output_dir = "workspace/debug_images"
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"image_{page_index+1}_{img_index+1}.png")
                            cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                            
                            print(f"    √ 成功提取图像: {img_array.shape}, 已保存到 {output_path}")
                            extracted_images += 1
                    except Exception as e:
                        print(f"    × 方法1提取失败: {str(e)}")
                        
                        # 方法2: 使用samples
                        try:
                            pix = fitz.Pixmap(doc, xref)
                            if pix and pix.width > 0 and pix.height > 0 and pix.samples:
                                mode = "RGBA" if pix.alpha else "RGB"
                                img = Image.frombuffer(mode, [pix.width, pix.height], pix.samples,
                                                   "raw", mode, 0, 1)
                                img_array = np.array(img)
                                
                                # 保存图像用于检查
                                output_dir = "workspace/debug_images"
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, f"image_{page_index+1}_{img_index+1}_method2.png")
                                cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                                
                                print(f"    √ 方法2成功提取图像: {img_array.shape}, 已保存到 {output_path}")
                                extracted_images += 1
                        except Exception as e:
                            print(f"    × 方法2提取失败: {str(e)}")
                            
                            # 方法3: 使用原始流数据
                            try:
                                # 获取原始图像数据
                                stream = doc.xref_stream(xref)
                                if stream:
                                    # 检测格式并保存
                                    output_dir = "workspace/debug_images"
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    if stream.startswith(b'\xff\xd8'):  # JPEG
                                        ext = ".jpg"
                                    elif stream.startswith(b'\x89PNG'):  # PNG
                                        ext = ".png"
                                    else:
                                        ext = ".bin"
                                    
                                    output_path = os.path.join(output_dir, f"image_{page_index+1}_{img_index+1}_raw{ext}")
                                    with open(output_path, "wb") as f:
                                        f.write(stream)
                                        
                                    print(f"    √ 方法3成功提取原始数据: {len(stream)}字节, 已保存到 {output_path}")
                                    extracted_images += 1
                            except Exception as e:
                                print(f"    × 方法3提取失败: {str(e)}")
                                
                except Exception as e:
                    print(f"  × 处理图像失败: {str(e)}")
        
        print("\n统计信息:")
        print(f"总计发现图像: {total_images}")
        print(f"成功提取图像: {extracted_images}")
        print(f"提取率: {extracted_images/total_images*100:.1f}% (如果为0则表示提取失败)")
        
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")

if __name__ == "__main__":
    test_image_extraction()

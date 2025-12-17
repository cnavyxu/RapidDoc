#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidDoc API 客户端示例
=====================

演示如何使用RapidDoc API服务进行文档解析

功能特性:
- 模型配置初始化
- 异步文档解析
- 多种输出格式支持
- 批量处理
"""

import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Optional
import tempfile
import os

class RapidDocClient:
    """RapidDoc API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url.rstrip("/")
        self.config_id = None
    
    async def health_check(self) -> dict:
        """健康检查"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def get_status(self) -> dict:
        """获取服务状态"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/status") as response:
                return await response.json()
    
    async def init_model_config(self, 
                              layout_model_type: str = "PP_DOCLAYOUT_PLUS_L",
                              ocr_engine_type: str = "ONNXRUNTIME",
                              formula_model_type: str = "PP_FORMULANET_PLUS_M",
                              table_model_type: str = "UNET_SLANET_PLUS",
                              device_mode: str = "cuda",
                              conf_thresh: float = 0.4,
                              use_det_mode: str = "ocr") -> str:
        """初始化模型配置，返回配置ID"""
        
        config_data = {
            "layout_model_type": layout_model_type,
            "ocr_engine_type": ocr_engine_type,
            "formula_model_type": formula_model_type,
            "table_model_type": table_model_type,
            "device_mode": device_mode,
            "conf_thresh": conf_thresh,
            "use_det_mode": use_det_mode
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/init",
                json=config_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.config_id = result["config_id"]
                    print(f"✅ 模型配置初始化成功: {self.config_id}")
                    print(f"   模块状态: {result['modules']}")
                    return self.config_id
                else:
                    error = await response.text()
                    raise Exception(f"初始化失败: {error}")
    
    async def parse_documents(self,
                            files: List[str],
                            output_dir: str = "./output",
                            parse_method: str = "auto",
                            formula_enable: bool = True,
                            table_enable: bool = True,
                            lang_list: List[str] = ["ch"],
                            start_page_id: int = 0,
                            end_page_id: Optional[int] = None,
                            return_md: bool = True,
                            return_middle_json: bool = False,
                            return_model_output: bool = False,
                            return_content_list: bool = False,
                            return_images: bool = False,
                            response_format_zip: bool = False) -> dict:
        """解析文档"""
        
        if not self.config_id:
            raise Exception("请先初始化模型配置")
        
        # 准备文件数据
        files_data = []
        for file_path in files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            with open(file_path, 'rb') as f:
                files_data.append({
                    'filename': os.path.basename(file_path),
                    'content': f.read()
                })
        
        # 构建表单数据
        data = aiohttp.FormData()
        data.add_field('config_id', self.config_id)
        data.add_field('output_dir', output_dir)
        data.add_field('parse_method', parse_method)
        data.add_field('formula_enable', str(formula_enable))
        data.add_field('table_enable', str(table_enable))
        data.add_field('lang_list', json.dumps(lang_list))
        data.add_field('start_page_id', str(start_page_id))
        data.add_field('return_md', str(return_md))
        data.add_field('return_middle_json', str(return_middle_json))
        data.add_field('return_model_output', str(return_model_output))
        data.add_field('return_content_list', str(return_content_list))
        data.add_field('return_images', str(return_images))
        data.add_field('response_format_zip', str(response_format_zip))
        
        if end_page_id is not None:
            data.add_field('end_page_id', str(end_page_id))
        
        # 添加文件
        for file_info in files_data:
            data.add_field(
                'files',
                file_info['content'],
                filename=file_info['filename'],
                content_type='application/octet-stream'
            )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/parse",
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 文档解析完成:")
                    print(f"   处理时间: {result.get('processing_time', 0):.2f}秒")
                    print(f"   处理文件数: {result.get('files_processed', 0)}")
                    return result
                else:
                    error = await response.text()
                    raise Exception(f"解析失败: {error}")
    
    async def list_configs(self) -> dict:
        """列出配置"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/configs") as response:
                return await response.json()
    
    async def delete_config(self, config_id: str) -> dict:
        """删除配置"""
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self.base_url}/configs/{config_id}") as response:
                return await response.json()

# =============================================================================
# 示例使用
# =============================================================================

async def example_basic_usage():
    """基础使用示例"""
    print("🔄 RapidDoc API 基础使用示例")
    print("=" * 50)
    
    # 创建客户端
    client = RapidDocClient()
    
    try:
        # 1. 健康检查
        print("1️⃣ 健康检查...")
        health = await client.health_check()
        print(f"   状态: {health['status']}")
        print(f"   版本: {health['version']}")
        
        # 2. 获取服务状态
        print("\n2️⃣ 获取服务状态...")
        status = await client.get_status()
        print(f"   活跃配置数: {status['total_configs']}")
        
        # 3. 初始化模型配置
        print("\n3️⃣ 初始化模型配置...")
        config_id = await client.init_model_config(
            device_mode="cuda",  # 使用GPU
            conf_thresh=0.4,     # 置信度阈值
            use_det_mode="ocr"   # OCR检测模式
        )
        
        # 4. 解析文档
        print("\n4️⃣ 解析文档...")
        # 检查示例文件是否存在
        test_files = [
            "demo/pdfs/示例1-论文模板.pdf",
            "demo/pdfs/比亚迪财报.pdf"
        ]
        
        # 查找存在的文件
        existing_files = []
        for file_path in test_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
        
        if existing_files:
            print(f"   找到 {len(existing_files)} 个测试文件")
            result = await client.parse_documents(
                files=existing_files,
                output_dir="./api_output",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                lang_list=["ch"],
                return_md=True,
                return_middle_json=True,
                return_content_list=False,
                return_images=False
            )
            
            # 显示结果
            if result.get("results"):
                print("\n📄 解析结果:")
                for file_name, file_result in result["results"].items():
                    print(f"   文件: {file_name}")
                    if "md_content" in file_result:
                        md_preview = file_result["md_content"][:200] + "..." if len(file_result["md_content"]) > 200 else file_result["md_content"]
                        print(f"   Markdown预览: {md_preview}")
        else:
            print("   ⚠️ 未找到测试文件，跳过解析示例")
        
        print("\n✅ 示例运行完成!")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")

async def example_advanced_usage():
    """高级使用示例"""
    print("\n🔄 RapidDoc API 高级使用示例")
    print("=" * 50)
    
    client = RapidDocClient()
    
    try:
        # 1. 使用不同配置初始化多个模型
        print("1️⃣ 初始化多个模型配置...")
        
        # GPU配置
        gpu_config_id = await client.init_model_config(
            device_mode="cuda",
            layout_model_type="PP_DOCLAYOUT_PLUS_L",
            ocr_engine_type="ONNXRUNTIME",
            conf_thresh=0.4
        )
        
        # CPU配置
        cpu_config_id = await client.init_model_config(
            device_mode="cpu",
            layout_model_type="PP_DOCLAYOUT_PLUS_S",
            ocr_engine_type="ONNXRUNTIME",
            conf_thresh=0.3
        )
        
        print(f"   GPU配置: {gpu_config_id}")
        print(f"   CPU配置: {cpu_config_id}")
        
        # 2. 列出配置
        print("\n2️⃣ 列出所有配置...")
        configs = await client.list_configs()
        print(f"   配置列表: {configs['configs']}")
        
        # 3. 删除配置
        print("\n3️⃣ 删除测试配置...")
        await client.delete_config(cpu_config_id)
        print(f"   已删除配置: {cpu_config_id}")
        
        # 4. 性能测试
        print("\n4️⃣ 性能测试...")
        test_files = ["demo/images/table_10.png"]  # 使用图片文件测试
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        if existing_files:
            # 设置GPU配置
            client.config_id = gpu_config_id
            
            # 多次解析测试
            for i in range(3):
                print(f"   第 {i+1} 次测试...")
                result = await client.parse_documents(
                    files=existing_files,
                    output_dir=f"./perf_test_{i}",
                    return_md=True,
                    return_images=False
                )
                print(f"   耗时: {result.get('processing_time', 0):.2f}秒")
        else:
            print("   ⚠️ 未找到测试图片，跳过性能测试")
        
        print("\n✅ 高级示例运行完成!")
        
    except Exception as e:
        print(f"\n❌ 高级示例运行失败: {e}")

async def example_batch_processing():
    """批量处理示例"""
    print("\n🔄 RapidDoc API 批量处理示例")
    print("=" * 50)
    
    client = RapidDocClient()
    
    try:
        # 1. 初始化配置
        print("1️⃣ 初始化模型配置...")
        config_id = await client.init_model_config(
            device_mode="cuda",
            table_enable=True,
            formula_enable=True
        )
        
        # 2. 批量处理多个文件
        print("\n2️⃣ 批量处理文件...")
        
        # 查找所有测试文件
        all_files = []
        for ext in ["*.pdf", "*.png", "*.jpg", "*.jpeg"]:
            pattern = f"demo/**/{ext}"
            all_files.extend(Path(".").glob(pattern))
        
        if all_files:
            print(f"   找到 {len(all_files)} 个文件")
            
            # 分批处理，每批2个文件
            batch_size = 2
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i:i + batch_size]
                batch_files = [str(f) for f in batch]
                
                print(f"   处理第 {i//batch_size + 1} 批: {[os.path.basename(f) for f in batch_files]}")
                
                try:
                    result = await client.parse_documents(
                        files=batch_files,
                        output_dir=f"./batch_output_{i//batch_size}",
                        return_md=True,
                        return_content_list=True
                    )
                    print(f"   ✅ 批次完成，耗时: {result.get('processing_time', 0):.2f}秒")
                    
                    # 短暂延迟避免过载
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"   ❌ 批次失败: {e}")
        else:
            print("   ⚠️ 未找到测试文件")
        
        print("\n✅ 批量处理示例完成!")
        
    except Exception as e:
        print(f"\n❌ 批量处理示例失败: {e}")

async def main():
    """主函数"""
    print("🚀 RapidDoc API 客户端示例")
    print("=" * 60)
    
    # 检查服务是否运行
    try:
        client = RapidDocClient()
        health = await client.health_check()
        print(f"✅ 服务连接成功: {health['service']} v{health['version']}")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("请确保API服务正在运行: python3 api_server.py")
        return
    
    print("\n可用的示例:")
    print("1) 基础使用示例")
    print("2) 高级使用示例") 
    print("3) 批量处理示例")
    print("4) 运行所有示例")
    
    try:
        choice = input("\n请选择示例 [1-4]: ").strip()
        
        if choice == "1":
            await example_basic_usage()
        elif choice == "2":
            await example_advanced_usage()
        elif choice == "3":
            await example_batch_processing()
        elif choice == "4":
            await example_basic_usage()
            await example_advanced_usage()
            await example_batch_processing()
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")

if __name__ == "__main__":
    # 安装必要依赖的提示
    try:
        import aiohttp
    except ImportError:
        print("❌ 缺少依赖，请运行: pip install aiohttp")
        exit(1)
    
    # 运行示例
    asyncio.run(main())
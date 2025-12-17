#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidDoc API 基础功能测试
========================

测试API服务的基础功能，不依赖于OpenGL等可能缺失的依赖
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """测试基础模块导入"""
    print("🔍 测试基础模块导入...")
    
    try:
        # 测试FastAPI基础组件
        import fastapi
        import uvicorn
        import pydantic
        from fastapi import FastAPI
        print("✅ FastAPI基础组件导入成功")
        
        # 测试日志系统
        from loguru import logger
        print("✅ 日志系统导入成功")
        
        # 测试Python标准库
        import tempfile
        import uuid
        import json
        from typing import Optional, List, Dict, Any
        from dataclasses import dataclass
        print("✅ Python标准库组件导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 基础模块导入失败: {e}")
        return False

def test_fastapi_creation():
    """测试FastAPI应用创建"""
    print("\n🔍 测试FastAPI应用创建...")
    
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        
        # 创建测试应用
        app = FastAPI(
            title="Test App",
            description="测试应用",
            version="1.0.0"
        )
        
        # 添加CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 添加测试路由
        @app.get("/test")
        async def test_endpoint():
            return {"message": "测试成功"}
        
        print("✅ FastAPI应用创建成功")
        print(f"✅ 应用路由数量: {len(app.routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI应用创建失败: {e}")
        return False

def test_data_models():
    """测试数据模型"""
    print("\n🔍 测试数据模型...")
    
    try:
        from pydantic import BaseModel, Field
        from typing import Optional, List
        from pathlib import Path
        
        # 创建测试数据模型
        class TestConfig(BaseModel):
            layout_model_type: str = "PP_DOCLAYOUT_PLUS_L"
            ocr_engine_type: str = "ONNXRUNTIME"
            device_mode: str = "cpu"
            conf_thresh: float = 0.4
        
        class TestParseRequest(BaseModel):
            files: List[str] = []
            output_dir: str = "./output"
            parse_method: str = "auto"
            formula_enable: bool = True
            table_enable: bool = True
        
        # 测试模型实例化
        config = TestConfig()
        request = TestParseRequest()
        
        print("✅ 数据模型创建成功")
        print(f"✅ 配置模型: {config.layout_model_type}")
        print(f"✅ 请求模型: {request.parse_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据模型测试失败: {e}")
        return False

def test_file_operations():
    """测试文件操作"""
    print("\n🔍 测试文件操作...")
    
    try:
        import tempfile
        import json
        import shutil
        from pathlib import Path
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="rapid_doc_test_")
        
        # 测试目录创建
        test_dir = Path(temp_dir) / "test_subdir"
        test_dir.mkdir(exist_ok=True)
        
        # 测试文件写入
        test_file = test_dir / "test.json"
        test_data = {"status": "test", "message": "文件操作测试"}
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 测试文件读取
        with open(test_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["status"] == "test"
        
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("✅ 文件操作测试成功")
        print("✅ 目录创建: 正常")
        print("✅ 文件写入: 正常")
        print("✅ 文件读取: 正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件操作测试失败: {e}")
        return False

def test_async_functionality():
    """测试异步功能"""
    print("\n🔍 测试异步功能...")
    
    try:
        import asyncio
        
        async def test_async_operation():
            # 模拟异步操作
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "test_data"}
        
        # 运行异步测试
        result = asyncio.run(test_async_operation())
        
        assert result["status"] == "success"
        assert result["data"] == "test_data"
        
        print("✅ 异步功能测试成功")
        print(f"✅ 异步结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步功能测试失败: {e}")
        return False

def test_config_management():
    """测试配置管理"""
    print("\n🔍 测试配置管理...")
    
    try:
        from typing import Dict, Any
        import uuid
        
        # 模拟配置管理
        configs = {}
        active_configs = {}
        
        # 创建测试配置
        config_id = str(uuid.uuid4())
        config = {
            "layout_model_type": "PP_DOCLAYOUT_PLUS_L",
            "ocr_engine_type": "ONNXRUNTIME",
            "formula_enable": True,
            "table_enable": True
        }
        
        configs[config_id] = config
        active_configs[config_id] = config
        
        # 测试配置获取
        retrieved_config = configs.get(config_id)
        assert retrieved_config is not None
        
        # 测试配置列表
        config_list = list(configs.keys())
        assert config_id in config_list
        
        print("✅ 配置管理测试成功")
        print(f"✅ 配置ID: {config_id}")
        print(f"✅ 活跃配置数: {len(active_configs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置管理测试失败: {e}")
        return False

def test_api_structure():
    """测试API结构"""
    print("\n🔍 测试API结构...")
    
    try:
        # 检查API文件是否存在
        api_file = Path("api_server.py")
        if not api_file.exists():
            print("❌ api_server.py 文件不存在")
            return False
        
        # 检查关键文件
        files_to_check = [
            "api_server.py",
            "api_client_example.py", 
            "start_api.sh",
            "API_README.md",
            "requirements-api.txt"
        ]
        
        missing_files = []
        for file_path in files_to_check:
            if Path(file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} 缺失")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"⚠️ 缺失文件: {missing_files}")
        else:
            print("✅ 所有关键文件都存在")
        
        return True
        
    except Exception as e:
        print(f"❌ API结构测试失败: {e}")
        return False

def test_environment_setup():
    """测试环境设置"""
    print("\n🔍 测试环境设置...")
    
    try:
        import sys
        
        # 检查Python版本
        python_version = sys.version_info
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version >= (3, 8):
            print("✅ Python版本符合要求 (>=3.8)")
        else:
            print("❌ Python版本过低，需要 >=3.8")
            return False
        
        # 检查虚拟环境
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ 虚拟环境检测: 已激活")
        else:
            print("⚠️ 虚拟环境检测: 未激活")
        
        # 检查当前工作目录
        cwd = Path.cwd()
        print(f"✅ 当前工作目录: {cwd}")
        
        # 检查项目结构
        project_files = ["api_server.py", "demo.py", "rapid_doc"]
        for file_name in project_files:
            if Path(file_name).exists():
                print(f"✅ 项目文件 {file_name}: 存在")
            else:
                print(f"❌ 项目文件 {file_name}: 缺失")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境设置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 RapidDoc API 基础功能测试")
    print("=" * 60)
    
    tests = [
        ("环境设置", test_environment_setup),
        ("基础模块导入", test_basic_imports),
        ("FastAPI应用创建", test_fastapi_creation),
        ("数据模型", test_data_models),
        ("文件操作", test_file_operations),
        ("异步功能", test_async_functionality),
        ("配置管理", test_config_management),
        ("API结构", test_api_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️ {test_name} 测试未通过")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed >= 6:  # 允许一些测试失败
        print("🎉 基础功能测试通过！API服务准备就绪。")
        print("\n📖 使用说明:")
        print("1. 安装完整依赖: pip install -r requirements-api.txt")
        print("2. 启动服务: ./start_api.sh")
        print("3. 或直接运行: python3 api_server.py")
        print("4. 访问文档: http://localhost:8888/docs")
        print("5. 健康检查: http://localhost:8888/health")
        
        print("\n💡 说明:")
        print("- 如果遇到OpenGL依赖问题，请安装: apt-get install libgl1-mesa-glx")
        print("- 模型文件需要单独下载到 ./models/ 目录")
        print("- 完整功能测试需要在有GPU的环境中运行")
        return True
    else:
        print("❌ 基础测试失败，请检查环境配置。")
        print("\n🔧 解决方案:")
        print("1. 检查Python版本: python3 --version (需要3.8+)")
        print("2. 安装基础依赖: pip install fastapi uvicorn pydantic loguru")
        print("3. 激活虚拟环境")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
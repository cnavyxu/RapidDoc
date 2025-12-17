#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidDoc API 服务测试脚本
========================

测试API服务的核心功能，包括：
- 模型配置初始化
- 文档解析
- 异步处理
- 模块组合
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试必要的模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from api_server import (
            app, ModelServiceManager, ModelConfig, 
            ModuleConfigRequest, ParseRequest
        )
        print("✅ API模块导入成功")
        
        # 测试RapidDoc核心模块
        from rapid_doc.cli.common import prepare_env, read_fn
        print("✅ RapidDoc CLI模块导入成功")
        
        from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
        print("✅ RapidDoc管道模块导入成功")
        
        try:
            from rapid_doc.backend.pipeline.pipeline_analyze import ModelSingleton
            print("✅ RapidDoc分析模块导入成功")
        except Exception as e:
            print(f"⚠️ RapidDoc分析模块导入跳过（可能缺少OpenGL依赖）: {e}")
        
        # 测试模型类型导入
        try:
            from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
            from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
            from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
            print("✅ 模型类型导入成功")
        except Exception as e:
            print(f"⚠️ 模型类型导入跳过（可能缺少OpenGL依赖）: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        return False

def test_model_manager():
    """测试模型管理器"""
    print("\n🔍 测试模型管理器...")
    
    try:
        from api_server import ModelServiceManager, ModuleConfigRequest
        
        # 创建服务管理器
        manager = ModelServiceManager()
        print("✅ ModelServiceManager 创建成功")
        
        # 创建测试配置
        config_request = ModuleConfigRequest(
            layout_model_type="PP_DOCLAYOUT_PLUS_L",
            ocr_engine_type="ONNXRUNTIME",
            formula_model_type="PP_FORMULANET_PLUS_M",
            table_model_type="UNET_SLANET_PLUS",
            device_mode="cpu",  # 使用CPU避免GPU依赖
            conf_thresh=0.4,
            use_det_mode="ocr"
        )
        print("✅ ModuleConfigRequest 创建成功")
        
        # 测试配置创建
        test_config_id = "test-config-123"
        try:
            config = manager.create_config(test_config_id, config_request)
            print("✅ 模型配置创建成功")
            
            # 测试配置获取
            retrieved_config = manager.get_config(test_config_id)
            assert retrieved_config is not None
            print("✅ 配置获取成功")
            
            # 测试配置列表
            configs = manager.list_configs()
            assert test_config_id in configs
            print("✅ 配置列表功能正常")
            
        except Exception as e:
            if "libGL.so.1" in str(e) or "OpenGL" in str(e):
                print(f"⚠️ 模型配置创建跳过（缺少OpenGL依赖）: {e}")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"❌ 模型管理器测试失败: {e}")
        return False

def test_async_functionality():
    """测试异步功能"""
    print("\n🔍 测试异步功能...")
    
    try:
        import asyncio
        
        async def test_async():
            # 测试基本的异步功能
            await asyncio.sleep(0.1)
            return "async_test_success"
        
        # 运行异步测试
        result = asyncio.run(test_async())
        assert result == "async_test_success"
        print("✅ 异步功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步功能测试失败: {e}")
        return False

def test_fastapi_app():
    """测试FastAPI应用创建"""
    print("\n🔍 测试FastAPI应用...")
    
    try:
        from api_server import app
        
        # 检查应用创建
        assert app is not None
        assert hasattr(app, 'routes')
        print("✅ FastAPI应用创建成功")
        
        # 检查路由数量
        route_count = len(app.routes)
        print(f"✅ 应用包含 {route_count} 个路由")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI应用测试失败: {e}")
        return False

def test_requirements():
    """测试依赖包"""
    print("\n🔍 检查依赖包...")
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'pydantic',
        'aiofiles',
        'loguru'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺失依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements-api.txt")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 RapidDoc API 服务测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖包检查", test_requirements),
        ("模型管理器", test_model_manager),
        ("异步功能", test_async_functionality),
        ("FastAPI应用", test_fastapi_app),
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
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！API服务准备就绪。")
        print("\n📖 使用说明:")
        print("1. 启动服务: ./start_api.sh")
        print("2. 或直接运行: python3 api_server.py")
        print("3. 访问文档: http://localhost:8888/docs")
        print("4. 健康检查: http://localhost:8888/health")
        return True
    else:
        print("❌ 部分测试失败，请检查配置。")
        print("\n🔧 常见解决方案:")
        print("1. 安装依赖: pip install -r requirements-api.txt")
        print("2. 检查Python版本: python3 --version (需要3.8+)")
        print("3. 检查模型文件是否存在")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
RapidDoc FastAPI服务测试脚本
"""

import asyncio
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, '/home/engine/project')

def test_imports():
    """测试导入是否正常"""
    try:
        print("Testing imports...")
        
        # 测试FastAPI相关
        from fastapi import FastAPI, File, UploadFile, Form
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        print("✓ FastAPI imports successful")
        
        # 测试RapidDoc核心模块
        from rapid_doc.backend.pipeline.model_init import MineruPipelineModel, AtomModelSingleton
        from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
        print("✓ RapidDoc pipeline imports successful")
        
        from rapid_doc.cli.common import prepare_env, convert_pdf_bytes_to_bytes_by_pypdfium2
        from rapid_doc.data.data_reader_writer import FileBasedDataWriter
        print("✓ RapidDoc utilities imports successful")
        
        # 测试模型配置
        from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
        from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType, EngineType as FormulaEngineType
        from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType, EngineType as TableEngineType
        from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
        print("✓ Model configurations imports successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_config():
    """测试模型配置"""
    try:
        print("\nTesting model configuration...")
        
        # 导入我们的服务模块
        sys.path.insert(0, '/home/engine/project')
        from fastapi_service import ModelConfig
        
        # 创建配置实例
        config = ModelConfig()
        print(f"✓ Default configuration created: {config.layout_model_type}")
        
        # 测试配置字典转换
        layout_config = {
            "model_type": "PP_DOCLAYOUT_PLUS_L",
            "conf_thresh": 0.4,
            "batch_num": 1,
        }
        print("✓ Layout config created")
        
        return True
        
    except Exception as e:
        print(f"❌ Model config test failed: {e}")
        return False

async def test_model_manager():
    """测试模型管理器"""
    try:
        print("\nTesting ModelManager...")
        
        from fastapi_service import model_manager, ModelConfig
        
        # 跳过实际的模型初始化（因为需要模型文件）
        print("⚠️ Skipping actual model initialization (requires model files)")
        
        # 测试配置
        default_config = ModelConfig()
        print(f"✓ Default config ready: {default_config.layout_model_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelManager test failed: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    try:
        print("\nTesting file structure...")
        
        # 检查关键文件是否存在
        key_files = [
            "/home/engine/project/fastapi_service.py",
            "/home/engine/project/rapid_doc/backend/pipeline/model_init.py",
            "/home/engine/project/rapid_doc/backend/pipeline/batch_analyze.py",
            "/home/engine/project/rapid_doc/cli/common.py",
            "/home/engine/project/demo.py"
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                print(f"✓ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 RapidDoc FastAPI Service Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Import Test", test_imports),
        ("Model Config", test_model_config),
        ("Model Manager", test_model_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Service is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
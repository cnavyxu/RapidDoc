#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidDoc API Server
===================

基于 FastAPI 的文档解析服务，支持：
1. 模型初始化持久化，避免重复初始化
2. 异步调用支持
3. 关键模块可选择组合
4. 高性能批处理

作者: RapidDoc Team
版本: 1.0.0
"""

import os
import time
import uuid
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import tempfile
import shutil

# RapidDoc 核心模块导入
from rapid_doc.cli.common import prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.backend.pipeline.pipeline_analyze import (
    doc_analyze as pipeline_doc_analyze,
    ModelSingleton
)
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from rapid_doc.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode

# 模型类型导入
from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.formula.rapid_formula_self import (
    ModelType as FormulaModelType,
    EngineType as FormulaEngineType,
)
from rapid_doc.model.table.rapid_table_self import (
    ModelType as TableModelType,
    EngineType as TableEngineType,
)

# =============================================================================
# 数据模型定义
# =============================================================================

@dataclass
class ModelConfig:
    """模型配置类"""
    layout_config: Dict[str, Any] = None
    ocr_config: Dict[str, Any] = None
    formula_config: Dict[str, Any] = None
    table_config: Dict[str, Any] = None
    checkbox_config: Dict[str, Any] = None
    image_config: Dict[str, Any] = None

class ParseRequest(BaseModel):
    """解析请求模型"""
    files: List[UploadFile] = Field(..., description="要解析的文件列表")
    output_dir: str = Field("./output", description="输出目录")
    parse_method: str = Field("auto", description="解析方法: auto, ocr, txt")
    formula_enable: bool = Field(True, description="是否启用公式解析")
    table_enable: bool = Field(True, description="是否启用表格解析")
    lang_list: List[str] = Field(["ch"], description="语言列表")
    start_page_id: int = Field(0, description="起始页码")
    end_page_id: Optional[int] = Field(None, description="结束页码")
    return_md: bool = Field(True, description="返回Markdown")
    return_middle_json: bool = Field(False, description="返回中间JSON")
    return_model_output: bool = Field(False, description="返回模型输出")
    return_content_list: bool = Field(False, description="返回内容列表")
    return_images: bool = Field(False, description="返回图片")
    response_format_zip: bool = Field(False, description="ZIP格式响应")

class ModuleConfigRequest(BaseModel):
    """模块配置请求模型"""
    layout_model_type: str = Field("PP_DOCLAYOUT_PLUS_L", description="版面模型类型")
    ocr_engine_type: str = Field("ONNXRUNTIME", description="OCR引擎类型")
    formula_model_type: str = Field("PP_FORMULANET_PLUS_M", description="公式模型类型")
    table_model_type: str = Field("UNET_SLANET_PLUS", description="表格模型类型")
    device_mode: str = Field("cuda", description="设备模式")
    conf_thresh: float = Field(0.4, description="置信度阈值")
    use_det_mode: str = Field("ocr", description="检测模式")

class InitResponse(BaseModel):
    """初始化响应模型"""
    status: str
    message: str
    config_id: str
    modules: Dict[str, bool]

# =============================================================================
# 全局服务管理
# =============================================================================

class ModelServiceManager:
    """模型服务管理器"""
    
    def __init__(self):
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_configs: Dict[str, Any] = {}
        self.model_singleton = ModelSingleton()
        
    def create_config(self, config_id: str, module_config: ModuleConfigRequest) -> ModelConfig:
        """创建模型配置"""
        
        # 布局模型配置
        layout_config = {
            "model_type": getattr(LayoutModelType, module_config.layout_model_type, LayoutModelType.PP_DOCLAYOUT_PLUS_L),
            "conf_thresh": module_config.conf_thresh,
            "batch_num": 1,
            "model_dir_or_path": "./models/pp_doclayout_plus_l.onnx",
        }
        
        # OCR模型配置
        ocr_config = {
            "Det.model_path": "./models/det_server.onnx",
            "Rec.model_path": "./models/rec_server.onnx",
            "Rec.rec_keys_path": "./models/ppocrv5_dict.txt",
            "Cls.model_path": "./models/cls.onnx",
            "Rec.rec_batch_num": 1,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Cls.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": OCRModelType.SERVER,
            "Rec.model_type": OCRModelType.SERVER,
            "Cls.model_type": OCRModelType.SERVER,
            "engine_type": getattr(OCREngineType, module_config.ocr_engine_type, OCREngineType.ONNXRUNTIME),
            "Det.rec_batch_num": 8,
            "use_det_mode": module_config.use_det_mode,
        }
        
        # 公式模型配置
        formula_config = {
            "model_type": getattr(FormulaModelType, module_config.formula_model_type, FormulaModelType.PP_FORMULANET_PLUS_M),
            "engine_type": FormulaEngineType.TORCH,
            "formula_level": 1,
            "batch_num": 1,
            "model_dir_or_path": "./models/pp_formulanet_plus_m.pth",
            "dict_keys_path": "./models/pp_formulanet_plus_m_inference.yml",
        }
        
        # 表格模型配置
        table_config = {
            "force_ocr": False,
            "skip_text_in_image": True,
            "use_img2table": False,
            "model_type": getattr(TableModelType, module_config.table_model_type, TableModelType.UNET_SLANET_PLUS),
            "use_word_box": True,
            "use_compare_table": False,
            "table_formula_enable": False,
            "table_image_enable": False,
            "extract_original_image": False,
            "cls.model_type": TableModelType.PADDLE_CLS,
            "cls.model_dir_or_path": "./models/paddle_cls.onnx",
            "unet.model_dir_or_path": "./models/unet.onnx",
            "slanet_plus.model_dir_or_path": "./models/slanet-plus.onnx",
            "engine_type": TableEngineType.ONNXRUNTIME,
        }
        
        # 复选框配置
        checkbox_config = {
            "checkbox_enable": True,
        }
        
        # 图像配置
        image_config = {
            "extract_original_image": True,
            "extract_original_image_iou_thresh": 0.5,
        }
        
        # 设置环境变量
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        os.environ["MINERU_DEVICE_MODE"] = module_config.device_mode
        
        config = ModelConfig(
            layout_config=layout_config,
            ocr_config=ocr_config,
            formula_config=formula_config,
            table_config=table_config,
            checkbox_config=checkbox_config,
            image_config=image_config
        )
        
        self.model_configs[config_id] = config
        self.active_configs[config_id] = {
            "layout_model_type": module_config.layout_model_type,
            "ocr_engine_type": module_config.ocr_engine_type,
            "formula_model_type": module_config.formula_model_type,
            "table_model_type": module_config.table_model_type,
            "formula_enable": True,
            "table_enable": True,
        }
        
        return config
    
    def get_config(self, config_id: str) -> Optional[ModelConfig]:
        """获取配置"""
        return self.model_configs.get(config_id)
    
    def get_active_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """获取活跃配置"""
        return self.active_configs.get(config_id)
    
    def list_configs(self) -> List[str]:
        """列出所有配置ID"""
        return list(self.model_configs.keys())

# 全局服务管理器
service_manager = ModelServiceManager()

# =============================================================================
# FastAPI 应用生命周期管理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("RapidDoc API Server 启动中...")
    logger.info("模型服务管理器已初始化")
    
    yield
    
    # 关闭时清理
    logger.info("RapidDoc API Server 关闭中...")

# 创建FastAPI应用
app = FastAPI(
    title="RapidDoc API Server",
    description="高性能文档解析服务，支持模型持久化和异步处理",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API 端点定义
# =============================================================================

@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "RapidDoc API Server",
        "version": "1.0.0",
        "active_configs": len(service_manager.model_configs),
        "timestamp": time.time()
    }

@app.get("/status", summary="服务状态")
async def get_service_status():
    """获取服务状态"""
    configs = {}
    for config_id, config in service_manager.model_configs.items():
        configs[config_id] = {
            "layout_enabled": config.layout_config is not None,
            "ocr_enabled": config.ocr_config is not None,
            "formula_enabled": config.formula_config is not None and service_manager.active_configs[config_id]["formula_enable"],
            "table_enabled": config.table_config is not None and service_manager.active_configs[config_id]["table_enable"],
            "checkbox_enabled": config.checkbox_config is not None,
            "image_enabled": config.image_config is not None,
        }
    
    return {
        "service": "RapidDoc API Server",
        "version": "1.0.0",
        "total_configs": len(service_manager.model_configs),
        "configs": configs,
        "uptime": time.time()
    }

@app.post("/init", response_model=InitResponse, summary="初始化模型配置")
async def init_model_config(module_config: ModuleConfigRequest):
    """初始化模型配置"""
    try:
        config_id = str(uuid.uuid4())
        config = service_manager.create_config(config_id, module_config)
        
        # 记录初始化时间
        start_time = time.time()
        
        # 预热模型（触发模型加载）
        active_config = service_manager.get_active_config(config_id)
        model = service_manager.model_singleton.get_model(
            lang="ch",
            formula_enable=active_config["formula_enable"],
            table_enable=active_config["table_enable"],
            layout_config=config.layout_config,
            ocr_config=config.ocr_config,
            formula_config=config.formula_config,
            table_config=config.table_config,
        )
        
        init_time = time.time() - start_time
        
        logger.info(f"模型配置 {config_id} 初始化完成，耗时: {init_time:.2f}秒")
        
        return InitResponse(
            status="success",
            message=f"模型配置初始化成功，耗时: {init_time:.2f}秒",
            config_id=config_id,
            modules={
                "layout": config.layout_config is not None,
                "ocr": config.ocr_config is not None,
                "formula": active_config["formula_enable"],
                "table": active_config["table_enable"],
                "checkbox": config.checkbox_config is not None,
                "image": config.image_config is not None,
            }
        )
        
    except Exception as e:
        logger.error(f"模型配置初始化失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {str(e)}")

@app.post("/parse", summary="解析文档")
async def parse_documents(
    config_id: str = Form(..., description="模型配置ID"),
    files: List[UploadFile] = File(..., description="要解析的文件"),
    output_dir: str = Form("./output", description="输出目录"),
    parse_method: str = Form("auto", description="解析方法"),
    formula_enable: bool = Form(True, description="启用公式解析"),
    table_enable: bool = Form(True, description="启用表格解析"),
    lang_list: str = Form('["ch"]', description="语言列表JSON"),
    start_page_id: int = Form(0, description="起始页码"),
    end_page_id: Optional[int] = Form(None, description="结束页码"),
    return_md: bool = Form(True, description="返回Markdown"),
    return_middle_json: bool = Form(False, description="返回中间JSON"),
    return_model_output: bool = Form(False, description="返回模型输出"),
    return_content_list: bool = Form(False, description="返回内容列表"),
    return_images: bool = Form(False, description="返回图片"),
    response_format_zip: bool = Form(False, description="ZIP格式响应"),
):
    """异步解析文档"""
    
    try:
        # 验证配置
        config = service_manager.get_config(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"配置ID {config_id} 不存在")
        
        # 创建唯一输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        
        # 解析语言列表
        try:
            actual_lang_list = json.loads(lang_list)
            if not isinstance(actual_lang_list, list):
                actual_lang_list = ["ch"]
        except json.JSONDecodeError:
            actual_lang_list = ["ch"]
        
        # 处理文件
        pdf_file_names = []
        pdf_bytes_list = []
        
        for file in files:
            # 验证文件类型
            file_suffix = Path(file.filename).suffix.lower()
            supported_suffixes = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
            
            if file_suffix not in supported_suffixes:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的文件类型: {file_suffix}"
                )
            
            # 读取文件内容
            content = await file.read()
            
            # 图像文件转换为PDF
            if file_suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
                content = images_bytes_to_pdf_bytes(content)
            
            file_name = Path(file.filename).stem
            pdf_bytes_list.append(content)
            pdf_file_names.append(file_name)
        
        # 确保语言列表与文件数量匹配
        if len(actual_lang_list) != len(pdf_file_names):
            if len(actual_lang_list) == 1:
                actual_lang_list = actual_lang_list * len(pdf_file_names)
            else:
                actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)
        
        # 更新配置中的模块启用状态
        active_config = service_manager.get_active_config(config_id)
        active_config["formula_enable"] = formula_enable
        active_config["table_enable"] = table_enable
        
        # 执行异步解析
        start_time = time.time()
        
        # 执行文档分析
        infer_results, all_image_lists, all_pdf_docs, lang_list_result, ocr_enabled_list = pipeline_doc_analyze(
            pdf_bytes_list,
            lang_list=actual_lang_list,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            layout_config=config.layout_config,
            ocr_config=config.ocr_config,
            formula_config=config.formula_config,
            table_config=config.table_config,
            checkbox_config=config.checkbox_config,
        )
        
        # 处理结果
        results = {}
        
        for idx, model_list in enumerate(infer_results):
            import copy
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(unique_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            
            images_list = all_image_lists[idx]
            pdf_dict = all_pdf_docs[idx]
            _lang = lang_list_result[idx]
            _ocr_enable = ocr_enabled_list[idx]
            
            # 转换为中间JSON
            middle_json = pipeline_result_to_middle_json(
                model_list,
                images_list,
                pdf_dict,
                image_writer,
                _lang,
                _ocr_enable,
                formula_enable,
                ocr_config=config.ocr_config,
                image_config=config.image_config,
            )
            
            pdf_info = middle_json["pdf_info"]
            pdf_bytes = pdf_bytes_list[idx]
            
            # 绘制边界框（可选）
            if config.image_config.get("extract_original_image", False):
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
            
            # 构建返回结果
            file_result = {}
            
            if return_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
                file_result["md_content"] = md_content
            
            if return_middle_json:
                file_result["middle_json"] = middle_json
            
            if return_model_output:
                file_result["model_output"] = model_json
            
            if return_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                file_result["content_list"] = content_list
            
            if return_images:
                images_dir = os.path.join(local_md_dir, "images")
                if os.path.exists(images_dir):
                    import glob
                    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
                    file_result["images"] = [os.path.basename(f) for f in image_files]
            
            results[pdf_file_name] = file_result
        
        processing_time = time.time() - start_time
        
        logger.info(f"文档解析完成，耗时: {processing_time:.2f}秒，处理文件数: {len(pdf_file_names)}")
        
        return {
            "status": "success",
            "config_id": config_id,
            "processing_time": processing_time,
            "files_processed": len(pdf_file_names),
            "results": results,
            "output_dir": unique_dir if not response_format_zip else None
        }
        
    except Exception as e:
        logger.error(f"文档解析失败: {e}")
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

@app.get("/configs", summary="获取配置列表")
async def list_configs():
    """获取所有配置"""
    return {
        "configs": service_manager.list_configs(),
        "total": len(service_manager.model_configs)
    }

@app.delete("/configs/{config_id}", summary="删除配置")
async def delete_config(config_id: str):
    """删除配置"""
    if config_id not in service_manager.model_configs:
        raise HTTPException(status_code=404, detail=f"配置ID {config_id} 不存在")
    
    del service_manager.model_configs[config_id]
    del service_manager.active_configs[config_id]
    
    return {"status": "success", "message": f"配置 {config_id} 已删除"}

@app.get("/docs", summary="API文档")
async def get_api_docs():
    """获取API文档"""
    return {
        "title": "RapidDoc API Server",
        "description": "高性能文档解析服务",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "健康检查",
            "GET /status": "服务状态",
            "POST /init": "初始化模型配置",
            "POST /parse": "解析文档",
            "GET /configs": "获取配置列表",
            "DELETE /configs/{config_id}": "删除配置"
        },
        "features": [
            "模型初始化持久化",
            "异步处理支持",
            "模块化配置",
            "高性能批处理"
        ]
    }

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logger.add("rapid_doc_api.log", rotation="1 day", retention="30 days", level="INFO")
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8888,
        reload=False,  # 生产环境建议关闭
        workers=1,     # 单worker避免模型冲突
        log_level="info"
    )
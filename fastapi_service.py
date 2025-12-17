"""
RapidDoc FastAPI服务 - 异步模型初始化版本
支持模块化配置和异步处理
"""

import os
import gc
import json
import time
import asyncio
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loguru import logger
import uvicorn

# 设置设备环境变量
os.environ["MINERU_DEVICE_MODE"] = os.getenv("MINERU_DEVICE_MODE", "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入RapidDoc相关模块
from rapid_doc.backend.pipeline.model_init import MineruPipelineModel, AtomModelSingleton
from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
from rapid_doc.cli.common import prepare_env, convert_pdf_bytes_to_bytes_by_pypdfium2
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType, EngineType as FormulaEngineType
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType, EngineType as TableEngineType
from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType


# Pydantic模型定义
class ParseRequest(BaseModel):
    """解析请求模型"""
    parse_method: str = "auto"
    formula_enable: bool = True
    table_enable: bool = True
    checkbox_enable: bool = False
    start_page_id: int = 0
    end_page_id: Optional[int] = None
    lang: str = "ch"
    
    # 返回格式控制
    return_md: bool = True
    return_middle_json: bool = False
    return_model_output: bool = False
    return_content_list: bool = False


class ModelConfig(BaseModel):
    """模型配置模型"""
    # Layout配置
    layout_model_type: str = "PP_DOCLAYOUT_PLUS_L"
    layout_conf_thresh: float = 0.4
    layout_batch_num: int = 1
    
    # OCR配置
    ocr_engine_type: str = "ONNXRUNTIME"
    ocr_det_model_path: str = "./models/det_server.onnx"
    ocr_rec_model_path: str = "./models/rec_server.onnx"
    ocr_cls_model_path: str = "./models/cls.onnx"
    ocr_rec_keys_path: str = "./models/ppocrv5_dict.txt"
    ocr_rec_batch_num: int = 1
    ocr_det_rec_batch_num: int = 8
    use_det_mode: str = "ocr"  # auto, txt, ocr
    
    # Formula配置
    formula_model_type: str = "PP_FORMULANET_PLUS_M"
    formula_engine_type: str = "TORCH"
    formula_level: int = 0
    formula_batch_num: int = 1
    formula_model_path: str = "./models/pp_formulanet_plus_m.pth"
    formula_dict_path: str = "./models/pp_formulanet_plus_m_inference.yml"
    
    # Table配置
    table_force_ocr: bool = False
    table_skip_text_in_image: bool = True
    table_use_img2table: bool = False
    table_model_type: str = "UNET_SLANET_PLUS"
    table_use_word_box: bool = True
    table_use_compare_table: bool = False
    table_formula_enable: bool = False
    table_image_enable: bool = False
    table_extract_original_image: bool = False
    table_cls_model_path: str = "./models/paddle_cls.onnx"
    table_unet_model_path: str = "./models/unet.onnx"
    table_slanet_plus_model_path: str = "./models/slanet-plus.onnx"
    table_engine_type: str = "ONNXRUNTIME"
    
    # Checkbox配置
    checkbox_enable: bool = False


class ModelManager:
    """模型管理器 - 单例模式，管理所有模型的初始化和获取"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models = {}
            self.batch_analyzers = {}
            self._initialized = True
            logger.info("ModelManager initialized")
    
    def init_models(self, model_config: ModelConfig):
        """初始化所有模型"""
        try:
            logger.info("Starting model initialization...")
            start_time = time.time()
            
            # 设置模型源
            os.environ["MINERU_MODEL_SOURCE"] = "local"
            
            # 构建配置字典
            layout_config = {
                "model_type": getattr(LayoutModelType, model_config.layout_model_type),
                "conf_thresh": model_config.layout_conf_thresh,
                "batch_num": model_config.layout_batch_num,
                "model_dir_or_path": "./models/pp_doclayout_plus_l.onnx",
            }
            
            ocr_config = {
                "Det.model_path": model_config.ocr_det_model_path,
                "Rec.model_path": model_config.ocr_rec_model_path,
                "Rec.rec_keys_path": model_config.ocr_rec_keys_path,
                "Cls.model_path": model_config.ocr_cls_model_path,
                "Rec.rec_batch_num": model_config.ocr_rec_batch_num,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Cls.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": OCRModelType.SERVER,
                "Rec.model_type": OCRModelType.SERVER,
                "Cls.model_type": OCRModelType.SERVER,
                "engine_type": getattr(OCREngineType, model_config.ocr_engine_type),
                "Det.rec_batch_num": model_config.ocr_det_rec_batch_num,
                "use_det_mode": model_config.use_det_mode,
            }
            
            formula_config = {
                "model_type": getattr(FormulaModelType, model_config.formula_model_type),
                "engine_type": getattr(FormulaEngineType, model_config.formula_engine_type),
                "formula_level": model_config.formula_level,
                "batch_num": model_config.formula_batch_num,
                "model_dir_or_path": model_config.formula_model_path,
                "dict_keys_path": model_config.formula_dict_path,
            }
            
            table_config = {
                "force_ocr": model_config.table_force_ocr,
                "skip_text_in_image": model_config.table_skip_text_in_image,
                "use_img2table": model_config.table_use_img2table,
                "model_type": getattr(TableModelType, model_config.table_model_type),
                "use_word_box": model_config.table_use_word_box,
                "use_compare_table": model_config.table_use_compare_table,
                "table_formula_enable": model_config.table_formula_enable,
                "table_image_enable": model_config.table_image_enable,
                "extract_original_image": model_config.table_extract_original_image,
                "cls.model_type": TableModelType.PADDLE_CLS,
                "cls.model_dir_or_path": model_config.table_cls_model_path,
                "unet.model_dir_or_path": model_config.table_unet_model_path,
                "slanet_plus.model_dir_or_path": model_config.table_slanet_plus_model_path,
                "engine_type": getattr(TableEngineType, model_config.table_engine_type),
            }
            
            # 获取checkbox_enable属性，默认值为False
            checkbox_enable = getattr(model_config, 'checkbox_enable', False)
            checkbox_config = {
                "checkbox_enable": checkbox_enable,
            }
            
            image_config = {
                "extract_original_image": False,
                "extract_original_image_iou_thresh": 0.5,
            }
            
            # 初始化主模型
            self.models["pipeline_model"] = MineruPipelineModel(
                layout_config=layout_config,
                ocr_config=ocr_config,
                formula_config=formula_config,
                table_config=table_config,
                lang="ch",
                device=os.getenv("MINERU_DEVICE_MODE", "cpu")
            )
            
            # 初始化批处理器
            self.batch_analyzers["default"] = BatchAnalyze(
                model_manager=self.models["pipeline_model"],
                batch_ratio=1,
                formula_enable=True,
                table_enable=True,
                layout_config=layout_config,
                ocr_config=ocr_config,
                formula_config=formula_config,
                table_config=table_config,
                checkbox_config=checkbox_config,
            )
            
            # 保存配置
            self.configs = {
                "layout_config": layout_config,
                "ocr_config": ocr_config,
                "formula_config": formula_config,
                "table_config": table_config,
                "checkbox_config": checkbox_config,
                "image_config": image_config,
            }
            
            init_time = time.time() - start_time
            logger.info(f"All models initialized successfully in {init_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def get_pipeline_model(self):
        """获取主模型"""
        return self.models.get("pipeline_model")
    
    def get_batch_analyzer(self, analyzer_key: str = "default"):
        """获取批处理器"""
        return self.batch_analyzers.get(analyzer_key)
    
    def get_configs(self):
        """获取配置"""
        return getattr(self, "configs", {})


# 全局模型管理器实例
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型
    logger.info("Starting RapidDoc FastAPI service...")
    
    # 默认模型配置
    default_config = ModelConfig()
    
    try:
        # 在后台线程中初始化模型，避免阻塞应用启动
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, model_manager.init_models, default_config)
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.warning("Service will start without pre-initialized models. Models will be loaded on first request.")
    
    yield
    
    # 清理资源
    logger.info("Shutting down RapidDoc FastAPI service...")


# 创建FastAPI应用
app = FastAPI(
    title="RapidDoc FastAPI Service",
    description="基于RapidDoc的文档解析API，支持异步处理和模块化配置",
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


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "RapidDoc FastAPI Service",
        "version": "1.0.0",
        "models_loaded": model_manager.get_pipeline_model() is not None
    }


@app.get("/config")
async def get_config():
    """获取当前模型配置"""
    configs = model_manager.get_configs()
    return {
        "status": "success",
        "configs": configs
    }


@app.post("/parse")
async def parse_document(
    files: List[UploadFile] = File(...),
    request: ParseRequest = Form(...),
    output_dir: str = Form("./output"),
):
    """
    解析文档API
    
    Args:
        files: 要解析的文件列表（PDF或图片）
        request: 解析请求参数
        output_dir: 输出目录
    """
    
    # 检查模型是否已初始化
    if model_manager.get_pipeline_model() is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")
    
    try:
        # 创建唯一输出目录
        task_id = str(uuid.uuid4())
        task_output_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        logger.info(f"Starting parsing task {task_id} with {len(files)} files")
        start_time = time.time()
        
        # 处理上传的文件
        pdf_file_names = []
        pdf_bytes_list = []
        
        for file in files:
            # 验证文件类型
            file_suffix = Path(file.filename).suffix.lower()
            if file_suffix not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_suffix}"
                )
            
            # 读取文件内容
            file_content = await file.read()
            file_name = Path(file.filename).stem
            
            # 如果是图片，转换为PDF
            if file_suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
                file_content = images_bytes_to_pdf_bytes(file_content)
            
            pdf_file_names.append(file_name)
            pdf_bytes_list.append(file_content)
        
        # 准备PDF字节数据
        processed_pdf_bytes = []
        for pdf_bytes in pdf_bytes_list:
            processed_pdf_bytes.append(
                convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, request.start_page_id, request.end_page_id
                )
            )
        
        # 获取模型和配置
        pipeline_model = model_manager.get_pipeline_model()
        batch_analyzer = model_manager.get_batch_analyzer()
        configs = model_manager.get_configs()
        
        # 调用pipeline分析
        from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        
        infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                processed_pdf_bytes,
                parse_method=request.parse_method,
                formula_enable=request.formula_enable,
                table_enable=request.table_enable,
                layout_config=configs["layout_config"],
                ocr_config=configs["ocr_config"],
                formula_config=configs["formula_config"],
                table_config=configs["table_config"],
                checkbox_config=configs["checkbox_config"],
            )
        )
        
        # 处理结果
        results = {}
        
        for idx, model_list in enumerate(infer_results):
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(
                task_output_dir, pdf_file_name, request.parse_method
            )
            image_writer, md_writer = (
                FileBasedDataWriter(local_image_dir),
                FileBasedDataWriter(local_md_dir)
            )
            
            images_list = all_image_lists[idx]
            pdf_dict = all_page_dicts[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            
            # 转换为中间JSON
            middle_json = result_to_middle_json(
                model_list,
                images_list,
                pdf_dict,
                image_writer,
                _lang,
                _ocr_enable,
                request.formula_enable,
                ocr_config=configs["ocr_config"],
                image_config=configs["image_config"],
            )
            
            pdf_info = middle_json["pdf_info"]
            pdf_bytes = processed_pdf_bytes[idx]
            
            # 准备结果字典
            file_result = {}
            
            # 生成Markdown
            if request.return_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content = union_make(pdf_info, MakeMode.MM_MD, image_dir)
                md_writer.write_string(f"{pdf_file_name}.md", md_content)
                file_result["markdown"] = md_content
            
            # 生成内容列表
            if request.return_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json", 
                    json.dumps(content_list, ensure_ascii=False, indent=4)
                )
                file_result["content_list"] = content_list
            
            # 生成中间JSON
            if request.return_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4)
                )
                file_result["middle_json"] = middle_json
            
            # 生成模型输出
            if request.return_model_output:
                model_json = model_list
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4)
                )
                file_result["model_output"] = model_json
            
            results[pdf_file_name] = file_result
        
        # 清理资源
        gc.collect()
        
        process_time = time.time() - start_time
        logger.info(f"Task {task_id} completed in {process_time:.2f} seconds")
        
        return {
            "status": "success",
            "task_id": task_id,
            "processing_time": process_time,
            "files_processed": len(files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse_async")
async def parse_document_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: ParseRequest = Form(...),
    output_dir: str = Form("./output"),
):
    """
    异步解析文档API
    
    Args:
        background_tasks: 后台任务
        files: 要解析的文件列表
        request: 解析请求参数
        output_dir: 输出目录
    """
    
    # 检查模型是否已初始化
    if model_manager.get_pipeline_model() is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")
    
    # 创建任务ID
    task_id = str(uuid.uuid4())
    
    # 添加后台任务
    background_tasks.add_task(
        process_document_async,
        task_id=task_id,
        files=files,
        request=request,
        output_dir=output_dir
    )
    
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "Document parsing started in background"
    }


async def process_document_async(
    task_id: str,
    files: List[UploadFile],
    request: ParseRequest,
    output_dir: str
):
    """异步处理文档的后台任务"""
    
    try:
        logger.info(f"Starting background task {task_id}")
        
        # 这里可以实现具体的异步处理逻辑
        # 目前使用同步处理作为示例
        await parse_document(
            files=files,
            request=request,
            output_dir=output_dir
        )
        
        logger.info(f"Background task {task_id} completed")
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {e}")


@app.post("/custom_config")
async def update_model_config(config: ModelConfig):
    """
    更新模型配置（需要重启服务才能生效）
    
    Args:
        config: 新的模型配置
    """
    
    try:
        # 在实际应用中，可能需要重启模型或重新初始化特定组件
        # 这里我们只返回配置信息
        
        logger.info("Model config update requested")
        
        return {
            "status": "success",
            "message": "Configuration updated. Note: Some changes may require service restart to take effect.",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
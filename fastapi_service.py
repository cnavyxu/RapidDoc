"""RapidDoc FastAPI服务。

修复点：
- 避免接口调用过程中的二次初始化：统一使用 rapid_doc.backend.pipeline.pipeline_analyze.ModelSingleton 进行模型缓存与预热
- 增加 OCR 独立开关参数（ocr_enable）
- 返回各模块耗时统计（timings / pipeline_metrics）
- 优化文本识别耗时：默认 use_det_mode=auto、OCR-rec 默认 batch_num 更大、默认关闭 tqdm
"""

import os
import gc
import json
import time
import asyncio
import uuid
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loguru import logger
import uvicorn

# 设置设备环境变量
os.environ["MINERU_DEVICE_MODE"] = os.getenv("MINERU_DEVICE_MODE", "cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rapid_doc.backend.pipeline.pipeline_analyze import (
    doc_analyze as pipeline_doc_analyze,
    ModelSingleton as PipelineModelSingleton,
)
from rapid_doc.cli.common import prepare_env, convert_pdf_bytes_to_bytes_by_pypdfium2
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from rapid_doc.utils.enum_class import MakeMode

from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.formula.rapid_formula_self import (
    ModelType as FormulaModelType,
    EngineType as FormulaEngineType,
)
from rapid_doc.model.table.rapid_table_self import (
    ModelType as TableModelType,
    EngineType as TableEngineType,
)
from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType


class ParseRequest(BaseModel):
    """解析请求模型"""

    parse_method: str = "auto"  # auto/txt/ocr

    # 模块开关
    ocr_enable: Optional[bool] = None  # True 强制OCR；False 强制TXT；None 走 parse_method
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
    layout_conf_thresh: float = 0.4
    layout_batch_num: int = 1

    # OCR配置
    ocr_det_model_path: str = "./models/det_server.onnx"
    ocr_rec_model_path: str = "./models/rec_server.onnx"
    ocr_cls_model_path: str = "./models/cls.onnx"
    ocr_rec_keys_path: str = "./models/ppocrv5_dict.txt"

    # OCR-rec 批量大小（文本识别加速的关键参数）
    ocr_rec_batch_num: int = 8

    # OCR-det 批量大小
    ocr_det_rec_batch_num: int = 8

    # auto: 非OCR页尽量走PDF-det；ocr: 强制走OCR-det；txt: 尽量走PDF-det
    use_det_mode: str = "auto"

    # 是否开启 tqdm 进度条（服务端默认关闭，减少日志与额外开销）
    tqdm_enable: bool = False

    # Formula配置
    formula_engine_type: str = "torch"
    formula_level: int = 0
    formula_batch_num: int = 1
    formula_model_path: str = "./models/pp_formulanet_plus_m.pth"
    formula_dict_path: str = "./models/pp_formulanet_plus_m_inference.yml"

    # Table配置
    table_force_ocr: bool = False
    table_skip_text_in_image: bool = True
    table_use_img2table: bool = False
    table_model_type: str = "unet_slanet_plus"
    table_use_word_box: bool = True
    table_use_compare_table: bool = False
    table_formula_enable: bool = False
    table_image_enable: bool = False
    table_extract_original_image: bool = False
    table_cls_model_path: str = "./models/paddle_cls.onnx"
    table_unet_model_path: str = "./models/unet.onnx"
    table_slanet_plus_model_path: str = "./models/slanet-plus.onnx"
    table_engine_type: str = "onnxruntime"


def build_pipeline_configs(model_config: ModelConfig) -> Dict[str, Any]:
    os.environ["MINERU_MODEL_SOURCE"] = "local"

    layout_config = {
        "model_type": LayoutModelType.PP_DOCLAYOUT_PLUS_L,
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
        "engine_type": OCREngineType.ONNXRUNTIME,
        "Det.rec_batch_num": model_config.ocr_det_rec_batch_num,
        "use_det_mode": model_config.use_det_mode,
        "tqdm_enable": model_config.tqdm_enable,
    }

    formula_config = {
        "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
        "engine_type": FormulaEngineType.TORCH,
        "formula_level": model_config.formula_level,
        "batch_num": model_config.formula_batch_num,
        "model_dir_or_path": model_config.formula_model_path,
        "dict_keys_path": model_config.formula_dict_path,
    }

    table_config = {
        "force_ocr": model_config.table_force_ocr,
        "skip_text_in_image": model_config.table_skip_text_in_image,
        "use_img2table": model_config.table_use_img2table,
        "model_type": TableModelType.UNET_SLANET_PLUS,
        "use_word_box": model_config.table_use_word_box,
        "use_compare_table": model_config.table_use_compare_table,
        "table_formula_enable": model_config.table_formula_enable,
        "table_image_enable": model_config.table_image_enable,
        "extract_original_image": model_config.table_extract_original_image,
        "cls.model_type": TableModelType.PADDLE_CLS,
        "cls.model_dir_or_path": model_config.table_cls_model_path,
        "unet.model_dir_or_path": model_config.table_unet_model_path,
        "slanet_plus.model_dir_or_path": model_config.table_slanet_plus_model_path,
        "engine_type": TableEngineType.ONNXRUNTIME,
    }

    image_config = {
        "extract_original_image": False,
        "extract_original_image_iou_thresh": 0.5,
    }

    return {
        "layout_config": layout_config,
        "ocr_config": ocr_config,
        "formula_config": formula_config,
        "table_config": table_config,
        "image_config": image_config,
    }


class ServiceState:
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.configs: Dict[str, Any] = build_pipeline_configs(self.model_config)
        self.pipeline_singleton = PipelineModelSingleton()

    def warmup(self):
        self.pipeline_singleton.get_model(
            lang="ch",
            formula_enable=True,
            table_enable=True,
            layout_config=self.configs["layout_config"],
            ocr_config=self.configs["ocr_config"],
            formula_config=self.configs["formula_config"],
            table_config=self.configs["table_config"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RapidDoc FastAPI service...")

    app.state.service_state = ServiceState()

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, app.state.service_state.warmup)
        logger.info("Models warmed up successfully")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        logger.warning("Service will start without warm models. Models will be loaded on first request.")

    yield

    logger.info("Shutting down RapidDoc FastAPI service...")


app = FastAPI(
    title="RapidDoc FastAPI Service",
    description="基于RapidDoc的文档解析API，支持模块化配置和耗时统计",
    version="1.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _effective_parse_method(req: ParseRequest) -> str:
    if req.ocr_enable is True:
        return "ocr"
    if req.ocr_enable is False:
        return "txt"
    return req.parse_method


async def _read_and_normalize_files(files: List[UploadFile]) -> Tuple[List[str], List[bytes]]:
    pdf_file_names: List[str] = []
    pdf_bytes_list: List[bytes] = []

    for file in files:
        file_suffix = Path(file.filename).suffix.lower()
        if file_suffix not in [
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
        ]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_suffix}")

        file_content = await file.read()
        file_name = Path(file.filename).stem

        if file_suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
            from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes

            file_content = images_bytes_to_pdf_bytes(file_content)

        pdf_file_names.append(file_name)
        pdf_bytes_list.append(file_content)

    return pdf_file_names, pdf_bytes_list


def _convert_pdf_bytes(
    pdf_bytes_list: List[bytes], start_page_id: int, end_page_id: Optional[int]
) -> List[bytes]:
    processed: List[bytes] = []
    for pdf_bytes in pdf_bytes_list:
        processed.append(
            convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        )
    return processed


@app.get("/health")
async def health_check():
    singleton = PipelineModelSingleton()
    loaded = len(getattr(singleton, "_models", {})) > 0
    return {
        "status": "healthy",
        "service": "RapidDoc FastAPI Service",
        "version": "1.0.1",
        "models_loaded": loaded,
    }


@app.get("/config")
async def get_config():
    state: ServiceState = app.state.service_state
    return {
        "status": "success",
        "model_config": state.model_config.model_dump(),
        "pipeline_configs": state.configs,
    }


@app.post("/parse")
async def parse_document(
    files: List[UploadFile] = File(...),
    request: str = Form(...),
    output_dir: str = Form("./output"),
):
    req = ParseRequest(**json.loads(request))
    state: ServiceState = app.state.service_state

    task_id = str(uuid.uuid4())
    task_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    timings: Dict[str, Any] = {"task_id": task_id}
    t_total = time.perf_counter()

    try:
        t0 = time.perf_counter()
        pdf_file_names, pdf_bytes_list = await _read_and_normalize_files(files)
        timings["upload_read"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        processed_pdf_bytes = _convert_pdf_bytes(
            pdf_bytes_list, req.start_page_id, req.end_page_id
        )
        timings["pdf_convert"] = time.perf_counter() - t0

        effective_parse_method = _effective_parse_method(req)

        checkbox_config = {"checkbox_enable": req.checkbox_enable}

        t0 = time.perf_counter()
        (
            infer_results,
            all_image_lists,
            all_page_dicts,
            lang_list,
            ocr_enabled_list,
            pipeline_metrics,
        ) = pipeline_doc_analyze(
            processed_pdf_bytes,
            lang_list=[req.lang] * len(processed_pdf_bytes),
            parse_method=effective_parse_method,
            formula_enable=req.formula_enable,
            table_enable=req.table_enable,
            layout_config=state.configs["layout_config"],
            ocr_config=state.configs["ocr_config"],
            formula_config=state.configs["formula_config"],
            table_config=state.configs["table_config"],
            checkbox_config=checkbox_config,
            return_metrics=True,
        )
        timings["doc_analyze"] = time.perf_counter() - t0

        results: Dict[str, Any] = {}
        per_file_timings: Dict[str, Any] = {}

        for idx, model_list in enumerate(infer_results):
            pdf_file_name = pdf_file_names[idx]

            local_image_dir, local_md_dir = prepare_env(
                task_output_dir, pdf_file_name, effective_parse_method
            )
            image_writer, md_writer = (
                FileBasedDataWriter(local_image_dir),
                FileBasedDataWriter(local_md_dir),
            )

            images_list = all_image_lists[idx]
            pdf_dict = all_page_dicts[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]

            file_timing: Dict[str, float] = {}

            t1 = time.perf_counter()
            middle_json = result_to_middle_json(
                model_list,
                images_list,
                pdf_dict,
                image_writer,
                _lang,
                _ocr_enable,
                req.formula_enable,
                ocr_config=state.configs["ocr_config"],
                image_config=state.configs["image_config"],
            )
            file_timing["middle_json"] = time.perf_counter() - t1

            pdf_info = middle_json["pdf_info"]

            file_result: Dict[str, Any] = {}

            if req.return_md:
                t1 = time.perf_counter()
                image_dir = str(os.path.basename(local_image_dir))
                md_content = union_make(pdf_info, MakeMode.MM_MD, image_dir)
                md_writer.write_string(f"{pdf_file_name}.md", md_content)
                file_result["markdown"] = md_content
                file_timing["make_md"] = time.perf_counter() - t1

            if req.return_content_list:
                t1 = time.perf_counter()
                image_dir = str(os.path.basename(local_image_dir))
                content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )
                file_result["content_list"] = content_list
                file_timing["make_content_list"] = time.perf_counter() - t1

            if req.return_middle_json:
                t1 = time.perf_counter()
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )
                file_result["middle_json"] = middle_json
                file_timing["dump_middle_json"] = time.perf_counter() - t1

            if req.return_model_output:
                t1 = time.perf_counter()
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_list, ensure_ascii=False, indent=4),
                )
                file_result["model_output"] = model_list
                file_timing["dump_model_output"] = time.perf_counter() - t1

            results[pdf_file_name] = file_result
            per_file_timings[pdf_file_name] = file_timing

        gc.collect()

        timings["total"] = time.perf_counter() - t_total

        return {
            "status": "success",
            "task_id": task_id,
            "parse_method": effective_parse_method,
            "files_processed": len(files),
            "timings": timings,
            "pipeline_metrics": pipeline_metrics,
            "per_file_timings": per_file_timings,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse_async")
async def parse_document_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: str = Form(...),
    output_dir: str = Form("./output"),
):
    req = ParseRequest(**json.loads(request))

    task_id = str(uuid.uuid4())

    # 先把文件读到内存，避免 BackgroundTask 执行时 UploadFile 已关闭
    pdf_file_names, pdf_bytes_list = await _read_and_normalize_files(files)

    background_tasks.add_task(
        _parse_in_background,
        task_id=task_id,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        req=req,
        output_dir=output_dir,
    )

    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "Document parsing started in background",
    }


def _parse_in_background(
    task_id: str,
    pdf_file_names: List[str],
    pdf_bytes_list: List[bytes],
    req: ParseRequest,
    output_dir: str,
):
    state: ServiceState = app.state.service_state

    try:
        task_output_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)

        processed_pdf_bytes = _convert_pdf_bytes(
            pdf_bytes_list, req.start_page_id, req.end_page_id
        )

        effective_parse_method = _effective_parse_method(req)

        checkbox_config = {"checkbox_enable": req.checkbox_enable}

        (
            infer_results,
            all_image_lists,
            all_page_dicts,
            lang_list,
            ocr_enabled_list,
        ) = pipeline_doc_analyze(
            processed_pdf_bytes,
            lang_list=[req.lang] * len(processed_pdf_bytes),
            parse_method=effective_parse_method,
            formula_enable=req.formula_enable,
            table_enable=req.table_enable,
            layout_config=state.configs["layout_config"],
            ocr_config=state.configs["ocr_config"],
            formula_config=state.configs["formula_config"],
            table_config=state.configs["table_config"],
            checkbox_config=checkbox_config,
            return_metrics=False,
        )

        for idx, model_list in enumerate(infer_results):
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(
                task_output_dir, pdf_file_name, effective_parse_method
            )
            image_writer, md_writer = (
                FileBasedDataWriter(local_image_dir),
                FileBasedDataWriter(local_md_dir),
            )

            images_list = all_image_lists[idx]
            pdf_dict = all_page_dicts[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]

            middle_json = result_to_middle_json(
                model_list,
                images_list,
                pdf_dict,
                image_writer,
                _lang,
                _ocr_enable,
                req.formula_enable,
                ocr_config=state.configs["ocr_config"],
                image_config=state.configs["image_config"],
            )

            if req.return_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content = union_make(middle_json["pdf_info"], MakeMode.MM_MD, image_dir)
                md_writer.write_string(f"{pdf_file_name}.md", md_content)

        gc.collect()
        logger.info(f"Background task {task_id} completed")

    except Exception as e:
        logger.exception(f"Background task {task_id} failed: {e}")


@app.post("/custom_config")
async def update_model_config(config: ModelConfig):
    state: ServiceState = app.state.service_state

    state.model_config = config
    state.configs = build_pipeline_configs(config)

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, state.warmup)
    except Exception as e:
        logger.error(f"Warmup after config update failed: {e}")

    return {
        "status": "success",
        "message": "Configuration updated and warmup triggered.",
        "model_config": config.model_dump(),
        "pipeline_configs": state.configs,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

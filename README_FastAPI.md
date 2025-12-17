# RapidDoc FastAPI Service

基于RapidDoc的异步文档解析API服务，支持模型预初始化、模块化配置和异步处理。

## 特性

- ✅ **模型预加载**: 应用启动时一次性初始化所有模型，避免重复初始化
- ✅ **异步处理**: 支持同步和异步两种文档解析方式  
- ✅ **模块化配置**: 可独立配置Layout、OCR、Formula、Table等模块
- ✅ **批量处理**: 支持多文档并发处理
- ✅ **多种输出**: 支持Markdown、JSON、内容列表等多种输出格式
- ✅ **健康监控**: 提供健康检查和配置查询接口

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install fastapi uvicorn pydantic python-multipart

# 安装RapidDoc依赖
pip install -e .
```

### 2. 启动服务

```bash
# 开发模式启动
python fastapi_service.py

# 或使用uvicorn
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 测试服务

```bash
# 运行测试脚本
python test_fastapi_service.py
```

## API接口

### 健康检查
```http
GET /health
```

### 获取配置
```http
GET /config
```

### 同步文档解析
```http
POST /parse
Content-Type: multipart/form-data

files: 文件列表
request: ParseRequest JSON字符串
output_dir: 输出目录
```

### 异步文档解析
```http
POST /parse_async
Content-Type: multipart/form-data

files: 文件列表  
request: ParseRequest JSON字符串
output_dir: 输出目录
```

### 更新配置
```http
POST /custom_config
Content-Type: application/json

{
  "layout_model_type": "PP_DOCLAYOUT_PLUS_L",
  "formula_enable": true,
  "table_enable": true,
  ...
}
```

## 数据模型

### ParseRequest
```python
class ParseRequest(BaseModel):
    parse_method: str = "auto"        # 解析方法: auto, txt, ocr
    formula_enable: bool = True       # 是否启用公式识别
    table_enable: bool = True         # 是否启用表格识别
    checkbox_enable: bool = False     # 是否启用复选框检测
    start_page_id: int = 0           # 起始页码
    end_page_id: Optional[int] = None # 结束页码
    lang: str = "ch"                 # 语言
    
    # 返回格式控制
    return_md: bool = True           # 返回Markdown内容
    return_middle_json: bool = False # 返回中间JSON
    return_model_output: bool = False # 返回模型输出
    return_content_list: bool = False # 返回内容列表
```

### ModelConfig
```python
class ModelConfig(BaseModel):
    # Layout配置
    layout_model_type: str = "PP_DOCLAYOUT_PLUS_L"
    layout_conf_thresh: float = 0.4
    layout_batch_num: int = 1
    
    # OCR配置
    ocr_engine_type: str = "ONNXRUNTIME"
    ocr_det_model_path: str = "./models/det_server.onnx"
    ocr_rec_model_path: str = "./models/rec_server.onnx"
    use_det_mode: str = "ocr"  # auto, txt, ocr
    
    # Formula配置
    formula_model_type: str = "PP_FORMULANET_PLUS_M"
    formula_level: int = 0
    formula_batch_num: int = 1
    
    # Table配置
    table_force_ocr: bool = False
    table_model_type: str = "UNET_SLANET_PLUS"
    table_use_word_box: bool = True
    ...
```

## 使用示例

### Python客户端示例

```python
import aiohttp
import asyncio
from pathlib import Path

async def parse_document():
    # 准备文件
    files = [
        ("files", ("document.pdf", open("document.pdf", "rb"), "application/pdf"))
    ]
    
    # 准备请求参数
    request_data = {
        "request": {
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_content_list": True
        }
    }
    
    # 调用API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/parse",
            data=request_data,
            files=files
        ) as response:
            result = await response.json()
            print(f"解析结果: {result}")

# 运行异步函数
asyncio.run(parse_document())
```

### cURL示例

```bash
# 基础文档解析
curl -X POST "http://localhost:8000/parse" \
  -H "accept: application/json" \
  -F "files=@document.pdf" \
  -F 'request={
    "parse_method": "auto",
    "formula_enable": true,
    "table_enable": true,
    "return_md": true,
    "return_content_list": false
  }'

# 异步解析
curl -X POST "http://localhost:8000/parse_async" \
  -H "accept: application/json" \
  -F "files=@document.pdf" \
  -F 'request={
    "parse_method": "ocr",
    "formula_enable": false,
    "table_enable": true
  }'
```

## 配置说明

### 环境变量

```bash
# 设置设备模式 (cpu, cuda, cuda:0, cuda:1等)
export MINERU_DEVICE_MODE="cuda"

# 设置模型源 (local, modelscope)
export MINERU_MODEL_SOURCE="local"

# 设置模型目录
export RAPID_MODELS_DIR="/path/to/models"
```

### 模型文件准备

确保以下模型文件存在：

```
./models/
├── pp_doclayout_plus_l.onnx     # Layout模型
├── det_server.onnx              # OCR检测模型
├── rec_server.onnx              # OCR识别模型
├── cls.onnx                     # OCR分类模型
├── ppocrv5_dict.txt             # OCR字典
├── pp_formulanet_plus_m.pth     # Formula模型
├── paddle_cls.onnx             # Table分类模型
├── unet.onnx                   # Table UNET模型
└── slanet-plus.onnx            # Table SLANET模型
```

## 部署

### Docker部署

创建Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN pip install fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 生产部署

```bash
# 使用Gunicorn部署
pip install gunicorn
gunicorn fastapi_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 或使用Docker Compose
docker-compose up -d
```

## 监控和日志

### 健康检查
```bash
curl http://localhost:8000/health
```

### 配置查询
```bash
curl http://localhost:8000/config
```

### 日志查看
服务启动时会输出详细的初始化日志，处理过程中也会记录进度和错误信息。

## 故障排除

### 常见问题

1. **模型初始化失败**
   - 检查模型文件是否存在
   - 确认设备模式设置正确
   - 查看内存是否充足

2. **导入错误**
   - 确保安装了所有依赖: `pip install -e .`
   - 检查Python路径设置

3. **内存不足**
   - 调整批处理大小
   - 使用GPU设备
   - 分批处理大文档

4. **端口占用**
   - 修改端口号: `uvicorn --port 8001`
   - 检查其他服务是否占用端口

### 性能调优

1. **批处理优化**: 调整`batch_num`参数
2. **内存优化**: 使用GPU设备减少内存占用
3. **并发控制**: 配置合适的worker数量

## 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题，请通过GitHub Issues联系我们。
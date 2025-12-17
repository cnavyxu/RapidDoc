# RapidDoc API Server

基于 FastAPI 的高性能文档解析服务，支持模型初始化持久化、异步调用和模块化配置。

## ✨ 核心特性

- **🚀 模型持久化**: 模型只需初始化一次，避免重复加载
- **⚡ 异步处理**: 支持真正的异步文档解析
- **🔧 模块化设计**: 支持Layout、OCR、Formula、Table等模块独立配置
- **📊 高性能**: 基于批处理和单例模式的优化架构
- **🔗 多种接口**: RESTful API + 便捷的Python客户端
- **📁 灵活输出**: 支持Markdown、中间JSON、模型输出等多种格式

## 🏗️ 架构设计

```
RapidDoc API Server
├── FastAPI 应用层
├── 模型服务管理器 (ModelServiceManager)
├── 模型单例池 (ModelSingleton)
├── 异步文档解析器
└── 结果处理模块
```

## 📋 系统要求

- Python 3.8+
- CUDA 11.8+ (GPU加速，可选)
- 内存: 建议8GB+
- 存储: 模型文件约5-10GB

## 🛠️ 安装部署

### 1. 克隆项目

```bash
git clone <repository-url>
cd rapid_doc
```

### 2. 安装依赖

```bash
# 安装API服务依赖
pip install -r requirements-api.txt

# 或使用虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements-api.txt
```

### 3. 模型文件准备

```bash
# 确保模型文件存在于 ./models/ 目录
mkdir -p models
# 下载或放置所需的模型文件
# - pp_doclayout_plus_l.onnx
# - det_server.onnx, rec_server.onnx, cls.onnx
# - pp_formulanet_plus_m.pth
# - paddle_cls.onnx, unet.onnx, slanet-plus.onnx
# - ppocrv5_dict.txt, pp_formulanet_plus_m_inference.yml
```

### 4. 启动服务

```bash
# 使用启动脚本 (推荐)
chmod +x start_api.sh
./start_api.sh

# 或直接启动
python3 api_server.py

# 或使用uvicorn (生产模式)
uvicorn api_server:app --host 0.0.0.0 --port 8888 --workers 1
```

### 5. 验证服务

```bash
# 健康检查
curl http://localhost:8888/health

# 获取服务状态
curl http://localhost:8888/status
```

## 📖 API 文档

### 基本端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/status` | 服务状态 |
| GET | `/docs` | API文档 |
| GET | `/configs` | 配置列表 |
| POST | `/init` | 初始化模型配置 |
| POST | `/parse` | 解析文档 |
| DELETE | `/configs/{config_id}` | 删除配置 |

### 详细API说明

#### 1. 初始化模型配置

**请求:**
```bash
POST /init
Content-Type: application/json

{
    "layout_model_type": "PP_DOCLAYOUT_PLUS_L",
    "ocr_engine_type": "ONNXRUNTIME", 
    "formula_model_type": "PP_FORMULANET_PLUS_M",
    "table_model_type": "UNET_SLANET_PLUS",
    "device_mode": "cuda",
    "conf_thresh": 0.4,
    "use_det_mode": "ocr"
}
```

**响应:**
```json
{
    "status": "success",
    "message": "模型配置初始化成功，耗时: 2.34秒",
    "config_id": "550e8400-e29b-41d4-a716-446655440000",
    "modules": {
        "layout": true,
        "ocr": true,
        "formula": true,
        "table": true,
        "checkbox": true,
        "image": true
    }
}
```

#### 2. 解析文档

**请求:**
```bash
POST /parse
Content-Type: multipart/form-data

config_id: 550e8400-e29b-41d4-a716-446655440000
files: @document.pdf
output_dir: ./output
parse_method: auto
formula_enable: true
table_enable: true
lang_list: ["ch"]
return_md: true
return_middle_json: false
return_model_output: false
return_content_list: false
return_images: false
response_format_zip: false
```

**响应:**
```json
{
    "status": "success",
    "config_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time": 3.45,
    "files_processed": 1,
    "results": {
        "document": {
            "md_content": "# 文档标题\n\n这里是解析的Markdown内容...",
            "middle_json": null,
            "model_output": null,
            "content_list": null,
            "images": []
        }
    },
    "output_dir": "./output/550e8400-e29b-41d4-a716-446655440000"
}
```

## 🐍 Python 客户端使用

### 安装客户端依赖

```bash
pip install aiohttp
```

### 基础使用示例

```python
import asyncio
from api_client_example import RapidDocClient

async def main():
    # 创建客户端
    client = RapidDocClient("http://localhost:8888")
    
    # 1. 初始化模型配置
    config_id = await client.init_model_config(
        device_mode="cuda",
        conf_thresh=0.4
    )
    
    # 2. 解析文档
    result = await client.parse_documents(
        files=["document.pdf"],
        output_dir="./output",
        return_md=True,
        return_middle_json=True
    )
    
    print(f"处理时间: {result['processing_time']}秒")

asyncio.run(main())
```

### 高级使用示例

```python
# 批量处理
files = ["doc1.pdf", "doc2.pdf", "doc3.png"]
result = await client.parse_documents(
    files=files,
    lang_list=["ch", "ch", "en"],
    table_enable=True,
    formula_enable=True,
    return_content_list=True
)

# 特定页面解析
result = await client.parse_documents(
    files=["long_document.pdf"],
    start_page_id=0,
    end_page_id=10,
    return_images=True
)
```

## ⚙️ 配置参数详解

### Layout模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `layout_model_type` | 版面模型类型 | PP_DOCLAYOUT_PLUS_L |
| `conf_thresh` | 置信度阈值 | 0.4 |
| `batch_num` | 批处理大小 | 1 |

### OCR模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ocr_engine_type` | OCR引擎类型 | ONNXRUNTIME |
| `use_det_mode` | 检测模式 | ocr |
| `rec_batch_num` | 识别批处理大小 | 1 |

### Formula模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `formula_model_type` | 公式模型类型 | PP_FORMULANET_PLUS_M |
| `formula_level` | 公式识别等级 | 1 |

### Table模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `table_model_type` | 表格模型类型 | UNET_SLANET_PLUS |
| `force_ocr` | 强制OCR | False |
| `use_word_box` | 使用单词框 | True |

## 🔧 性能优化

### 1. 模型初始化优化

- 使用单例模式避免重复加载模型
- 预热模型提高首次响应速度
- 按需启用模块减少资源占用

### 2. 异步处理优化

- 支持真正的异步文档解析
- 批处理提高处理效率
- 内存管理和垃圾回收优化

### 3. 资源配置优化

```bash
# 设置批处理大小
export MINERU_MIN_BATCH_INFERENCE_SIZE=512

# 设置GPU内存
export MINERU_VIRTUAL_VRAM_SIZE=16

# 模型源设置
export MINERU_MODEL_SOURCE="local"  # 本地模型
# export MINERU_MODEL_SOURCE="modelscope"  # ModelScope下载
```

## 🐳 Docker 部署

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 8888

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8888"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  rapid-doc-api:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - MINERU_DEVICE_MODE=cuda
      - MINERU_MODEL_SOURCE=local
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🧪 测试和验证

### 1. 运行内置示例

```bash
# 运行客户端示例
python3 api_client_example.py

# 选择示例类型进行测试
```

### 2. 性能基准测试

```bash
# 单文件处理测试
curl -X POST "http://localhost:8888/parse" \
  -F "config_id=$CONFIG_ID" \
  -F "files=@test.pdf" \
  -F "return_md=true"

# 批量处理测试
# 参见 api_client_example.py 中的性能测试示例
```

### 3. 压力测试

```bash
# 使用 ab (Apache Bench)
ab -n 100 -c 10 -p test.pdf -T "multipart/form-data" \
   http://localhost:8888/parse

# 使用 wrk
wrk -t12 -c400 -d30s --script=test.lua http://localhost:8888/parse
```

## 🚨 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型文件路径
   ls -la models/
   
   # 检查环境变量
   echo $MINERU_MODEL_SOURCE
   ```

2. **GPU内存不足**
   ```bash
   # 调整批处理大小
   export MINERU_MIN_BATCH_INFERENCE_SIZE=64
   
   # 切换到CPU模式
   export MINERU_DEVICE_MODE=cpu
   ```

3. **端口占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8888
   
   # 杀死进程
   kill -9 <PID>
   ```

### 日志调试

```bash
# 查看服务日志
tail -f rapid_doc_api.log

# 启用详细日志
export LOG_LEVEL=DEBUG
python3 api_server.py
```

## 📈 监控和运维

### 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

RESPONSE=$(curl -s http://localhost:8888/health)
if [[ $RESPONSE == *"healthy"* ]]; then
    echo "✅ 服务正常运行"
else
    echo "❌ 服务异常"
    # 发送告警
fi
```

### 自动重启脚本

```bash
#!/bin/bash
# auto_restart.sh

while true; do
    if ! curl -s http://localhost:8888/health > /dev/null; then
        echo "$(date): 服务异常，重启中..."
        pkill -f "uvicorn.*api_server"
        nohup uvicorn api_server:app --host 0.0.0.0 --port 8888 > api.log 2>&1 &
    fi
    sleep 30
done
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用与 RapidDoc 相同的许可证。

## 🆘 支持

如有问题或建议，请：

1. 查看 [Issues](../../issues) 页面
2. 创建新的 Issue
3. 联系维护团队

---

**RapidDoc API Server** - 让文档解析更简单、更高效！ 🚀
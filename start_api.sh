#!/bin/bash

# RapidDoc API Server 启动脚本
# ================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示欢迎信息
echo -e "${BLUE}"
echo "=========================================="
echo "     RapidDoc API Server"
echo "=========================================="
echo -e "${NC}"

# 检查Python版本
print_info "检查Python环境..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    print_success "找到Python: $python_version"
else
    print_error "未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "未检测到虚拟环境，建议使用虚拟环境运行"
    read -p "是否继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "虚拟环境: $VIRTUAL_ENV"
fi

# 安装依赖
print_info "检查依赖包..."
if [[ -f "requirements-api.txt" ]]; then
    print_info "安装API依赖..."
    pip install -r requirements-api.txt
    print_success "依赖安装完成"
else
    print_error "未找到requirements-api.txt文件"
    exit 1
fi

# 设置环境变量
export MINERU_MODEL_SOURCE=${MINERU_MODEL_SOURCE:-"local"}
export MINERU_DEVICE_MODE=${MINERU_DEVICE_MODE:-"cuda"}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

print_info "环境变量配置:"
echo "  MINERU_MODEL_SOURCE: $MINERU_MODEL_SOURCE"
echo "  MINERU_DEVICE_MODE: $MINERU_DEVICE_MODE"
echo "  PYTHONPATH: $PYTHONPATH"

# 创建必要的目录
print_info "创建必要目录..."
mkdir -p output
mkdir -p logs
mkdir -p temp

# 检查端口
PORT=${PORT:-8888}
print_info "服务端口: $PORT"

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "端口 $PORT 已被占用"
    read -p "是否继续？(可能覆盖现有服务): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动模式选择
print_info "选择启动模式:"
echo "1) 开发模式 (reload=True, debug)"
echo "2) 生产模式 (高性能, 推荐)"
read -p "请选择 [1-2]: " -n 1 -r
echo

case $REPLY in
    1)
        print_info "开发模式启动..."
        export ENV_MODE="development"
        python3 api_server.py
        ;;
    2)
        print_info "生产模式启动..."
        export ENV_MODE="production"
        uvicorn api_server:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers 1 \
            --log-level info \
            --access-log
        ;;
    *)
        print_error "无效选择，使用生产模式"
        export ENV_MODE="production"
        uvicorn api_server:app \
            --host 0.0.0.0 \
            --port $PORT \
            --workers 1 \
            --log-level info
        ;;
esac

print_success "RapidDoc API Server 启动完成!"
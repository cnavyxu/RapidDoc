#!/bin/bash

# RapidDoc FastAPI服务启动脚本

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

# 显示帮助信息
show_help() {
    echo "RapidDoc FastAPI服务启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -d, --dev           开发模式启动 (自动重载)"
    echo "  -p, --port PORT     指定端口号 (默认: 8000)"
    echo "  --host HOST         指定主机地址 (默认: 0.0.0.0)"
    echo "  --workers NUM       指定工作进程数 (默认: 1)"
    echo "  --test              运行测试并退出"
    echo "  --check             检查环境和依赖"
    echo ""
    echo "示例:"
    echo "  $0 --dev                    # 开发模式启动"
    echo "  $0 --port 8080              # 指定端口启动"
    echo "  $0 --workers 4              # 多进程启动"
    echo "  $0 --test                   # 运行测试"
}

# 检查环境和依赖
check_environment() {
    print_info "检查环境和依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查是否在虚拟环境中
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "建议在虚拟环境中运行"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # 检查依赖
    python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null || {
        print_error "缺少必要的Python依赖"
        print_info "请运行: pip install fastapi uvicorn pydantic python-multipart"
        exit 1
    }
    
    # 检查RapidDoc是否已安装
    python3 -c "import rapid_doc" 2>/dev/null || {
        print_error "RapidDoc未安装"
        print_info "请运行: pip install -e ."
        exit 1
    }
    
    print_success "环境检查通过"
}

# 运行测试
run_tests() {
    print_info "运行服务测试..."
    
    if [[ -f "test_fastapi_service.py" ]]; then
        python3 test_fastapi_service.py
    else
        print_error "测试文件不存在"
        exit 1
    fi
}

# 启动服务
start_service() {
    local dev_mode=$1
    local port=$2
    local host=$3
    local workers=$4
    
    print_info "启动RapidDoc FastAPI服务..."
    print_info "主机: $host"
    print_info "端口: $port"
    print_info "工作进程: $workers"
    
    if [[ "$dev_mode" == "true" ]]; then
        print_info "开发模式启动 (自动重载)"
        uvicorn fastapi_service:app --host $host --port $port --reload
    else
        print_info "生产模式启动"
        if command -v gunicorn &> /dev/null; then
            print_info "使用Gunicorn启动"
            gunicorn fastapi_service:app -w $workers -k uvicorn.workers.UvicornWorker --bind $host:$port
        else
            print_info "使用Uvicorn启动"
            uvicorn fastapi_service:app --host $host --port $port --workers $workers
        fi
    fi
}

# 主函数
main() {
    local dev_mode="false"
    local port="8000"
    local host="0.0.0.0"
    local workers="1"
    local run_test="false"
    local check_only="false"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dev)
                dev_mode="true"
                shift
                ;;
            -p|--port)
                port="$2"
                shift 2
                ;;
            --host)
                host="$2"
                shift 2
                ;;
            --workers)
                workers="$2"
                shift 2
                ;;
            --test)
                run_test="true"
                shift
                ;;
            --check)
                check_only="true"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果只是检查环境
    if [[ "$check_only" == "true" ]]; then
        check_environment
        exit 0
    fi
    
    # 如果要运行测试
    if [[ "$run_test" == "true" ]]; then
        check_environment
        run_tests
        exit 0
    fi
    
    # 检查环境
    check_environment
    
    # 启动服务
    start_service "$dev_mode" "$port" "$host" "$workers"
}

# 如果直接执行脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
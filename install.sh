#!/bin/bash
# VisionCore 自动安装脚本
# 适用于Linux开发板（如RK3588）

set -e  # 遇到错误立即退出

echo "========================================"
echo "VisionCore 无界面视觉系统 安装脚本"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印彩色信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "请使用root权限运行此脚本: sudo ./install.sh"
        exit 1
    fi
}

# 检查系统架构
check_architecture() {
    arch=$(uname -m)
    print_info "检测到系统架构: $arch"
    
    if [[ "$arch" != "aarch64" && "$arch" != "x86_64" ]]; then
        print_warning "未经测试的架构: $arch"
    fi
}

# 安装系统依赖
install_dependencies() {
    print_info "安装系统依赖..."
    
    # 更新包管理器
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python3 python3-pip python3-venv python3-dev
        apt-get install -y build-essential cmake pkg-config
        apt-get install -y libopencv-dev python3-opencv
        apt-get install -y curl wget git
    elif command -v yum &> /dev/null; then
        yum update -y
        yum install -y python3 python3-pip python3-devel
        yum groupinstall -y "Development Tools"
        yum install -y opencv-devel python3-opencv
        yum install -y curl wget git
    else
        print_error "不支持的包管理器，请手动安装依赖"
        exit 1
    fi
}

# 创建安装目录
create_directories() {
    print_info "创建安装目录..."
    
    INSTALL_DIR="/opt/VisionCore"
    LOG_DIR="/var/log/visioncore"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/models"
    
    print_info "安装目录: $INSTALL_DIR"
    print_info "日志目录: $LOG_DIR"
}

# 复制程序文件
copy_files() {
    print_info "复制程序文件..."
    
    # 复制所有Python文件和配置
    cp -r ./* "$INSTALL_DIR/"
    
    # 设置权限
    chmod +x "$INSTALL_DIR/main.py"
    chmod +x "$INSTALL_DIR/install.sh"
    
    # 创建符号链接到日志目录
    ln -sf "$LOG_DIR" "$INSTALL_DIR/logs"
    
    print_info "程序文件复制完成"
}

# 安装Python依赖
install_python_deps() {
    print_info "安装Python依赖..."
    
    cd "$INSTALL_DIR"
    
    # 创建requirements.txt（如果不存在）
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << EOF
paho-mqtt>=1.6.0
PyYAML>=6.0
numpy>=1.21.0
opencv-python>=4.5.0
watchdog>=2.1.0
shapely>=1.8.0
ultralytics>=8.0.0
EOF
    fi
    
    # 安装依赖
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    
    print_info "Python依赖安装完成"
}

# 配置systemd服务
setup_service() {
    print_info "配置systemd服务..."
    
    # 复制服务文件
    cp "$INSTALL_DIR/visioncore.service" "/etc/systemd/system/"
    
    # 重新加载systemd
    systemctl daemon-reload
    
    # 启用服务（开机自启）
    systemctl enable visioncore
    
    print_info "systemd服务配置完成"
}

# 创建配置文件
setup_config() {
    print_info "配置系统..."
    
    cd "$INSTALL_DIR"
    
    # 备份原始配置
    if [ -f "Config/config.yaml" ]; then
        cp "Config/config.yaml" "Config/config.yaml.backup"
    fi
    
    print_info "配置文件已准备就绪"
    print_warning "请根据实际情况修改 $INSTALL_DIR/Config/config.yaml"
}

# 创建管理脚本
create_management_scripts() {
    print_info "创建管理脚本..."
    
    # 创建启动脚本
    cat > /usr/local/bin/visioncore-start << 'EOF'
#!/bin/bash
systemctl start visioncore
systemctl status visioncore
EOF
    chmod +x /usr/local/bin/visioncore-start
    
    # 创建停止脚本
    cat > /usr/local/bin/visioncore-stop << 'EOF'
#!/bin/bash
systemctl stop visioncore
EOF
    chmod +x /usr/local/bin/visioncore-stop
    
    # 创建重启脚本
    cat > /usr/local/bin/visioncore-restart << 'EOF'
#!/bin/bash
systemctl restart visioncore
systemctl status visioncore
EOF
    chmod +x /usr/local/bin/visioncore-restart
    
    # 创建状态查看脚本
    cat > /usr/local/bin/visioncore-status << 'EOF'
#!/bin/bash
echo "=== 服务状态 ==="
systemctl status visioncore
echo ""
echo "=== 最近日志 ==="
journalctl -u visioncore -n 20 --no-pager
EOF
    chmod +x /usr/local/bin/visioncore-status
    
    # 创建日志查看脚本
    cat > /usr/local/bin/visioncore-logs << 'EOF'
#!/bin/bash
if [ "$1" == "-f" ]; then
    journalctl -u visioncore -f
else
    journalctl -u visioncore -n 50 --no-pager
fi
EOF
    chmod +x /usr/local/bin/visioncore-logs
    
    print_info "管理脚本创建完成"
}

# 主安装流程
main() {
    print_info "开始安装 VisionCore..."
    
    check_root
    check_architecture
    install_dependencies
    create_directories
    copy_files
    install_python_deps
    setup_config
    setup_service
    create_management_scripts
    
    print_info "安装完成！"
    echo ""
    echo "========================================"
    echo "使用说明："
    echo "========================================"
    echo "1. 配置文件位置: /opt/VisionCore/Config/config.yaml"
    echo "2. 日志文件位置: /var/log/visioncore/"
    echo "3. 管理命令："
    echo "   - 启动服务: visioncore-start"
    echo "   - 停止服务: visioncore-stop"
    echo "   - 重启服务: visioncore-restart"
    echo "   - 查看状态: visioncore-status"
    echo "   - 查看日志: visioncore-logs"
    echo "   - 实时日志: visioncore-logs -f"
    echo ""
    echo "4. 手动启动方式:"
    echo "   cd /opt/VisionCore && python3 main.py"
    echo ""
    print_warning "请先修改配置文件，然后启动服务："
    echo "   nano /opt/VisionCore/Config/config.yaml"
    echo "   visioncore-start"
}

# 运行主函数
main "$@" 
# 使用基于 Debian Buster 的 Python slim 镜像。
# 这个基础镜像 python:3.10-slim-buster 支持多种CPU架构 (例如 linux/amd64, linux/arm64)。
# 当你使用 `docker build --platform <platform>` 时，Docker 会拉取对应平台的基础镜像层。
FROM python:3.10-slim-buster

# 设置镜像作者标签 (可选，但推荐)
LABEL authors="wuxiong"

# 设置容器内的工作目录
WORKDIR /app

# 安装必要的系统依赖
# libgl1-mesa-glx, libsm6, libxext6 通常是 OpenCV GUI 功能所需要的，
# 即便使用 headless OpenCV，有时也可能间接依赖。
# ffmpeg 用于视频处理。
# --no-install-recommends 避免安装不必要的推荐包。
# rm -rf /var/lib/apt/lists/* 清理 apt 缓存，减小镜像体积。
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt 文件。
# 这样可以利用 Docker 的层缓存机制：如果 requirements.txt 没有变化，
# 下面的 pip install 层就不会重新执行，从而加快构建速度。
COPY requirements.txt .

# 安装 Python 依赖。
# --no-cache-dir 避免存储 pip 缓存，减小镜像体积。
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目中的所有文件（包括 app.py, 你的 .pt 模型文件等）到容器的 /app 目录。
# 请确保 yolov8x-face-lindevs.pt 模型文件与 Dockerfile 在同一目录，
# 或者在 COPY 命令中正确指定其相对路径。
COPY . /app

# 声明容器运行时监听的端口。
# 这只是一个元数据声明，实际端口映射需要在 `docker run -p` 命令中指定。
EXPOSE 8080

# 定义容器启动时执行的命令。
CMD ["python", "app.py"]

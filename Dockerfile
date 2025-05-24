# 使用官方 ultralytics 镜像（包含 PyTorch 和 Ultralytics）
FROM ultralytics/ultralytics:latest
LABEL authors="wuxiong"

# 设置工作目录
WORKDIR /app

# 复制项目代码和模型文件到镜像中
COPY . /app

# 安装依赖（如需自定义版本，在 requirements.txt 中指定）
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 Flask 默认端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"]

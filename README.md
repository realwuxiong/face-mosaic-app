# face-mosaic-app 人脸打码
## 说明
✨ 代码简单
🌟 先进人脸检测 - 基于 YOLOv8，精准定位人脸。
🎯 灵活打码区域 - 可选完整人脸、眼部或鼻嘴区域。
🎨 多样打码样式 - 支持传统块状马赛克及自定义图案。
⚙️ 参数可调 - 马赛克级别可自由调整。
⚡ 高效处理 - 快速完成图像打码。
🖼️ 图像预览 - 上传后即时预览主图像和图案。
🛡️ 隐私安全 - 所有处理均在服务器端安全进行，图像不被存储。
📤 格式保留 - 尽可能保留原始图片格式 (PNG/WEBP)，其余输出为JPG。

## Doker 运行
最新版
> docker run -d --name face-mosaic-app --restart always -p 8082:8080 realwuxiong/face-mosaic-app:latest

稳定版
> docker run -d --name face-mosaic-app --restart always -p 8082:8080 realwuxiong/face-mosaic-app:stable


## 人脸识别模型参考
https://github.com/lindevs/yolov8-face

## 效果展示：
<img width="1438" alt="image" src="https://github.com/user-attachments/assets/fc4441d8-e85b-4df7-8de1-ca061346f369" />

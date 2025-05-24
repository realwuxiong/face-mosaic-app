import io
import logging
import os

import cv2
import numpy as np
from flask import Flask, request, send_file, render_template_string, flash, redirect
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super_secret_key_for_production_change_this"  # Change in production

# Initialize YOLOv8 face detection model
model = YOLO('yolov8x-face-lindevs.pt')  # Replace with path to your YOLOv8 face detection model

# Configuration parameters
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_region_bounding_box(image_rows, image_cols, box, region_key="full"):
    """
    Calculate bounding box for a specific facial region based on YOLO bounding box
    box: [x_center, y_center, width, height] (normalized)
    Returns (xmin, ymin, width, height) for the specified region
    """
    try:
        # Convert YOLO bounding box (normalized) to absolute coordinates
        x_center, y_center, width, height = box
        xmin_abs = int((x_center - width / 2) * image_cols)
        ymin_abs = int((y_center - height / 2) * image_rows)
        width_abs = int(width * image_cols)
        height_abs = int(height * image_rows)

        # Default to full face bounding box
        if region_key == "full":
            pass  # Use the full bounding box as is
        elif region_key == "eyes":
            # Estimate eye region: top ~30% of the face
            height_abs = int(height_abs * 0.3)
            ymin_abs += int(height_abs * 0.1)  # Slight offset from top
            width_abs = int(width_abs * 0.8)   # Slightly narrower
            xmin_abs += int(width_abs * 0.1)   # Center horizontally
        elif region_key == "nose_mouth":
            # Estimate nose/mouth region: bottom ~50% of the face
            ymin_abs += int(height_abs * 0.5)  # Start from middle
            height_abs = int(height_abs * 0.5) # Bottom half
            width_abs = int(width_abs * 0.8)   # Slightly narrower
            xmin_abs += int(width_abs * 0.1)   # Center horizontally

        # Ensure coordinates are within image bounds
        xmin_abs = max(0, xmin_abs)
        ymin_abs = max(0, ymin_abs)
        xmax_abs = min(image_cols - 1, xmin_abs + width_abs)
        ymax_abs = min(image_rows - 1, ymin_abs + height_abs)

        # Recalculate width and height based on clipped coordinates
        width_abs = max(1, xmax_abs - xmin_abs)
        height_abs = max(1, ymax_abs - ymin_abs)

        return xmin_abs, ymin_abs, width_abs, height_abs

    except Exception as e:
        logger.error(f"Error getting region bounding box: {e}")
        # Return default face box
        return (
            int((x_center - width / 2) * image_cols),
            int((y_center - height / 2) * image_rows),
            int(width * image_cols),
            int(height * image_rows)
        )

def apply_mosaic_or_pattern(image, x, y, w, h, mosaic_level=20, custom_pattern_cv2=None):
    """
    Apply block mosaic or custom pattern to the specified ROI
    Modifies 'image' directly
    """
    try:
        if w <= 0 or h <= 0:
            return image  # Skip if region has no area

        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        if w <= 0 or h <= 0:
            return image

        target_roi = image[y:y + h, x:x + w]

        if target_roi.size == 0:
            return image

        if custom_pattern_cv2 is not None:
            # Resize pattern to fit target ROI
            pattern_resized = cv2.resize(custom_pattern_cv2, (w, h), interpolation=cv2.INTER_AREA)

            # Handle different channel cases
            if len(pattern_resized.shape) == 3 and pattern_resized.shape[2] == 4:  # Pattern has alpha channel
                if len(image.shape) == 3 and image.shape[2] == 3:  # Image has no alpha
                    alpha = pattern_resized[:, :, 3:4] / 255.0
                    alpha_inv = 1.0 - alpha
                    for c in range(3):
                        target_roi[:, :, c] = (
                            target_roi[:, :, c] * alpha_inv[:, :, 0] +
                            pattern_resized[:, :, c] * alpha[:, :, 0]
                        )
                else:
                    target_roi[:, :] = pattern_resized[:, :, :3]
            elif len(pattern_resized.shape) == 3 and pattern_resized.shape[2] == 3:  # Both BGR
                target_roi[:, :] = pattern_resized
            elif len(pattern_resized.shape) == 2:  # Pattern is grayscale
                if len(image.shape) == 3:
                    target_roi[:, :] = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
                else:
                    target_roi[:, :] = pattern_resized
            else:
                # Fallback to block mosaic
                logger.warning("Unsupported pattern format, using block mosaic")
                return apply_block_mosaic(image, x, y, w, h, mosaic_level)
        else:
            # Apply block mosaic
            return apply_block_mosaic(image, x, y, w, h, mosaic_level)

        image[y:y + h, x:x + w] = target_roi
        return image

    except Exception as e:
        logger.error(f"Error applying mosaic: {e}")
        return image

def apply_block_mosaic(image, x, y, w, h, mosaic_level):
    """Apply block mosaic effect"""
    try:
        target_roi = image[y:y + h, x:x + w]
        roi_h, roi_w = target_roi.shape[:2]

        if roi_w == 0 or roi_h == 0:
            return image

        # Ensure mosaic_level is reasonable
        mosaic_level = max(2, min(mosaic_level, min(roi_w, roi_h) // 2))

        # 1. Downscale
        small_w = max(1, roi_w // mosaic_level)
        small_h = max(1, roi_h // mosaic_level)
        small_roi = cv2.resize(target_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # 2. Upscale to create mosaic effect
        mosaic_roi = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        image[y:y + h, x:x + w] = mosaic_roi

        return image
    except Exception as e:
        logger.error(f"Error applying block mosaic: {e}")
        return image

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        try:
            # Check if a file was uploaded
            if 'file' not in request.files:
                flash("请选择图像文件。", "error")
                return redirect(request.url)

            main_image_file = request.files['file']
            if main_image_file.filename == '':
                flash("请选择图像文件。", "error")
                return redirect(request.url)

            # Check file type
            if not allowed_file(main_image_file.filename):
                flash("不支持的文件格式。请上传 PNG, JPG, JPEG, GIF, BMP 或 WEBP 格式的图像。", "error")
                return redirect(request.url)

            # Get form parameters
            custom_pattern_cv2 = None
            mosaic_type = request.form.get('mosaic_type', 'block')
            mosaic_region_key = request.form.get('mosaic_region', 'full')
            mosaic_level = max(2, min(50, int(request.form.get('mosaic_level', 20))))

            # Handle custom pattern
            if mosaic_type == 'pattern':
                if 'pattern_file' in request.files:
                    pattern_file = request.files['pattern_file']
                    if pattern_file.filename != '' and allowed_file(pattern_file.filename):
                        try:
                            pattern_img_bytes = pattern_file.read()
                            if len(pattern_img_bytes) > MAX_FILE_SIZE:
                                flash("图案文件过大，请选择小于16MB的文件。", "error")
                                return redirect(request.url)

                            pattern_nparr = np.frombuffer(pattern_img_bytes, np.uint8)
                            custom_pattern_cv2 = cv2.imdecode(pattern_nparr, cv2.IMREAD_UNCHANGED)

                            if custom_pattern_cv2 is None:
                                flash("无法读取图案图像。使用块状马赛克代替。", "warning")
                                mosaic_type = 'block'
                        except Exception as e:
                            logger.error(f"处理图案时出错: {e}")
                            flash(f"处理图案时出错。使用块状马赛克代替。", "warning")
                            mosaic_type = 'block'
                    else:
                        flash("选择了图案马赛克但未上传有效的图案文件。使用块状马赛克代替。", "warning")
                        mosaic_type = 'block'

            # Process main image
            img_bytes = main_image_file.read()
            if len(img_bytes) > MAX_FILE_SIZE:
                flash("图像文件过大，请选择小于16MB的文件。", "error")
                return redirect(request.url)

            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                flash("错误：无法读取主图像。", "error")
                return redirect(request.url)

            # Optional: Resize image to improve detection performance
            max_size = 1280
            image_rows, image_cols = image.shape[:2]
            scale = min(max_size / image_cols, max_size / image_rows)
            if scale < 1:
                image = cv2.resize(image, (int(image_cols * scale), int(image_rows * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                image_rows, image_cols = image.shape[:2]
            processed_image = image.copy()
            faces_processed = 0

            # Use YOLO for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_image, conf=0.3)  # Confidence threshold set to 0.3

            detected_faces = len(results[0].boxes) if results[0].boxes else 0
            logger.info(f"检测到 {detected_faces} 张人脸")

            if detected_faces > 0:
                logger.info(f"处理 {detected_faces} 张人脸")
                # Process each detected face
                for detection_idx, box in enumerate(results[0].boxes):
                    try:
                        confidence = box.conf.item()  # Get confidence score
                        logger.info(f"处理第 {detection_idx + 1} 张人脸，置信度: {confidence:.3f}")

                        # Get bounding box (YOLO format: [x_center, y_center, width, height], normalized)
                        box_xywh = box.xywhn[0].tolist()  # Normalized coordinates
                        logger.info(f"人脸 {detection_idx + 1} 原始框: {box_xywh}")

                        # Get region-specific bounding box
                        xmin, ymin, w, h = get_region_bounding_box(
                            image_rows, image_cols, box_xywh, mosaic_region_key
                        )
                        logger.info(f"人脸 {detection_idx + 1} 区域: x={xmin}, y={ymin}, w={w}, h={h}")

                        if w > 0 and h > 0:
                            # Apply mosaic or pattern to the face
                            processed_image = apply_mosaic_or_pattern(
                                processed_image, xmin, ymin, w, h,
                                mosaic_level=mosaic_level,
                                custom_pattern_cv2=custom_pattern_cv2 if mosaic_type == 'pattern' else None
                            )
                            faces_processed += 1
                            logger.info(f"成功处理第 {detection_idx + 1} 张人脸")
                        else:
                            logger.warning(f"第 {detection_idx + 1} 张人脸的区域无效: w={w}, h={h}")
                    except Exception as e:
                        logger.error(f"处理第 {detection_idx + 1} 张人脸时出错: {e}")
                        continue
            else:
                flash("❌ 图像中未检测到人脸。建议：1) 确保人脸清晰可见 2) 尝试更高分辨率的图像 3) 确保光线充足",
                      "warning")
                logger.warning("未检测到任何人脸")

            if faces_processed > 0:
                flash(f"✅ 成功处理了 {faces_processed} 张人脸！", "success")
            elif detected_faces > 0:
                flash("⚠️ 检测到人脸但处理失败，请检查图像质量或上传其他图像。", "warning")
            else:
                flash("❌ 未检测到人脸，请确保图像包含清晰的人脸。", "warning")

            # Encode the processed image
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            is_success, buffer = cv2.imencode(".jpg", processed_image, encode_params)

            if not is_success:
                flash("编码处理后的图像时出错。", "error")
                return redirect(request.url)

            img_io = io.BytesIO(buffer)
            img_io.seek(0)

            return send_file(
                img_io,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"mosaiced_{main_image_file.filename}"
            )

        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}")
            flash(f"处理过程中发生错误: {str(e)}", "error")
            return redirect(request.url)

    # HTML form (GET request)
    return render_template_string('''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>高级人脸打码工具</title>
<style>
* { box-sizing: border-box; }
body {
    font-family: 'Segoe UI', 'Microsoft YaHei', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0; padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    color: #333;
}
.container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    width: 100%;
    max-width: 650px;
    margin-bottom: 20px;
}
h1 {
    color: #4a5568;
    text-align: center;
    margin-bottom: 30px;
    font-size: 2.2em;
    font-weight: 300;
}
.form-group {
    margin-bottom: 25px;
}
label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #4a5568;
    font-size: 0.95em;
}
input[type="file"], select, input[type="number"] {
    width: 100%;
    padding: 12px 15px;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.3s ease;
    background: white;
}
input[type="file"]:focus, select:focus, input[type="number"]:focus {
    border-color: #667eea;
    outline: none;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}
.radio-group {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}
.radio-group label {
    margin: 0;
    font-weight: normal;
    display: flex;
    align-items: center;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 6px;
    transition: background-color 0.2s;
}
.radio-group label:hover {
    background-color: #f7fafc;
}
input[type="radio"] {
    margin-right: 8px;
    width: auto;
}
input[type="submit"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 25px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
input[type="submit"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
.messages {
    list-style: none;
    padding: 0;
    margin-bottom: 25px;
}
.messages li {
    padding: 12px 15px;
    margin-bottom: 10px;
    border-radius: 8px;
    font-weight: 500;
}
.messages .error {
    background-color: #fed7d7;
    color: #c53030;
    border-left: 4px solid #e53e3e;
}
.messages .warning {
    background-color: #fef5e7;
    color: #d69e2e;
    border-left: 4px solid #ed8936;
}
.messages .success {
    background-color: #c6f6d5;
    color: #2f855a;
    border-left: 4px solid #38a169;
}
.options-section {
    margin-top: 15px;
    padding: 20px;
    border: 2px dashed #e2e8f0;
    border-radius: 8px;
    background-color: #f8fafc;
}
#preview {
    max-width: 100%;
    margin-top: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: none;
}
.file-info {
    font-size: 0.85em;
    color: #718096;
    margin-top: 5px;
}
.feature-list {
    background: #f8fafc;
    padding: 20px;
    border-radius: 8px;
    margin-top: 20px;
}
.feature-list h3 {
    color: #4a5568;
    margin-bottom: 15px;
    font-size: 1.1em;
}
.feature-list ul {
    margin: 0;
    padding-left: 20px;
    color: #718096;
}
.feature-list li {
    margin-bottom: 8px;
}
</style>
<script>
function toggleOptions() {
    var mosaicType = document.querySelector('input[name="mosaic_type"]:checked').value;
    document.getElementById('block_options').style.display = (mosaicType === 'block') ? 'block' : 'none';
    document.getElementById('pattern_options').style.display = (mosaicType === 'pattern') ? 'block' : 'none';
}
function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('preview');
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(file);
        const fileInfo = document.getElementById('file-info');
        const fileSize = (file.size / 1024 / 1024).toFixed(2);
        fileInfo.textContent = `文件大小: ${fileSize} MB`;
    } else {
        preview.style.display = 'none';
        document.getElementById('file-info').textContent = '';
    }
}
window.onload = function() {
    toggleOptions();
};
</script>
</head>
<body>
<div class="container">
<h1>🎭 高级人脸打码工具</h1>
{% with messages = get_flashed_messages(with_categories=true) %}
{% if messages %}
<ul class="messages">
{% for category, message in messages %}
<li class="{{ category }}">{{ message }}</li>
{% endfor %}
</ul>
{% endif %}
{% endwith %}
<form method="post" enctype="multipart/form-data" action="/">
<div class="form-group">
<label for="file">📸 上传图像:</label>
<input type="file" name="file" id="file" onchange="previewImage(event)"
accept=".png,.jpg,.jpeg,.gif,.bmp,.webp" required>
<div class="file-info" id="file-info"></div>
</div>
<div class="form-group">
<label>🎯 打码区域:</label>
<select name="mosaic_region">
<option value="full" selected>完整人脸</option>
<option value="eyes">仅眼部</option>
<option value="nose_mouth">鼻嘴区域</option>
</select>
</div>
<div class="form-group">
<label>🎨 打码类型:</label>
<div class="radio-group">
<label>
<input type="radio" name="mosaic_type" value="block" checked onchange="toggleOptions()">
块状马赛克
</label>
<label>
<input type="radio" name="mosaic_type" value="pattern" onchange="toggleOptions()">
自定义图案
</label>
</div>
</div>
<div id="block_options" class="options-section">
<div class="form-group">
<label for="mosaic_level">🔲 马赛克级别:</label>
<input type="number" name="mosaic_level" id="mosaic_level"
value="20" min="2" max="50" step="1">
<div class="file-info">数值越小，马赛克块越大</div>
</div>
</div>
<div id="pattern_options" class="options-section" style="display:none;">
<div class="form-group">
<label for="pattern_file">🖼️ 上传图案图像:</label>
<input type="file" name="pattern_file" id="pattern_file"
accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
<div class="file-info">如果不上传图案，将使用块状马赛克作为后备方案</div>
</div>
</div>
<input type="submit" value="🚀 开始处理图像">
</form>
<img id="preview" alt="图像预览">
</div>
<div class="container feature-list">
<h3>✨ 功能特点</h3>
<ul>
<li>🔍 <strong>多人脸检测</strong> - 自动检测并处理图像中的所有人脸</li>
<li>🎯 <strong>精确区域选择</strong> - 支持完整人脸、仅眼部或鼻嘴区域打码</li>
<li>🎨 <strong>多种打码方式</strong> - 块状马赛克或自定义图案覆盖</li>
<li>⚡ <strong>高质量输出</strong> - 保持图像质量的同时完成打码处理</li>
<li>🛡️ <strong>隐私保护</strong> - 所有处理在本地完成，不会上传到外部服务器</li>
</ul>
</div>
</body>
</html>
''')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    # Set debug=False in production
    app.run(host='0.0.0.0', port=port, debug=False)
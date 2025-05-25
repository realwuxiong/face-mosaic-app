import base64
import logging
import os

import cv2
import numpy as np
import torch
from flask import Flask, request, render_template_string, jsonify
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "super_secret_key_for_production_change_this"  # Change in production

# Monkey-patch torch.load for YOLO model loading
_original_torch_load = torch.load


def _custom_torch_load_for_yolo(*args, **kwargs):
    original_weights_only = kwargs.pop('weights_only', None)
    logger.info(f"Custom torch.load called for YOLO: Forcing weights_only=False. Original was: {original_weights_only}")
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return _original_torch_load(*args, **kwargs, weights_only=False)


logger.info("Attempting to load YOLO model with monkey-patched torch.load (weights_only=False).")
torch.load = _custom_torch_load_for_yolo

try:
    model = YOLO('yolov8x-face-lindevs.pt')  # Replace with path to your YOLOv8 face detection model
    logger.info("Successfully loaded YOLO model with monkey-patch.")
except Exception as e:
    logger.error(f"Failed to load YOLO model even with monkey-patch: {e}", exc_info=True)
    torch.load = _original_torch_load
    logger.info("Restored original torch.load after model loading failure.")
    raise RuntimeError(f"Could not load YOLO model. Please check the model path and integrity. Error: {e}")
finally:
    torch.load = _original_torch_load
    logger.info("Restored original torch.load after model loading attempt.")

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
        x_center, y_center, width, height = box
        xmin_abs = int((x_center - width / 2) * image_cols)
        ymin_abs = int((y_center - height / 2) * image_rows)
        width_abs = int(width * image_cols)
        height_abs = int(height * image_rows)

        if region_key == "full":
            pass
        elif region_key == "eyes":
            height_abs = int(height_abs * 0.3)
            ymin_abs += int(height_abs * 0.4)
            width_abs = int(width_abs * 1)
            xmin_abs += int(width_abs * 0.001)
        elif region_key == "nose_mouth":
            ymin_abs += int(height_abs * 0.5)
            height_abs = int(height_abs * 0.5)
            width_abs = int(width_abs * 0.8)
            xmin_abs += int(width_abs * 0.1)

        xmin_abs = max(0, xmin_abs)
        ymin_abs = max(0, ymin_abs)
        xmax_abs = min(image_cols - 1, xmin_abs + width_abs)
        ymax_abs = min(image_rows - 1, ymin_abs + height_abs)
        width_abs = max(1, xmax_abs - xmin_abs)
        height_abs = max(1, ymax_abs - ymin_abs)

        return xmin_abs, ymin_abs, width_abs, height_abs
    except Exception as e:
        logger.error(f"Error getting region bounding box: {e}")
        return (
            int((box[0] - box[2] / 2) * image_cols),
            int((box[1] - box[3] / 2) * image_rows),
            int(box[2] * image_cols),
            int(box[3] * image_rows)
        )


def apply_mosaic_or_pattern(image, x, y, w, h, mosaic_level=20, custom_pattern_cv2=None):
    """
    Apply block mosaic or custom pattern to the specified ROI
    Modifies 'image' directly
    """
    try:
        if w <= 0 or h <= 0:
            return image

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
            pattern_resized = cv2.resize(custom_pattern_cv2, (w, h), interpolation=cv2.INTER_AREA)
            if len(pattern_resized.shape) == 3 and pattern_resized.shape[2] == 4:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    alpha = pattern_resized[:, :, 3:4] / 255.0
                    alpha_inv = 1.0 - alpha
                    for c in range(3):
                        target_roi[:, :, c] = (
                                target_roi[:, :, c] * alpha_inv[:, :, 0] +
                                pattern_resized[:, :, c] * alpha[:, :, 0]
                        )
                else:
                    target_roi[:, :] = pattern_resized[:, :, :image.shape[2]]
            elif len(pattern_resized.shape) == 3 and pattern_resized.shape[2] == 3:
                target_roi[:, :] = pattern_resized
            elif len(pattern_resized.shape) == 2:
                if len(image.shape) == 3:
                    target_roi[:, :] = cv2.cvtColor(pattern_resized, cv2.COLOR_GRAY2BGR)
                else:
                    target_roi[:, :] = pattern_resized
            else:
                return apply_block_mosaic(image, x, y, w, h, mosaic_level)
        else:
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

        mosaic_level = max(2, min(mosaic_level, min(roi_w, roi_h) // 2 if min(roi_w, roi_h) > 3 else 2))
        small_w = max(1, roi_w // mosaic_level)
        small_h = max(1, roi_h // mosaic_level)
        small_roi = cv2.resize(target_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic_roi = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        image[y:y + h, x:x + w] = mosaic_roi
        return image
    except Exception as e:
        logger.error(f"Error applying block mosaic: {e}")
        return image


def image_to_base64(image, format='JPEG'):
    """Convert OpenCV image to base64 string for web display"""
    try:
        if format.upper() == 'PNG':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            ext = '.png'
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            ext = '.jpg'

        is_success, buffer = cv2.imencode(ext, image, encode_params)
        if not is_success:
            return None

        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'message': "请选择图像文件。"})

            main_image_file = request.files['file']
            if main_image_file.filename == '':
                return jsonify({'success': False, 'message': "请选择图像文件。"})

            if not allowed_file(main_image_file.filename):
                return jsonify({'success': False,
                                'message': "不支持的文件格式。请上传 PNG, JPG, JPEG, GIF, BMP 或 WEBP 格式的图像。"})

            custom_pattern_cv2 = None
            mosaic_type = request.form.get('mosaic_type', 'block')
            mosaic_region_key = request.form.get('mosaic_region', 'full')
            mosaic_level = max(2, min(50, int(request.form.get('mosaic_level', 20))))

            if mosaic_type == 'pattern':
                if 'pattern_file' in request.files:
                    pattern_file = request.files['pattern_file']
                    if pattern_file.filename != '' and allowed_file(pattern_file.filename):
                        try:
                            pattern_img_bytes = pattern_file.read()
                            if len(pattern_img_bytes) > MAX_FILE_SIZE:
                                return jsonify({'success': False, 'message': "图案文件过大，请选择小于16MB的文件。"})
                            pattern_nparr = np.frombuffer(pattern_img_bytes, np.uint8)
                            custom_pattern_cv2 = cv2.imdecode(pattern_nparr, cv2.IMREAD_UNCHANGED)
                            if custom_pattern_cv2 is None:
                                mosaic_type = 'block'
                        except Exception as e:
                            logger.error(f"处理图案时出错: {e}")
                            mosaic_type = 'block'
                    else:
                        mosaic_type = 'block'
                else:
                    mosaic_type = 'block'

            img_bytes = main_image_file.read()
            if len(img_bytes) > MAX_FILE_SIZE:
                return jsonify({'success': False, 'message': "图像文件过大，请选择小于16MB的文件。"})

            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({'success': False, 'message': "错误：无法读取主图像。"})

            max_size = 1280
            image_rows, image_cols = image.shape[:2]
            scale = min(max_size / image_cols, max_size / image_rows, 1.0)
            if scale < 1:
                image = cv2.resize(image, (int(image_cols * scale), int(image_rows * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                image_rows, image_cols = image.shape[:2]

            processed_image = image.copy()
            faces_processed = 0

            results = model.predict(image, conf=0.3)
            detected_faces = len(results[0].boxes) if results and results[0].boxes is not None else 0
            logger.info(f"检测到 {detected_faces} 张人脸")

            if detected_faces > 0:
                for detection_idx, box_obj in enumerate(results[0].boxes):
                    try:
                        confidence = box_obj.conf.item()
                        logger.info(f"处理第 {detection_idx + 1} 张人脸，置信度: {confidence:.3f}")
                        box_xywhn = box_obj.xywhn[0].tolist()
                        logger.info(f"人脸 {detection_idx + 1} 原始框 (归一化 xywh): {box_xywhn}")

                        xmin, ymin, w, h = get_region_bounding_box(image_rows, image_cols, box_xywhn, mosaic_region_key)
                        logger.info(f"人脸 {detection_idx + 1} 区域 (绝对 xywh): x={xmin}, y={ymin}, w={w}, h={h}")

                        if w > 0 and h > 0:
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

            original_ext = main_image_file.filename.rsplit('.', 1)[-1].lower()
            img_format = 'PNG' if original_ext == 'png' else 'JPEG'
            processed_img_base64 = image_to_base64(processed_image, img_format)
            if processed_img_base64 is None:
                return jsonify({'success': False, 'message': "处理图像时发生错误。"})

            if original_ext not in ['png', 'webp']:
                output_ext = 'jpg'
                mimetype = 'image/jpeg'
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif original_ext == 'png':
                output_ext = 'png'
                mimetype = 'image/png'
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            else:
                output_ext = 'webp'
                mimetype = 'image/webp'
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, 95]

            is_success, buffer = cv2.imencode(f".{output_ext}", processed_image, encode_params)
            if not is_success:
                return jsonify({'success': False, 'message': "编码处理后的图像时出错。"})

            download_data = base64.b64encode(buffer).decode('utf-8')
            base_filename, _ = os.path.splitext(main_image_file.filename)
            safe_base_filename = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in base_filename)
            download_filename = f"mosaiced_{safe_base_filename}.{output_ext}"

            result_message = ""
            if faces_processed > 0:
                result_message = f"✅ 成功处理了 {faces_processed} 张人脸！"
            elif detected_faces > 0:
                result_message = "⚠️ 检测到人脸但处理失败，请检查图像质量。"
            else:
                result_message = "❌ 图像中未检测到人脸。建议：1) 确保人脸清晰可见 2) 尝试更高分辨率的图像 3) 确保光线充足"

            return jsonify({
                'success': True,
                'message': result_message,
                'processed_image': processed_img_base64,
                'download_data': f"data:{mimetype};base64,{download_data}",
                'download_filename': download_filename,
                'faces_detected': detected_faces,
                'faces_processed': faces_processed
            })

        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'message': f"处理过程中发生错误: {str(e)}"})

    return render_template_string('''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>高级人脸打码工具</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎭</text></svg>">
<style>
* { box-sizing: border-box; }
body {
    font-family: 'Segoe UI', 'Microsoft YaHei', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0; padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
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
    margin: 0 auto 20px;
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
.btn {
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
    text-decoration: none;
    display: inline-block;
    text-align: center;
    margin-bottom: 10px;
}
.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
.btn:disabled {
    background: #ccc;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}
.btn-download {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    margin-top: 15px;
}
.btn-download:hover {
    box-shadow: 0 5px 15px rgba(72, 187, 120, 0.4);
}
.alert {
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
    font-weight: 500;
}
.alert-error {
    background-color: #fed7d7;
    color: #c53030;
    border-left: 4px solid #e53e3e;
}
.alert-warning {
    background-color: #fef5e7;
    color: #d69e2e;
    border-left: 4px solid #ed8936;
}
.alert-success {
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
.preview-image {
    max-width: 100%;
    max-height: 400px;
    margin-top: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: none;
    object-fit: contain;
}
.result-section {
    display: none;
    margin-top: 30px;
    text-align: center;
    padding: 20px;
    background: #f8fafc;
    border-radius: 8px;
    border: 2px solid #e2e8f0;
}
.processed-image {
    max-width: 100%;
    max-height: 500px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 20px;
    object-fit: contain;
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
.spinner {
    border: 4px solid rgba(0, 0, 0, 0.1);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border-left-color: #667eea;
    animation: spin 1s ease infinite;
    display: none;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.stats {
    background: #edf2f7;
    padding: 10px 15px;
    border-radius: 6px;
    margin-bottom: 15px;
    font-size: 0.9em;
    color: #4a5568;
}
@media (max-width: 768px) {
    body { padding: 10px; }
    .container { padding: 20px; }
    h1 { font-size: 1.8em; }
    .radio-group { flex-direction: column; gap: 10px; }
}
</style>
<script>
function toggleOptions() {
    var mosaicType = document.querySelector('input[name="mosaic_type"]:checked').value;
    document.getElementById('block_options').style.display = mosaicType === 'block' ? 'block' : 'none';
    document.getElementById('pattern_options').style.display = mosaicType === 'pattern' ? 'block' : 'none';
}

function previewImage(event, type) {
    const file = event.target.files[0];
    const previewId = type === 'main' ? 'preview' : 'pattern_preview';
    const fileInfoId = type === 'main' ? 'file-info' : 'pattern-file-info';
    const preview = document.getElementById(previewId);
    const fileInfo = document.getElementById(fileInfoId);

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            if (preview) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
        }
        reader.readAsDataURL(file);
        if (fileInfo) {
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            fileInfo.textContent = `文件大小: ${fileSize} MB. 类型: ${file.type || '未知'}`;
        }
    } else {
        if (preview) preview.style.display = 'none';
        if (fileInfo) fileInfo.textContent = type === 'main' ? '最大 16MB. 支持 PNG, JPG, GIF, BMP, WEBP.' : '如果不上传图案，或图案无效，将使用块状马赛克。';
    }
}

function handleSubmit(event) {
    event.preventDefault();
    const form = document.getElementById('upload-form');
    const submitButton = form.querySelector('.btn');
    const spinner = document.getElementById('spinner');
    const messageArea = document.getElementById('message-area');
    const resultSection = document.getElementById('result-section');
    const processedImage = document.getElementById('processed-image');
    const downloadLink = document.getElementById('download-link');
    const stats = document.getElementById('stats');

    if (!form.querySelector('input[name="file"]').files[0]) {
        showMessage('error', '请选择一个图像文件。');
        return;
    }

    spinner.style.display = 'block';
    submitButton.disabled = true;
    submitButton.textContent = '正在处理...';

    const formData = new FormData(form);
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        spinner.style.display = 'none';
        submitButton.disabled = false;
        submitButton.textContent = '🚀 开始处理图像';

        if (data.success) {
            showMessage(data.message.includes('成功') ? 'success' : (data.faces_detected > 0 ? 'warning' : 'error'), data.message);
            resultSection.style.display = 'block';
            processedImage.src = data.processed_image;
            downloadLink.href = data.download_data;
            downloadLink.download = data.download_filename;
            stats.innerHTML = `检测到人脸: ${data.faces_detected} | 处理成功: ${data.faces_processed}`;
        } else {
            showMessage('error', data.message);
            resultSection.style.display = 'none';
        }
    })
    .catch(error => {
        spinner.style.display = 'none';
        submitButton.disabled = false;
        submitButton.textContent = '🚀 开始处理图像';
        showMessage('error', '处理过程中发生错误: ' + error.message);
        resultSection.style.display = 'none';
    });
}

function showMessage(type, message) {
    const messageArea = document.getElementById('message-area');
    messageArea.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
}

window.onload = function() {
    toggleOptions();
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.value = '';
        const event = new Event('change');
        input.dispatchEvent(event);
    });
    const submitButton = document.querySelector('.btn');
    if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = '🚀 开始处理图像';
    }
    const spinner = document.getElementById('spinner');
    if (spinner) spinner.style.display = 'none';
};
</script>
</head>
<body>
<div class="container">
<h1>🎭 高级人脸打码工具</h1>

<div id="message-area"></div>

<form id="upload-form" enctype="multipart/form-data" onsubmit="handleSubmit(event);">
<div class="form-group">
<label for="file">📸 上传图像:</label>
<input type="file" name="file" id="file" onchange="previewImage(event, 'main')"
accept=".png,.jpg,.jpeg,.gif,.bmp,.webp" required>
<div class="file-info" id="file-info">最大 16MB. 支持 PNG, JPG, GIF, BMP, WEBP.</div>
</div>
<img id="preview" class="preview-image" alt="图像预览">

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
<label for="mosaic_level">🔲 马赛克级别 (2-50):</label>
<input type="number" name="mosaic_level" id="mosaic_level" value="20" min="2" max="50" step="1">
<div class="file-info">数值越小，马赛克块越大 (效果越强)。</div>
</div>
</div>

<div id="pattern_options" class="options-section" style="display:none;">
<div class="form-group">
<label for="pattern_file">🖼️ 上传图案图像 (可选):</label>
<input type="file" name="pattern_file" id="pattern_file" onchange="previewImage(event, 'pattern')"
accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
<div class="file-info" id="pattern-file-info">如果不上传图案，或图案无效，将使用块状马赛克。</div>
</div>
<img id="pattern_preview" class="preview-image" alt="图案预览">
</div>

<button type="submit" class="btn">🚀 开始处理图像</button>
<div class="spinner" id="spinner"></div>
</form>

<div id="result-section" class="result-section">
<div id="stats" class="stats"></div>
<img id="processed-image" class="processed-image" alt="处理后的图像">
<a id="download-link" class="btn btn-download">📥 下载处理后的图像</a>
</div>
</div>

<div class="container feature-list">
<h3>✨ 功能特点</h3>
<ul>
<li>🌟 <strong>先进人脸检测</strong> - 基于 YOLOv8，精准定位人脸。</li>
<li>🎯 <strong>灵活打码区域</strong> - 可选完整人脸、眼部或鼻嘴区域。</li>
<li>🎨 <strong>多样打码样式</strong> - 支持传统块状马赛克及自定义图案。</li>
<li>⚙️ <strong>参数可调</strong> - 马赛克级别可自由调整。</li>
<li>⚡ <strong>高效处理</strong> - 快速完成图像打码。</li>
<li>🖼️ <strong>图像预览</strong> - 上传后即时预览主图像和图案。</li>
<li>🛡️ <strong>隐私安全</strong> - 所有处理均在服务器端安全进行，图像不被存储。</li>
<li>📤 <strong>格式保留</strong> - 尽可能保留原始图片格式 (PNG/WEBP)，其余输出为JPG。</li>
</ul>
</div>
</body>
</html>
''')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

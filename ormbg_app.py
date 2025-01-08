from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
from skimage import io
from ormbg import ORMBG
import torch.nn.functional as F

app = Flask(__name__)

# Configure static folder and subdirectories
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Model path
MODEL_PATH = os.path.join('models', 'ormbg.pth')

# Preprocess image
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    return image

# Postprocess image
def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

# Inference
def remove_background(image_path, output_path):
    net = ORMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(MODEL_PATH))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()

    model_input_size = [1024, 1024]
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    result = net(image)

    # Post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # Save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(image_path)
    no_bg_image.paste(orig_image, mask=pil_im)

    # Ensure output is saved as PNG
    if not output_path.lower().endswith(".png"):
        output_path = os.path.splitext(output_path)[0] + ".png"

    no_bg_image.save(output_path, format="PNG")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")
            file.save(input_path)

            # Process the image
            remove_background(input_path, output_path)

            # Convert paths for static serving
            input_url = url_for('static', filename=f'uploads/{filename}')
            output_url = url_for('static', filename=f'outputs/processed_{os.path.splitext(filename)[0]}.png')

            return render_template('index.html', input_image=input_url, output_image=output_url)
    return render_template('index.html', input_image=None, output_image=None)

if __name__ == '__main__':
    app.run(debug=True)

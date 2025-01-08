import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*size_average and reduce args will be deprecated.*")

import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from skimage import io
from rembg import remove  # Import rembg for background removal
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from isnet import ISNetDIS  # Assuming you have the ISNetDIS model
from werkzeug.utils import secure_filename
import io as io_lib
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from ormbg import ORMBG
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

app = Flask(__name__)

# Folder for saving uploaded images
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load the ISNetDIS model
model_path = os.path.join('models', 'isnet-general-use.pth')
MODEL_PATH = os.path.join('models', 'ormbg.pth')
net = ISNetDIS()

if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
else:
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

net.eval()

# Handle images with 4 channels (RGBA)
def handle_rgba(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:  # If the image has 4 channels (RGBA)
        image = image[:, :, :3]  # Convert to RGB by removing the alpha channel
    elif image.shape[2] != 3:  # If the image has an unsupported number of channels
        raise ValueError("Unsupported image format. Please upload a valid image with 3 (RGB) channels.")
    return image

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

def remove_bg_isnet(image_path):
    im = imread(image_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]  # Add a channel dimension if it's grayscale

    if im.shape[2] == 4:  # If the image has 4 channels (RGBA)
        rgb_image = im[:, :, :3]  # Extract only the RGB part
    else:
        rgb_image = im  # If it's already RGB, no change needed

    # Prepare image tensor for ISNetDIS model
    input_size = [1024, 1024]
    im_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    if torch.cuda.is_available():
        image = image.cuda()

    # Run ISNetDIS model
    result = net(image)
    result = torch.squeeze(F.interpolate(result[0][0], rgb_image.shape[0:2], mode='bilinear'), 0)

    # Normalize result to [0, 1]
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)

    # Create a binary mask: foreground = 1, background = 0
    mask = result > 0.2  # Threshold to refine foreground (adjustable for sharper edges)

    # Convert mask to NumPy array (if it's a tensor)
    mask_np = mask.cpu().numpy()  # Move to CPU and convert to NumPy

    # Apply Gaussian blur to smooth the edges (adjustable sigma for edge softness)
    mask_blurred = gaussian_filter(mask_np.astype(np.float32), sigma=1)

    # Normalize blurred mask to range [0, 1]
    mask_blurred = np.clip(mask_blurred, 0, 1)

    # Ensure dimensions match: Add extra dimension to mask_blurred for broadcasting
    mask_blurred = mask_blurred.squeeze()  # Remove extra dimensions
    mask_blurred = mask_blurred[:, :, np.newaxis]  # Add singleton dimension for broadcasting

    # Apply mask to the original image
    rgb_image_np = rgb_image / 255.0  # Normalize to [0, 1] for blending
    masked_image = rgb_image_np * mask_blurred  # Mask applied correctly

    # Convert the masked image back to [0, 255] and ensure integer type
    final_image = (masked_image * 255).astype(np.uint8)

    # Create an alpha channel based on the mask
    alpha_channel = (mask_blurred.squeeze() * 255).astype(np.uint8)  # Remove extra dimension for alpha

    # Combine RGB and alpha into RGBA format
    final_image_rgba = np.dstack((final_image, alpha_channel))  # Combine RGB and alpha

    return final_image_rgba

# Inference
def remove_bg_ormbg(image_path):
    net = ORMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(MODEL_PATH,weights_only=False))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",weights_only=False))
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

    return no_bg_image

# Convert a PIL image to NumPy array
def pil_to_numpy(image):
    return np.array(image)

# Convert NumPy array to RGBA (if not already)
def convert_to_rgba(image):
    if isinstance(image, Image.Image):  # Check if input is a PIL image
        image = pil_to_numpy(image)
    if image.shape[2] == 3:  # RGB
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255
        rgba_image = cv2.merge((image, alpha_channel))  # Add alpha channel
    elif image.shape[2] == 4:  # Already RGBA
        rgba_image = image
    else:
        raise ValueError("Unsupported image format.")
    return rgba_image

# Resize an image to match the target shape
def resize_image_to_target(image, target_shape):
    target_height, target_width = target_shape[:2]
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

# Detect content type based on features
def detect_content_type(image):
    # Load YOLO configuration and weights
    yolo_config = os.path.join('models', 'yolov3.cfg')
    yolo_weights = os.path.join('models', 'yolov3.weights')
    yolo_classes = os.path.join('models', 'coco.names')

    # Load class names
    with open(yolo_classes, 'r') as f:
        class_names = f.read().strip().split('\n')

    # Define categories
    animal_classes = {
        'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'bird', 'monkey', 'rabbit', 'tiger', 'lion', 'deer', 'fox', 'panda', 'camel',
        'goat', 'kangaroo', 'penguin', 'crocodile', 'dolphin', 'whale', 'shark', 'sofa', 'chair'
    }

    product_classes = {'bottle', 'laptop', 'tvmonitor', 'book', 'cell phone', 'backpack'}

    # Load YOLO network
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if CUDA is installed

    # Prepare the image for detection
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()

    # Handle scalar or array output
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, (np.ndarray, list)):
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    else:
        output_layers = [layer_names[unconnected_out_layers - 1]]  # When scalar

    # Forward pass to get detections
    detections = net.forward(output_layers)

    detected_category = 'other'  # Default category
    for detection in detections:
        for obj in detection:
            scores = obj[5:]  # Class confidence scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check confidence threshold
            if confidence > 0.5:
                detected_class = class_names[class_id]

                # Check if it's a person
                if detected_class == 'person':
                    print(f"Detected a person: {detected_class}")
                    return 'organic'

                # Check if it's an animal
                if detected_class in animal_classes:
                    print(f"Detected an animal: {detected_class}")
                    return 'organic'

                # Check if it's a product
                if detected_class in product_classes:
                    print(f"Detected a product item: {detected_class}")
                    return 'rigid'

    print("No relevant object detected.")
    return detected_category


# Compare ISNetDIS and ORMBG results
def compare_images(image1, image2, original_image, content_type=None):
    # Convert all images to RGBA
    image1 = convert_to_rgba(image1)
    image2 = convert_to_rgba(image2)
    original_image = convert_to_rgba(original_image)

    # Resize to original image dimensions
    target_shape = original_image.shape
    image1_resized = resize_image_to_target(image1, target_shape)
    image2_resized = resize_image_to_target(image2, target_shape)

    # Ensure shapes match
    if image1_resized.shape != image2_resized.shape or image1_resized.shape != original_image.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Explicit bias based on content type
    if content_type == 'organic':
        print("Content detected as organic. Prioritizing ORMBG.")
        return image2_resized, 'ORMBG'
    elif content_type == 'rigid':
        print("Content detected as rigid. Prioritizing ISNetDIS.")
        return image1_resized, 'ISNetDIS'
    else:
        return image1_resized, 'ISNetDIS'
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            raise ValueError("No file part")
        
        file = request.files['file']
        if file.filename == '':
            raise ValueError("No selected file")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the original image for comparison
        original_image = imread(filepath)

        
        # Process the image using both ISNetDIS and ORMBG
        isnet_result = remove_bg_isnet(filepath)
        ormbg_result = remove_bg_ormbg(filepath)

        
        # # Validate and process the image using process_image
        # processed_image, best_method = process_image(original_image, isnet_result, ormbg_result)

        # if processed_image is None:
        #     raise ValueError('Invalid image format. Please upload a valid image.')
        best_result, best_method = compare_images(isnet_result, ormbg_result, original_image)

        # Detect content type (organic or rigid)
        content_type = detect_content_type(original_image)

        # Compare the two results based on PSNR, SSIM, and content type
        best_result, best_method = compare_images(isnet_result, ormbg_result, original_image, content_type)

        # Save the best result
        base_filename = os.path.splitext(filename)[0]
        result_filename = f"{base_filename}_{best_method.lower()}_result.png"
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # Convert NumPy array to an image and save it
        Image.fromarray(best_result).save(result_filepath)

        # Render the result on the frontend
        return render_template('index.html', uploaded_image=filename, result_image=result_filename)

    except Exception as e:
        # If any error occurs, show the error message in the template
        # error_message = str(e)
        return render_template('index.html', error_message="An error occurred while processing your request. Please ensure the file is valid and try again.")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 2000))
    app.run(debug=True, host='0.0.0.0', port=port)


from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def prewitt_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if np.max(magnitude) == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def roberts_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if np.max(magnitude) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def laplace_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    laplace = cv2.Laplacian(gray, cv2.CV_32F)
    if np.max(np.abs(laplace)) == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    laplace = cv2.normalize(
        np.abs(laplace), None, 0, 255, cv2.NORM_MINMAX
    )
    return laplace.astype(np.uint8)

def freichen_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    kx = np.array([
        [1, np.sqrt(2), 1],
        [0, 0, 0],
        [-1, -np.sqrt(2), -1]
    ], dtype=np.float32)

    ky = np.array([
        [1, 0, -1],
        [np.sqrt(2), 0, -np.sqrt(2)],
        [1, 0, -1]
    ], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    magnitude = cv2.magnitude(gx, gy)
    if np.max(magnitude) == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    )
    return magnitude.astype(np.uint8)

def canny_edge_detection(image, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(blur, lower, upper)

def log_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX)
    return log.astype(np.uint8)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    result_image = None
    filename = None

    if request.method == 'POST':

        # ================== RESET ==================
        if 'reset' in request.form:
            filename = request.form.get('filename')
            if filename:
                original_image = f'uploads/{filename}'
            return render_template(
                'index.html',
                original_image=original_image,
                result_image=None,
                filename=filename
            )

        # ================== DETEKSI ==================
        method = request.form['method']
        filename = request.form.get('filename')

        # upload hanya sekali
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = str(uuid.uuid4()) + '.png'
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)

        img = cv2.imread(upload_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if method == 'sobel':
            edge = sobel_edge_detection(img)

        elif method == 'prewitt':
            edge = prewitt_edge_detection(img)

        elif method == 'roberts':
            edge = roberts_edge_detection(img)

        elif method == 'laplace':
            edge = laplace_edge_detection(img)

        elif method == 'freichen':
            edge = freichen_edge_detection(img)

        elif method == 'canny':
            edge = canny_edge_detection(img)

        elif method == 'log':
            edge = log_edge_detection(img)

        elif method == 'reset':
            original_image = f'uploads/{filename}'
            return render_template(
                'index.html',
                original_image=original_image,
                result_image=None,
                filename=filename
            )


        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(result_path, edge)

        original_image = f'uploads/{filename}'
        result_image = f'results/{filename}'

    return render_template(
        'index.html',
        original_image=original_image,
        result_image=result_image,
        filename=filename
    )

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'  # Needed for session if you add messages

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def splash():
    # Splash screen — meta refresh in splash.html will take user to /spider_check
    return render_template('splash.html')


@app.route('/spider_check', methods=['GET', 'POST'])
def spider_check():
    if request.method == 'POST':
        selected_images = request.form.getlist('images')
        correct_images = {'web1.jpg', 'web2.jpg', 'web3.jpg', 'web4.jpg'}

        if set(selected_images) == correct_images:
            return redirect(url_for('upload_file'))
        else:
            return render_template(
                'spider_check.html',
                error="Verification failed. Try again!"
            )

    # First GET request shows captcha page
    return render_template('spider_check.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', error="No file selected")

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Analyze the uploaded web image
            closed_count, description = analyze_web(filepath)

            return render_template(
                'result.html',
                closed_count=closed_count,
                description=description
            )

    return render_template('upload.html')


def analyze_web(image_path, save_debug=True):
    """Analyze a spider web image and estimate how many flies it could catch."""

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, "Could not read image"

    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Morphological closing to connect gaps in web lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional dilation to further connect thin/broken lines
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Find contours and hierarchy (to detect nested/closed shapes)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    closed_count = 0
    min_area = (img.shape[0] * img.shape[1]) * 0.0005  # Area filter: 0.05% of image size

    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Flatten the hierarchy list
        for i, h in enumerate(hierarchy):
            # h[3] != -1 → has a parent → likely a closed inner loop
            if h[3] != -1 and cv2.contourArea(contours[i]) > min_area:
                closed_count += 1

    # Estimate fly-catching potential
    estimated_flies = max(0, closed_count * 2 - 2)
    description = f"This web can catch approximately {estimated_flies} flies!"

    # Optional: Save debug image showing detected loops
    if save_debug:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, h in enumerate(hierarchy):
            if h[3] != -1 and cv2.contourArea(contours[i]) > min_area:
                cv2.drawContours(debug_img, contours, i, (0, 255, 0), 2)

        debug_path = os.path.join(os.path.dirname(image_path), "debug_output.jpg")
        cv2.imwrite(debug_path, debug_img)

    return closed_count, description




if __name__ == '__main__':
    app.run(debug=True)

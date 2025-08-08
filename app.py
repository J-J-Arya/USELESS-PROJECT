from flask import Flask, render_template, request, redirect, url_for
import os
import cv2

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


def analyze_web(image_path):
    """Analyze uploaded spider web image and estimate flies it can catch."""

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, "Could not read image"

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    closed_count = 0
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1 and cv2.contourArea(contours[i]) > 100:
                closed_count += 1

    estimated_flies = closed_count * 3
    description = f"This web can catch approximately {estimated_flies} flies!"

    return closed_count, description


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Splash screen
@app.route('/')
def splash():
    return render_template('splash.html')

# Spider CAPTCHA (spider_check.html)
@app.route('/spider_check', methods=['GET', 'POST'])
def spider_check():
    if request.method == 'POST':
        answers = request.form.getlist('images')
        correct_answers = {'web1.jpg', 'web2.jpg'}  # names of spider web images in /static
        if set(answers) == correct_answers:
            return redirect(url_for('upload'))
        else:
            return render_template('spider_check.html', error="Oops! You're not a spider.")
    return render_template('spider_check.html')

# Upload and process image
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            closed_count = count_closed_figures(filepath)
            return render_template('result.html', count=closed_count)
    return render_template('upload.html')

# Count closed figures
def count_closed_figures(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

if __name__ == "__main__":
    app.run(debug=True)

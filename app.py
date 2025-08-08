# 

import os
from flask import Flask, render_template, request, redirect, url_for
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🧠 Better shape detection using inversion + adaptive threshold + CCOMP
def count_closed_shapes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the image so black lines become white
    inverted = cv2.bitwise_not(gray)

    # Adaptive threshold handles different lighting and thin lines
    thresh = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Dilate to close small gaps in web lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Only count contours with parents (i.e., closed loops)
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(cnt)
                if area > 10:  # Lowered threshold for thin web lines
                    count += 1
    return count

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/spider-check', methods=['GET', 'POST'])
def spider_check():
    if request.method == 'POST':
        legs = request.form.get('legs')
        webs = request.form.get('webs')
        if legs == 'yes' and webs == 'yes':
            return redirect(url_for('upload'))
        else:
            return render_template('spider_check.html', not_spider=True)
    return render_template('spider_check.html', not_spider=False)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            count = count_closed_shapes(filepath)
            return render_template('result.html', count=count)
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

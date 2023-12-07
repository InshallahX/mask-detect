from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, redirect, request, send_from_directory
import os
import torch
from PIL import Image
import io
import cv2
import numpy as np
import random, string

app = Flask(__name__, static_url_path='/static')
run_with_ngrok(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def genRandStr(length):
    letter_and_digit = string.ascii_letters + string.digits
    return ''.join(random.choice(letter_and_digit) for _ in range(length))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        gen_image_name = genRandStr(12) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], gen_image_name)
        file.save(file_path)
        return predict(file_path)

def predict(image_path):
    image_data = Image.open(image_path)
    results = model([np.array(image_data)], size=416)
    output = results.pandas().xyxy[0]
    print(output)

    image = np.array(image_data)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for obj in output.values.tolist():
        xmin, ymin, xmax, ymax, confidence, _, name = obj
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=1)
        img = cv2.putText(image, f'{name} {round(confidence, 2)}', (int(xmin), int(ymax)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 225, 0), 1)

    output_image_path = f'static/results/{genRandStr(12)}.jpg'
    cv2.imwrite(output_image_path, image)

    #return {"Status": "Finish", "output_image_path": output_image_path, "uploaded_image_path": image_path}
    return render_template('index.html', uploaded_image_path=image_path, output_image_path=output_image_path)

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt', force_reload=True, skip_validation=True)
    # Assuming your YOLOv5 model is named 'net'
    #torch.save(model.state_dict(), 'test.pt')
    app.run()

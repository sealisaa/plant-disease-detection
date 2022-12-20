from flask import Flask, render_template, request, redirect
import pickle
import torch
import io
import shutil
from PIL import Image

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

@app.route("/", methods=["GET", "POST"])
def detect():
    img = ""
    resultImg = ""
    hiddenImg = "hidden"
    file = ""
    thresholdValue = 0.5
    disableBtn = "disabled"
    if request.method == "POST":
        shutil.rmtree("static/result/", ignore_errors=True)
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        thresholdValue = float(request.form["threshold"])
        model.conf = thresholdValue
        output = model([img])
        output.render()
        output.save(save_dir="static/result/")
        img = img.save('static/result/input.jpg')
        file = 'static/result/input.jpg'
        resultImg = 'static/result/image0.jpg'
        hiddenImg = ''
    return render_template('index.html', result=resultImg, hidden=hiddenImg, input=file, threshold=thresholdValue, disabled=disableBtn)

if __name__ == "__main__":
    app.run()
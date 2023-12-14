from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import warnings
import blip_model  # Replace with your actual model import
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Function to convert uploaded images to JPG format
def convert_to_jpg(image):
    try:
        img = Image.open(image)
        img = img.convert('RGB')
        jpg_image_path = "static/uploaded_image.jpg"
        img.save(jpg_image_path, format='JPEG')
        return jpg_image_path
    except Exception as e:
        print(f"Error converting image to JPG: {str(e)}")
        return None

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/caption', methods=['GET', 'POST'])
def caption():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                jpg_image_path = convert_to_jpg(image)

                if jpg_image_path:
                    try:
                        blip_model_caption = blip_model.img_predict(jpg_image_path)  # Replace with your model's caption generation code

                        return render_template('index.html', blip_model_caption=blip_model_caption, image_path=jpg_image_path)
                    except Exception as e:
                        return render_template('index.html', blip_model_caption=None, image_path=None, error=f"Error: {str(e)}")

    return render_template('index.html', blip_model_caption=None)

if __name__ == '__main__':
    app.run(debug=True)

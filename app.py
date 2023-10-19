# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import numpy as np
from keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from pickle import load
import warnings
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

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

@app.route('/')
def index():
    return render_template('index.html', caption=None)

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # Convert the uploaded image to JPG format
            jpg_image_path = convert_to_jpg(image)

            if jpg_image_path:
                xception_model = Xception(include_top=False, pooling="avg")
                max_length = 32
                tokenizer = load(open("tokenizer.p", "rb"))
                model = load_model('models/model_9.h5')

                photo = extract_features(jpg_image_path, xception_model)

                description = generate_desc(model, tokenizer, photo, max_length)
                words = description.split()
                new_sentence = ''
                if len(words) >= 3:
                    new_sentence = ' '.join(words[1:-1])
                return render_template('index.html', caption=new_sentence, image_path=jpg_image_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

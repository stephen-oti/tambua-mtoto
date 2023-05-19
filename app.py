from flask import Flask, flash, request, redirect, url_for, render_template
import os
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_similarity(query_image_path, image_folder):
    model = VGG16(weights='imagenet', include_top=False)

    image_files = os.listdir(image_folder)

    # Load and preprocess the query image
    query_image = Image.open(query_image_path)
    query_image = query_image.resize((224, 224))
    query_image = np.expand_dims(query_image, axis=0)
    query_image = preprocess_input(query_image)

    # Extract features for the query image
    query_features = model.predict(query_image)

    similarity_scores = []
    for file in image_files:
        # Load and preprocess the current image
        current_image_path = os.path.join(image_folder, file)
        current_image = Image.open(current_image_path)
        current_image = current_image.resize((224, 224))
        current_image = np.expand_dims(current_image, axis=0)
        current_image = preprocess_input(current_image)

        # Extract features for the current image
        current_features = model.predict(current_image)

        # Calculate cosine similarity between query and current image features
        similarity = cosine_similarity(query_features.reshape(1, -1), current_features.reshape(1, -1))[0][0]
        similarity_scores.append((file, similarity))

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Return the top 5 matches
    return similarity_scores[:5]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Search Results for: ')

        # Get the top 5 matching images
        matches = get_image_similarity(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'static/images')

        # Prepare the results
        results = []
        for match in matches:
            image_name = os.path.splitext(os.path.basename(match[0]))[0]
            similarity_index = round(match[1] * 100, 2)
            image_path = f"static/images/{image_name}.jpg"
    
            results.append({
                'image_name': image_name,
                'similarity_index': similarity_index,
                'image_path': image_path
            })

        return render_template('index.html', filename=filename, results=results)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

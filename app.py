import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from io import BytesIO

app = Flask(__name__)

# Load the face detection and recognition models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embeddings(image):
    img = Image.fromarray(image)
    face_tensors = mtcnn(img)
    if face_tensors is None:
        return None
    embeddings = resnet(face_tensors)
    return embeddings.detach().numpy()

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index.html')

@app.route('/match_faces', methods=['POST'])
def match_faces():
    # Check if image file is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Get the target image embeddings
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_embeddings = get_embeddings(image_rgb)

    if target_embeddings is None:
        return jsonify({'error': 'No faces found in the target image.'}), 400

    # Load the gallery images and compute similarity
    gallery_folder = 'static\\images'
    gallery_embeddings = []
    gallery_filenames = []
    for filename in os.listdir(gallery_folder):
        gallery_image = cv2.imread(os.path.join(gallery_folder, filename))
        if gallery_image is not None:
            gallery_image_rgb = cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB)
            gallery_embedding = get_embeddings(gallery_image_rgb)
            if gallery_embedding is not None:
                gallery_embeddings.append(gallery_embedding)
                gallery_filenames.append(filename)

    if len(gallery_embeddings) == 0:
        return jsonify({'error': 'No faces found in the gallery images.'}), 400

    # Compute the similarity between target and gallery embeddings
    similarities = []
    for i, gallery_embedding in enumerate(gallery_embeddings):
        similarity = np.dot(target_embeddings, gallery_embedding.T)
        similarities.append((gallery_filenames[i], similarity[0][0]))

    # Sort the results by similarity
    sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)

    if len(sorted_results) == 0:
        return jsonify({'message': 'No matches found.'}), 200

    # Get the top 5 matches
    top_matches = sorted_results[:5]
    response_data = []
    for match in top_matches:
        match_filename = match[0]
        match_similarity = match[1]
        response_data.append({
            'filename': match_filename,
            'filepath': os.path.join(gallery_folder, match_filename),
            'similarity': float(match_similarity)
        })

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run()

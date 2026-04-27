from flask import Flask, request, jsonify
from flask_cors import CORS
from tmdb_files import tmdb_algorithm as tmdb
import os

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file.save(os.path.join("./~input_photos_temp", file.filename))
        IMG_PATH = os.path.join("./~input_photos_temp", file.filename)
        SAVE_PATH = './~extracted_photos_temp'
        response = tmdb.execute_program_on_image(img_path=IMG_PATH, save_path=SAVE_PATH)
        
        for filename in os.listdir(SAVE_PATH): # remove files after running model
            file_path = os.path.join(SAVE_PATH, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return response

if __name__ == "__main__":
    app.run(port=5000)
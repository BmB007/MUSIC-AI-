from flask import Flask, render_template, request, jsonify, send_file
from chord_extraction_system import ChordExtractionSystem
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

chord_system = ChordExtractionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    result = chord_system.process_song(filepath)
    if not result:
        return jsonify({'error': 'Processing failed'}), 500

    visualization_url = f"/visualization/{os.path.basename(result['visualization'])}"
    return jsonify({
        'chord_chart': result['chord_chart'],
        'visualization': visualization_url
    })

@app.route('/visualization/<filename>')
def serve_visualization(filename):
    return send_file(os.path.join(os.path.dirname(__file__), filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

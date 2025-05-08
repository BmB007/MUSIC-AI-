Great! Since you’ve trained and saved your own `.h5` model for chord prediction, I’ll update the `README.md` to reflect that and show how to use it within the project.

---

### ✅ Updated `README.md`

```markdown
# 🎵 Chord Extraction Web Application (with Custom Trained Model)

A web app to extract chords from audio files using your **custom trained neural network model** (`.h5`). It displays chord charts and visualizes the chord progression for both piano and guitar.

---

## 📁 Project Structure

```

project/
├── app.py                          # Flask backend
├── chord\_extraction\_system.py     # Core logic (includes your trained model loader)
├── your\_model.h5                  # ✅ Your custom trained model (place it here)
├── uploads/                       # Uploaded audio files
├── results/                       # Generated visual outputs
└── templates/
└── index.html                 # Web interface

````

---

## ⚙️ Requirements

- Python 3.7+
- Virtualenv (recommended)

### 📦 Install dependencies

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
````

Install packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```txt
Flask
tensorflow
librosa
matplotlib
numpy
pandas
scikit-learn
```

---

## 🚀 How to Run

1. **Place your model**: Make sure your model file (e.g., `your_model.h5`) is placed in the root directory.

2. **Update `app.py`** to load your model:

```python
chord_system = ChordExtractionSystem(model_path="your_model.h5")
```

3. **Start the app**:

```bash
python app.py
```

4. Open in your browser: [http://localhost:5000](http://localhost:5000)

---

## 🧠 How It Works

* Upload an audio file
* The app extracts chroma features and predicts chords using your `.h5` model
* It shows:

  * Timestamps and chord labels
  * Fingering for guitar and piano
  * A visual timeline of chord progression (PNG)

---

## 🎯 Tips

* Train your `.h5` model using `ChordExtractionSystem.create_synthetic_dataset()` or real audio data.
* Improve predictions by training with real-world chromagram features from annotated datasets (e.g., Beatles Chords).
* Customize the `segment_size` and `window_size` in `chord_extraction_system.py`.

---

## 📷 Example Output

```
00:00 - 00:02: Cmaj7
  Guitar: (x,3,2,0,0,0)
  Piano: (C, E, G, B)
...
```

---

## 🧑‍💻 Author

**Bharath Bala**
Founder, Web Crafters Foundation
📧 [bharathbala1503@gmail.com](mailto:bharathbala1503@gmail.com)
🌐 wcff.in

---

## 📄 License

This project is for academic and research purposes.  ALL RIGHTS RESERVED TO wcff.in

```

---

Let me know:
- What is the exact filename of your model? (`chord_recognition_model.h5`?)
- Do you want the `requirements.txt` generated for this setup?
```

# ✅ Setup Checklist

Use this checklist to ensure everything is ready before running the app.

---

## 📋 Pre-Installation

- [ ] **Python 3.8+** installed
  ```bash
  python --version
  # Should show: Python 3.8.x or higher
  ```

- [ ] **Webcam** working
  - Test with built-in camera app
  - Make sure no other app is using it

- [ ] **Internet connection** (for Gemini API)

---

## 📦 Installation Steps

- [ ] **Downloaded/Cloned project**
  ```bash
  cd Deep_learning_Project
  ```

- [ ] **Installed requirements**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Verified Streamlit installation**
  ```bash
  streamlit --version
  # Should show version number
  ```

- [ ] **Verified TensorFlow installation**
  ```bash
  python -c "import tensorflow; print(tensorflow.__version__)"
  # Should show version number
  ```

---

## 🤖 Model Setup

- [ ] **Have a trained model** (`.h5`, `.keras`, `.pt`, `.pth`, or `.onnx`)

- [ ] **Placed model in `models/` folder**
  ```
  models/
  └── [your_model_file]
  ```

- [ ] **Renamed model to `ayumi_chan`**
  ```bash
  # Example: ayumi_chan.h5
  models/
  └── ayumi_chan.h5  ✅
  ```

- [ ] **Verified model file exists**
  ```bash
  # Windows
  dir models\ayumi_chan.h5
  
  # Mac/Linux
  ls models/ayumi_chan.h5
  ```

---

## 🔑 Gemini API Setup

- [ ] **Got Gemini API key**
  - Visit: https://makersuite.google.com/app/apikey
  - Sign in with Google
  - Create API Key
  - Copy the key

- [ ] **Created `.env` file** in project root
  ```
  Deep_learning_Project/
  ├── .env  ← This file
  ├── app.py
  ├── ...
  ```

- [ ] **Added API key to `.env`**
  ```env
  GEMINI_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  ```

- [ ] **Verified `.env` file format**
  - ✅ No spaces around `=`
  - ✅ No quotes around API key
  - ✅ Key starts with `AIzaSy`

- [ ] **Verified `.env` file exists**
  ```bash
  # Windows
  type .env
  
  # Mac/Linux
  cat .env
  ```

---

## 🚀 Ready to Launch

- [ ] **All above steps completed**

- [ ] **Run the app**
  ```bash
  streamlit run app.py
  ```

- [ ] **App opened in browser** at `http://localhost:8501`

- [ ] **No error messages** in terminal

- [ ] **Model loaded successfully**
  - Check app sidebar for model info

---

## 🎯 First-Time Usage

- [ ] **Camera permission granted** in browser

- [ ] **Camera shows video feed**

- [ ] **Hand detection working** (yellow points + cyan lines on hand)

- [ ] **Prediction working** (shows letter when making sign)

- [ ] **Gemini API working** (Translation mode functional)

---

## 📊 Optional Training

If you want to train your own model:

- [ ] **Have ASL dataset** in `datasets/asl_dataset/`
  - Folders: 0-9, a-z
  - Each folder has training images

- [ ] **Run training script**
  ```bash
  python train_improved_model.py
  ```

- [ ] **Training completed** (30-60 minutes)

- [ ] **Model saved** as `models/resnet50_improved.h5`

- [ ] **Renamed to `ayumi_chan.h5`**
  ```bash
  cd models
  rename resnet50_improved.h5 ayumi_chan.h5  # Windows
  mv resnet50_improved.h5 ayumi_chan.h5      # Mac/Linux
  ```

---

## 🆘 If Something's Wrong

### Model Issues
- [ ] Checked model file name is exactly `ayumi_chan` (+ extension)
- [ ] Verified model file is not corrupted
- [ ] Tried different model format (.h5, .keras, etc.)

### API Issues
- [ ] Verified API key is correct
- [ ] Checked `.env` file format (no spaces, no quotes)
- [ ] Tested API key at https://makersuite.google.com

### Camera Issues
- [ ] Granted camera permission in browser
- [ ] Closed other apps using camera
- [ ] Tried different browser (Chrome recommended)
- [ ] Refreshed the page (F5)

### Installation Issues
- [ ] Updated pip: `pip install --upgrade pip`
- [ ] Reinstalled requirements: `pip install -r requirements.txt --force-reinstall`
- [ ] Checked Python version: `python --version` (should be 3.8+)

---

## 📚 Documentation

Still having issues? Check these guides:

- [ ] **README.md** - Full documentation
- [ ] **QUICKSTART.md** - Quick start guide
- [ ] **Troubleshooting section** in README.md

Or create an issue:
- https://github.com/Chonapatcc/Deep_learning_Project/issues

---

**All checked? You're ready to go! 🎉**

```bash
streamlit run app.py
```

**Happy learning! 🤟**

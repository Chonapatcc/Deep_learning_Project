# 🎯 One-Page Setup Guide

**ASL Fingerspelling Trainer - Setup in 5 Minutes!**

---

## 📋 What You Need

| Item | Description | Required? |
|------|-------------|-----------|
| 🐍 Python 3.8+ | Programming language | ✅ Required |
| 📷 Webcam | Built-in or USB camera | ✅ Required |
| 🤖 Trained Model | `.h5`, `.keras`, `.pt`, or `.onnx` file | ✅ Required |
| 🔑 Gemini API Key | For translation mode | ⚠️ Optional |
| 💻 Modern Browser | Chrome, Edge, or Firefox | ✅ Required |

---

## 🚀 Setup Steps

### 1. Install Python Packages

```bash
pip install -r requirements.txt
```

**Time:** ~5-10 minutes | **One-time only**

---

### 2. Add Your Model

```bash
# Place your model file in models/ folder
# Rename to: ayumi_chan.h5 (or .keras, .pt, .onnx)

models/
└── ayumi_chan.h5  ← Your trained model
```

**Rename command:**
```bash
# Windows
rename models\your_model.h5 models\ayumi_chan.h5

# Mac/Linux  
mv models/your_model.h5 models/ayumi_chan.h5
```

---

### 3. Setup Gemini API (Optional)

**Get free API key:** https://makersuite.google.com/app/apikey

**Create `.env` file:**
```env
GEMINI_API_KEY=your_actual_api_key_here
```

**Quick create:**
```bash
# Windows
echo GEMINI_API_KEY=AIzaSy... > .env

# Mac/Linux
echo "GEMINI_API_KEY=AIzaSy..." > .env
```

---

### 4. Run the App

```bash
streamlit run app.py
```

**Open browser:** http://localhost:8501

---

## ✅ Success Indicators

| What to Check | Expected Result |
|--------------|-----------------|
| Terminal output | `Local URL: http://localhost:8501` |
| Browser | App loads successfully |
| Sidebar | Model info displayed |
| Camera | Video feed shows |
| Hand detection | Yellow points + cyan lines on hand |
| Prediction | Shows letter when making sign |

---

## 🆘 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| ❌ Model not found | Check `models/ayumi_chan.h5` exists |
| ❌ Streamlit not found | Run: `pip install streamlit` |
| ❌ Camera not working | Allow camera in browser settings |
| ❌ API key error | Check `.env` file format (no spaces, no quotes) |
| ❌ Import errors | Run: `pip install -r requirements.txt --force-reinstall` |

---

## 🎮 Quick Start After Setup

### Try Each Mode:

**1. 📚 Learning Mode**
- Click on any letter (A-Z)
- See example image and instructions

**2. ✋ Practice Mode**  
- Select a letter
- Enable camera
- Make the sign
- Get instant feedback

**3. 🎯 Test Mode**
- Take quiz on A-Z
- See your score
- Track progress

**4. 🌐 Translation Mode** (needs Gemini API)
- Sign letters
- Auto-translate to text
- AI refines sentences

---

## 📊 Expected Performance

| Metric | Target |
|--------|--------|
| Hand Detection | < 100ms |
| Prediction Speed | < 500ms |
| Model Accuracy | > 90% |
| FPS | > 15 fps |

---

## 💡 Pro Tips

### For Best Results:
- ✅ Good lighting (natural light best)
- ✅ Simple background
- ✅ Hand 20-30% of screen size
- ✅ Slow, clear movements
- ✅ Palm facing camera

### Avoid:
- ❌ Backlit environment
- ❌ Complex backgrounds
- ❌ Fast movements
- ❌ Multiple hands
- ❌ Hand too small or out of frame

---

## 📁 File Structure

```
Deep_learning_Project/
├── .env                    ← Your API key
├── app.py                  ← Main application
├── requirements.txt        ← Dependencies
├── models/
│   └── ayumi_chan.h5      ← Your model
├── utils/
│   ├── model_loader.py    ← Model loading
│   ├── preprocessing.py   ← Image processing
│   └── prediction.py      ← Prediction logic
└── datasets/
    └── asl_dataset/       ← Training data (optional)
```

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| Get Gemini API Key | https://makersuite.google.com/app/apikey |
| Python Download | https://www.python.org/downloads/ |
| Chrome Browser | https://www.google.com/chrome/ |
| GitHub Issues | https://github.com/Chonapatcc/Deep_learning_Project/issues |

---

## 📚 Full Documentation

For detailed information, see:
- **[README.md](README.md)** - Complete guide
- **[QUICKSTART.md](QUICKSTART.md)** - 3-step quick start
- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Detailed checklist

---

## 🎓 Training Your Own Model

**Quick training command:**
```bash
python train_improved_model.py
```

**Requirements:**
- Dataset in `datasets/asl_dataset/`
- 30-60 minutes time
- GPU recommended (but not required)

**Output:**
- `models/resnet50_improved.h5` (rename to `ayumi_chan.h5`)
- ~94-96% test accuracy
- ~80-90% real-world accuracy

---

**Ready? Let's go! 🚀**

```bash
streamlit run app.py
```

**Happy Learning ASL! 🤟**

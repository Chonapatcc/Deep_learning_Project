# ASL Fingerspelling Trainer 🤟

แอปพลิเคชันเว็บสำหรับฝึกทำท่าภาษามือตัวอักษรภาษาอังกฤษ (American Sign Language Fingerspelling) โดยใช้เทคโนโลยี AI และ Computer Vision

**🚀 Quick Demo/Prototype สร้างด้วย Streamlit - ใช้งานง่าย รวดเร็ว!**

> 📚 **New here?** See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for all guides!  
> ⚡ **Want quick setup?** Go to [ONE_PAGE_SETUP.md](ONE_PAGE_SETUP.md)!

---

## 📖 Table of Contents
- [Quick Start](#-quick-start-3-easy-steps) ⚡
- [Features](#-คุณสมบัติหลัก)
- [Technology](#-เทคโนโลยีที่ใช้)
- [Installation](#-การติดตั้งและใช้งาน)
- [Usage Guide](#-วิธีใช้งาน)
- [Troubleshooting](#-การแก้ปัญหา)
- [Documentation](#-documentation)

---

## 📚 Documentation

### 🚀 Getting Started
- **[ONE_PAGE_SETUP.md](ONE_PAGE_SETUP.md)** - ⚡ Setup ใน 1 หน้า (แนะนำ!)
- **[QUICKSTART.md](QUICKSTART.md)** - เริ่มใช้งานใน 3 ขั้นตอน
- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Checklist ตรวจสอบก่อนเริ่มใช้งาน
- **[.env.example](.env.example)** - ตัวอย่างไฟล์ .env

### 🤖 Model & Training  
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - วิธี train model แบบละเอียด
- **[MULTI_FRAMEWORK_GUIDE.md](MULTI_FRAMEWORK_GUIDE.md)** - ใช้ PyTorch/ONNX models
- **[MOBILENETV2_PREPROCESSING.md](MOBILENETV2_PREPROCESSING.md)** - เพิ่มประสิทธิภาพด้วย MobileNetV2 ⭐
- **[DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md)** - ข้อกำหนด dataset

### 🎨 Features & Customization
- **[TRANSLATION_MODE_GUIDE.md](TRANSLATION_MODE_GUIDE.md)** - คู่มือโหมด Translation
- **[SKELETON_COLOR_GUIDE.md](SKELETON_COLOR_GUIDE.md)** - ปรับแต่งสี skeleton
- **[ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md)** - ตั้งค่า .env และ Gemini API

### 📝 Project Info
- **[CHANGELOG.md](CHANGELOG.md)** - ประวัติการอัปเดต
- **[README.md](README.md)** - เอกสารฉบับเต็ม (ไฟล์นี้)

---

## ⚡ Quick Start (3 Easy Steps)

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Place Model File                                      │
├─────────────────────────────────────────────────────────────────┤
│  models/                                                        │
│  └── ayumi_chan.h5  ← Your trained model here!                 │
│                                                                 │
│  Supported: .h5, .keras, .pt, .pth, .onnx                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Setup Gemini API Key                                  │
├─────────────────────────────────────────────────────────────────┤
│  Create file: .env                                              │
│  Content:     GEMINI_API_KEY=your_api_key_here                  │
│                                                                 │
│  🔑 Get free key: https://makersuite.google.com/app/apikey      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Install & Run                                          │
├─────────────────────────────────────────────────────────────────┤
│  $ pip install -r requirements.txt                              │
│  $ streamlit run app.py                                         │
│                                                                 │
│  ✅ Open: http://localhost:8501                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Instructions:

### 1️⃣ **Place Your Trained Model**
```bash
# Place your trained model file in the models/ folder
# Rename it to: ayumi_chan.h5 (or ayumi_chan.keras, ayumi_chan.pt, etc.)

models/
└── ayumi_chan.h5    # Your model here
```

**Rename file:**
```bash
# Windows
cd models
rename your_model.h5 ayumi_chan.h5

# Mac/Linux
cd models
mv your_model.h5 ayumi_chan.h5
```

### 2️⃣ **Setup Gemini API Key**
```bash
# Create a .env file in the project root
# Add your Gemini API key:

GEMINI_API_KEY=your_actual_api_key_here
```

🔑 **Get your free Gemini API key**: https://makersuite.google.com/app/apikey

**How to create .env file:**

**Windows (Notepad):**
1. Open Notepad
2. Type: `GEMINI_API_KEY=your_actual_api_key_here`
3. Save As → All Files → `.env`

**Command line:**
```bash
# Windows
echo GEMINI_API_KEY=your_actual_api_key_here > .env

# Mac/Linux
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

### 3️⃣ **Install & Run**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**That's it! 🎉** Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

โครงสร้างโปรเจคที่จัดระเบียบและเข้าใจง่าย เหมาะสำหรับการพัฒนาและ reproducible research:

```
Deep_learning_Project/
├── 📂 data/                    # ชุดข้อมูลสำหรับการ training
│   ├── raw/                   # ข้อมูลดิบ (ถ้ามี)
│   ├── processed/             # ข้อมูลที่ประมวลผลแล้ว
│   └── README.md              # คำแนะนำเกี่ยวกับข้อมูล
│
├── 📂 src/                     # Source code หลัก
│   ├── classifier.py          # ⭐ ASL Classifier architecture (ResNet-based)
│   ├── dataset.py             # ⭐ Dataset handler
│   ├── config.py              # Configuration settings
│   ├── __init__.py
│   │
│   ├── 📂 models/             # ⚠️ Organization only (ว่างเปล่า)
│   │   ├── __init__.py
│   │   └── README.md
│   │
│   ├── 📂 controllers/        # Logic สำหรับ training/prediction
│   │   ├── trainer.py        # Training logic
│   │   ├── predictor.py      # Prediction logic
│   │   ├── evaluator.py      # Model evaluation
│   │   └── __init__.py
│   │
│   ├── 📂 utils/              # Utility functions
│   │   ├── model_loader.py   # โหลดโมเดลต่างๆ (TF/PyTorch/ONNX)
│   │   ├── preprocessing.py  # Preprocessing frames
│   │   ├── prediction.py     # Prediction utilities
│   │   ├── hand_processing.py # Hand landmark processing
│   │   ├── letter_data.py    # ข้อมูลตัวอักษร A-Z
│   │   ├── confirmation.py   # Confirmation manager
│   │   ├── pytorch_utils/    # PyTorch-specific utilities
│   │   │   ├── preprocessor.py
│   │   │   └── data_handler.py
│   │   └── __init__.py
│   │
│   ├── 📂 views/              # Display/Visualization
│   │   ├── camera_view.py    # Camera display
│   │   ├── visualizer.py     # Visualization tools
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📂 models/                  # ⭐ Model weights ONLY (.pth, .pkl files)
│   ├── best_asl_model2.pth    # PyTorch model weights
│   ├── label_encoder2.pkl     # Label encoder
│   ├── asl_processed2.pkl     # Processed data
│   └── README.md              # คำอธิบายโมเดล
│
├── 📂 demos/                   # Demo videos/screenshots
│   ├── easy.mp4               # Scenario 1: Easy
│   ├── medium.mp4             # Scenario 2: Medium
│   ├── hard.mp4               # Scenario 3: Hard
│   └── README.md              # คำอธิบาย demos
│
├── 📄 app.py                   # 🚀 Main Streamlit Application
├── 📄 requirements.txt         # Python dependencies
├── 📄 .env                     # Environment variables (API keys)
├── 📄 .env.example             # ตัวอย่างไฟล์ .env
├── 📄 README.md                # เอกสารหลัก (ไฟล์นี้)
├── 📄 CHANGELOG.md             # ประวัติการอัปเดต
└── 📄 todo.txt                 # Task list

📚 Documentation/ (ย้ายไปยัง root level)
├── ONE_PAGE_SETUP.md           # Setup ใน 1 หน้า
├── QUICKSTART.md               # Quick start guide
├── TRAINING_GUIDE.md           # Training guide
├── TRANSLATION_MODE_GUIDE.md   # Translation mode guide
└── ... (ดูรายละเอียดทั้งหมดในส่วน Documentation)
```

### 🎯 โครงสร้างนี้มีประโยชน์อย่างไร?

- ✅ **Reproducible**: ใครก็สามารถ clone และรันได้ทันที
- ✅ **Maintainable**: แยกส่วนต่างๆ ชัดเจน แก้ไขง่าย
- ✅ **Scalable**: เพิ่มฟีเจอร์ใหม่ได้สะดวก
- ✅ **Academic-friendly**: เหมาะสำหรับ research และ presentation
- ✅ **Industry-standard**: ตามมาตรฐานโครงสร้าง ML project

---

## 🎯 วัตถุประสงค์

ช่วยให้ผู้เริ่มต้นสามารถเรียนรู้และฝึกฝนการสะกดคำด้วยมือ (fingerspelling) ได้อย่างมีประสิทธิภาพ พร้อมรับผลตอบรับที่ชัดเจนและทันที

## ✨ คุณสมบัติหลัก

### 📚 โหมดเรียนรู้ (Learning Mode)
- ดูตัวอย่างท่ามือตัวอักษร A-Z
- คำแนะนำวิธีทำท่าแต่ละตัวอักษร
- ติดตามความคืบหน้าการฝึกฝนแต่ละตัวอักษร

### ✋ โหมดฝึกฝน (Practice Mode)
- ตรวจจับท่ามือแบบ Real-time
- ให้ feedback ทันที (ภายใน 0.5 วินาที)
- แสดงความถูกต้องเป็นเปอร์เซ็นต์
- คำแนะนำในการปรับปรุงท่าทาง
- สถิติการฝึกฝน

### 🎯 โหมดทดสอบ (Test Mode)
- ทดสอบความสามารถทั้ง 26 ตัวอักษร
- จับเวลาการทำแบบทดสอบ
- คำนวณคะแนนและผลการทดสอบ
- บันทึกประวัติการทดสอบ

### 🌐 โหมดแปลภาษาแบบ Real-time (NEW! ✨)
- แปลท่ามือเป็นข้อความอัตโนมัติ
- ส่งข้อความไปยัง **Gemini API** เพื่อปรับปรุง
- รับข้อความที่ถูกต้องและมีความหมายกลับมา
- Auto-refine ทุกๆ 5 ตัวอักษร (ตั้งค่าได้)
- ช่วยสื่อสารแบบ real-time ผ่านภาษามือ
- 📖 [อ่านคู่มือโหมด Translation](TRANSLATION_MODE_GUIDE.md)

## 🔧 เทคโนโลยีที่ใช้

- **Streamlit**: Quick web app framework - รวดเร็ว ใช้งานง่าย
- **MediaPipe Hands**: ตรวจจับมือและ 21 keypoints แบบ Real-time
- **TensorFlow/Keras**: Deep learning framework สำหรับ AI model
- **MobileNetV2**: Lightweight preprocessing - เหมาะสำหรับ real-time inference ⭐
- **OpenCV**: ประมวลผลภาพและวิดีโอ
- **Python**: ภาษาหลักในการพัฒนา
- **Gemini API**: AI สำหรับปรับปรุงข้อความในโหมด Translation (ตัวเลือก)

## 📋 ข้อกำหนดระบบ

### ฮาร์ดแวร์
- กล้องเว็บแคมหรือกล้องในตัว
- CPU สมัยใหม่ (ไม่ต้องการ GPU)

### ซอฟต์แวร์
- Python 3.8 ขึ้นไป
- เบราว์เซอร์ที่รองรับ: Chrome, Edge, Firefox

### สภาพแวดล้อม
- แสงสว่างเพียงพอและสม่ำเสมอ
- พื้นหลังที่ไม่ซับซ้อนจนเกินไป

## 🚀 การติดตั้งและใช้งาน

### ข้อกำหนดเบื้องต้น

**ฮาร์ดแวร์:**
- 💻 Computer with webcam
- 📷 กล้องเว็บแคมหรือกล้องในตัว
- 🖥️ CPU สมัยใหม่ (ไม่ต้องการ GPU)

**ซอฟต์แวร์:**
- 🐍 Python 3.8 ขึ้นไป ([Download Python](https://www.python.org/downloads/))
- 🌐 เบราว์เซอร์: Chrome, Edge, หรือ Firefox
- 💡 แสงสว่างเพียงพอในห้อง

---

### 📥 Installation Steps

#### Step 1: Clone/Download โปรเจกต์

**Option A: ใช้ Git (แนะนำ)**
```bash
git clone https://github.com/Chonapatcc/Deep_learning_Project.git
cd Deep_learning_Project
```

**Option B: Download ZIP**
1. ไปที่ https://github.com/Chonapatcc/Deep_learning_Project
2. คลิก "Code" → "Download ZIP"
3. แตกไฟล์และเข้าไปในโฟลเดอร์

---

#### Step 2: ติดตั้ง Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**หมายเหตุ:** การติดตั้งอาจใช้เวลา 5-10 นาที (ขึ้นอยู่กับความเร็วอินเทอร์เน็ต)

**ตรวจสอบการติดตั้ง:**
```bash
# Check if Streamlit is installed
streamlit --version

# Check if TensorFlow is installed
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

#### Step 3: Setup Model File 🤖

**⚠️ สำคัญมาก: คุณต้องมีไฟล์ Model ที่ Train แล้ว!**

1. **วางไฟล์ Model** ในโฟลเดอร์ `models/`
2. **เปลี่ยนชื่อไฟล์** เป็น `ayumi_chan` (ไม่ต้องใส่นามสกุล)

**รูปแบบไฟล์ที่รองรับ:**
- ✅ `.h5` - TensorFlow/Keras (แนะนำ)
- ✅ `.keras` - Keras SavedModel format
- ✅ `.pt` / `.pth` - PyTorch
- ✅ `.onnx` - ONNX Runtime

**ตัวอย่าง:**
```bash
# โครงสร้างโฟลเดอร์ที่ถูกต้อง:
models/
├── ayumi_chan.h5        # ✅ ชื่อถูกต้อง!
└── labels.pkl           # (optional - จะถูกสร้างอัตโนมัติ)

# ❌ ชื่อที่ไม่ถูกต้อง:
models/
└── my_model.h5          # ❌ ต้องเปลี่ยนชื่อเป็น ayumi_chan.h5
```

**วิธีเปลี่ยนชื่อไฟล์:**

**Windows (Command Prompt):**
```cmd
cd models
rename your_model.h5 ayumi_chan.h5
```

**Windows (File Explorer):**
1. เปิดโฟลเดอร์ `models/`
2. คลิกขวาที่ไฟล์ model
3. เลือก "Rename"
4. เปลี่ยนเป็น `ayumi_chan.h5`

**Mac/Linux:**
```bash
cd models
mv your_model.h5 ayumi_chan.h5
```

---

#### Step 4: Setup Gemini API Key 🔑

Gemini API ใช้สำหรับโหมด Translation (แปลภาษามือเป็นข้อความ)

**4.1 รับ API Key (ฟรี!)**
1. ไปที่ https://makersuite.google.com/app/apikey
2. Sign in ด้วย Google Account
3. คลิก "Create API Key"
4. คัดลอก API Key

**4.2 สร้างไฟล์ `.env`**

**Windows (Command Prompt):**
```cmd
# สร้างไฟล์ .env ในโฟลเดอร์โปรเจกต์
echo GEMINI_API_KEY=your_actual_api_key_here > .env
```

**Windows (Notepad):**
1. เปิด Notepad
2. พิมพ์: `GEMINI_API_KEY=your_actual_api_key_here`
3. Save As → เลือก "All Files"
4. ตั้งชื่อ: `.env` (มีจุดข้างหน้า!)
5. Save ในโฟลเดอร์โปรเจกต์

**Mac/Linux:**
```bash
# สร้างไฟล์ .env
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**ตัวอย่างไฟล์ `.env` ที่ถูกต้อง:**
```env
GEMINI_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**ตรวจสอบ:**
```bash
# โครงสร้างโปรเจกต์ที่ถูกต้อง:
Deep_learning_Project/
├── .env                    # ✅ ไฟล์ API key
├── app.py
├── requirements.txt
└── models/
    └── ayumi_chan.h5      # ✅ Model file
```

---

#### Step 5: Run the Application! 🚀

```bash
streamlit run app.py
```

**เมื่อรันสำเร็จ คุณจะเห็น:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

✅ แอปจะเปิดในเบราว์เซอร์อัตโนมัติที่ `http://localhost:8501`

---

### 🎓 Optional: Train Your Own Model

หากต้องการ train model ใหม่จาก dataset ของคุณเอง:

**Quick Training (Improved ResNet50 - แนะนำ):**
```bash
python train_improved_model.py
```

**คุณสมบัติ:**
- ✅ ใช้ ResNet50 + Transfer Learning
- ✅ Data Augmentation แบบครอบคลุม
- ✅ Skeleton Overlay Preprocessing
- ✅ แก้ปัญหา high test accuracy, low real-world performance
- ⏱️ เวลา: 30-60 นาที (GPU แนะนำ)
- 📊 Test Accuracy: ~94-96%
- 🌍 Real-world Accuracy: ~80-90%

**Output:**
```bash
models/
├── resnet50_improved.h5    # Trained model
└── labels.pkl              # Label mapping
```

**หลังจาก train เสร็จ:**
```bash
# เปลี่ยนชื่อ model เป็น ayumi_chan
cd models
rename resnet50_improved.h5 ayumi_chan.h5
```

📖 **อ่านเพิ่มเติม:** 
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - คู่มือ Training แบบละเอียด
- [MULTI_FRAMEWORK_GUIDE.md](MULTI_FRAMEWORK_GUIDE.md) - การใช้ PyTorch/ONNX

---

## 📖 วิธีใช้งาน

### โหมดเรียนรู้ 📚
1. เลือก "Learning Mode" ในแถบด้านข้าง
2. คลิกตัวอักษรที่ต้องการดู
3. ศึกษาวิธีทำท่าและดูภาพตัวอย่าง

### โหมดฝึกฝน ✋
1. เลือก "Practice Mode" ในแถบด้านข้าง
2. เลือกตัวอักษรที่ต้องการฝึก
3. เปิดกล้อง (check "เปิดกล้อง")
4. วางมือในกรอบสีเขียว
5. ทำท่ามือตามตัวอย่าง
6. รับ feedback ทันที

### โหมดทดสอบ 🎯
1. เลือก "Test Mode" ในแถบด้านข้าง
2. คลิก "เริ่มทำแบบทดสอบ"
3. ทำท่ามือตามตัวอักษรที่กำหนด
4. คลิก "ยืนยันคำตอบ" เมื่อพร้อม
5. ดูผลการทดสอบเมื่อทำครบทั้งหมด

## 🎓 คำแนะนำในการใช้งาน

### เพื่อผลลัพธ์ที่ดีที่สุด:
- ✅ ใช้งานในที่มีแสงสว่างเพียงพอ
- ✅ วางมือให้อยู่ในกรอบสีเขียว (Region of Interest)
- ✅ ทำท่าช้าๆ และชัดเจน
- ✅ ให้มือมีขนาดอย่างน้อย 20% ของหน้าจอ
- ✅ หันมือให้กล้องเห็นท่าทางชัดเจน

### หลีกเลี่ยง:
- ❌ แสงย้อนหลังหรือเงามากเกินไป
- ❌ การเคลื่อนไหวเร็วหรือกระตุก
- ❌ พื้นหลังที่มีลวดลายซับซ้อน
- ❌ มือเล็กเกินไปหรืออยู่นอกกรอบ

## 📊 ตัวชี้วัดประสิทธิภาพ

### เป้าหมายของระบบ:
- **ความแม่นยำในการรู้จำ**: ≥ 95%
- **ความแม่นยำรูปทรงมือ**: ≥ 90%
- **ความสอดคล้องจังหวะ**: ≥ 85%
- **ความถูกต้องตำแหน่งมือ**: ≥ 95%
- **เวลาตอบสนอง**: ≤ 500ms (0.5 วินาที)

## 🗂️ โครงสร้างโปรเจกต์

```
Deep_learning_Project/
├── app.py                         # แอปพลิเคชัน Streamlit หลัก
├── requirements.txt               # Python dependencies
├── README.md                      # เอกสารนี้
├── QUICKSTART.md                  # คู่มือเริ่มต้นด่วน
├── preprocess_data.py             # Script ประมวลผลข้อมูล (optional)
├── train_model.py                 # Script train model (optional)
└── assets/                        # รูปภาพและ resources
    └── asl/                       # รูปภาพตัวอย่างท่ามือ A-Z
```

## 🔬 การพัฒนาและปรับปรุง

### สำหรับการพัฒนา AI Model (Optional):

ดูรายละเอียดใน:
- `preprocess_data.py` - สำหรับประมวลผลข้อมูล
- `train_model.py` - สำหรับ train model
- `DATASET_REQUIREMENTS.md` - ข้อกำหนดการเก็บข้อมูล

## 🐛 การแก้ปัญหา

### ❌ Model ไม่พบหรือโหลดไม่ได้

**Error:** `Could not load model: ayumi_chan`

**แก้ไข:**
1. ตรวจสอบว่ามีไฟล์ model ในโฟลเดอร์ `models/`
2. ตรวจสอบชื่อไฟล์ว่าเป็น `ayumi_chan.h5` (หรือ .keras, .pt, .onnx)
3. ตรวจสอบว่าไฟล์ไม่เสียหาย (ลอง train ใหม่)

```bash
# ตรวจสอบไฟล์ในโฟลเดอร์ models/
dir models\      # Windows
ls models/       # Mac/Linux

# ผลลัพธ์ที่ต้องการเห็น:
# ayumi_chan.h5
```

---

### ❌ Gemini API Key ไม่ทำงาน

**Error:** `Invalid API key` หรือ `API key not found`

**แก้ไข:**
1. ตรวจสอบว่ามีไฟล์ `.env` ในโฟลเดอร์โปรเจกต์
2. เปิดไฟล์ `.env` และตรวจสอบว่ามี API key ที่ถูกต้อง
3. **ไม่มีช่องว่าง** รอบๆ `=`
4. **ไม่มี quotes** รอบ API key

**ตัวอย่างที่ถูกต้อง:**
```env
GEMINI_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**ตัวอย่างที่ผิด:**
```env
GEMINI_API_KEY = AIzaSy...           # ❌ มีช่องว่าง
GEMINI_API_KEY="AIzaSy..."           # ❌ มี quotes
GEMINI_API_KEY='AIzaSy...'           # ❌ มี quotes
```

**ทดสอบ API Key:**
```bash
# Windows
type .env

# Mac/Linux
cat .env

# ควรเห็น: GEMINI_API_KEY=AIzaSy...
```

---

### ❌ Streamlit ไม่ทำงาน

**Error:** `streamlit: command not found` หรือ `'streamlit' is not recognized`

**แก้ไข:**
```bash
# ติดตั้ง Streamlit อีกครั้ง
pip install streamlit --upgrade

# ตรวจสอบการติดตั้ง
streamlit --version

# หากยังไม่ได้ ให้ใช้ python -m
python -m streamlit run app.py
```

---

### ❌ กล้องไม่ทำงาน

**Symptoms:** หน้าจอกล้องดำ หรือ "No camera detected"

**แก้ไข:**
1. **ตรวจสอบการเข้าถึงกล้อง:**
   - เบราว์เซอร์จะถามสิทธิ์ → คลิก "Allow"
   - ตรวจสอบ Settings → Privacy → Camera
   
2. **ปิดแอปอื่นที่ใช้กล้อง:**
   - Zoom, Teams, Skype, etc.
   
3. **ลองรีเฟรชหน้าเว็บ** (F5 หรือ Ctrl+R)

4. **ลองเบราว์เซอร์อื่น:**
   - Chrome (แนะนำมากที่สุด)
   - Edge
   - Firefox

---

### ❌ ตรวจจับมือไม่ได้

**Symptoms:** กล้องเปิดแต่ไม่เห็น skeleton บนมือ

**แก้ไข:**
1. ✅ **ตรวจสอบแสงสว่าง** - ควรมีแสงเพียงพอ
2. ✅ **วางมือในกรอบสีเขียว** (Region of Interest)
3. ✅ **ให้มือมีขนาดใหญ่พอ** (~20-30% ของหน้าจอ)
4. ✅ **หันฝ่ามือเข้ากล้อง** ให้เห็นท่าทางชัดเจน
5. ✅ **หลีกเลี่ยงพื้นหลังที่ซับซ้อน** หรือสีคล้ายผิวมือ

---

### ❌ ความแม่นยำในการทำนายต่ำ

**Symptoms:** Model ทำนายผิดบ่อย

**แก้ไข:**
1. **ทำท่าช้าๆ และชัดเจน**
2. **ศึกษาท่ามือให้ถูกต้อง** ใน Learning Mode
3. **ให้มือนิ่ง** ไม่สั่นหรือกระตุก
4. **ตรวจสอบท่าทาง** ว่าตรงตามตัวอย่าง
5. **ปรับแสง** ให้เหมาะสม (ไม่มืดหรือสว่างจ้าเกินไป)

**หาก accuracy ต่ำมาก (< 50%):**
- Model อาจต้อง train ใหม่ด้วย dataset ที่ดีกว่า
- ลองใช้ `train_improved_model.py`

---

### ❌ แอปช้า หรือค้าง

**Symptoms:** FPS ต่ำ, กระตุก, delay สูง

**แก้ไข:**
1. **ปิดแท็บอื่นๆ ในเบราว์เซอร์**
2. **ใช้ Chrome** (performance ดีที่สุด)
3. **ปิดโปรแกรมพื้นหลัง** ที่ไม่จำเป็น
4. **ลดความละเอียดกล้อง** ใน config.py:
   ```python
   CAMERA_WIDTH = 640   # ลดจาก 1280
   CAMERA_HEIGHT = 480  # ลดจาก 720
   ```
5. **ตรวจสอบ CPU usage** - ควรไม่เกิน 80%

---

### ❌ Installation Errors

**Error:** `pip: command not found`
```bash
# ติดตั้ง Python ใหม่ และเลือก "Add Python to PATH"
# Download: https://www.python.org/downloads/
```

**Error:** `TensorFlow installation failed`
```bash
# สำหรับ Windows 10/11:
pip install tensorflow-cpu  # ถ้าไม่มี GPU

# สำหรับ Mac M1/M2:
pip install tensorflow-macos
pip install tensorflow-metal
```

**Error:** `MediaPipe error`
```bash
# ติดตั้ง Visual C++ Redistributable (Windows)
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

pip install mediapipe --upgrade
```

---

### ❓ หากยังแก้ไม่ได้

1. **ตรวจสอบ Requirements:**
   - Python version: `python --version` (ควรเป็น 3.8+)
   - Installed packages: `pip list`

2. **ลองติดตั้งใหม่:**
   ```bash
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

3. **ดู Error Log อย่างละเอียด:**
   - เบราว์เซอร์ Console (F12)
   - Terminal Output
   - Streamlit Error Messages

4. **สร้าง Issue ใน GitHub:**
   - https://github.com/Chonapatcc/Deep_learning_Project/issues
   - แนบ error log และ screenshots

---

## 📝 To-Do List

- [ ] เพิ่ม dataset จริงสำหรับการ train model
- [ ] Train โมเดลด้วยข้อมูลจริง และ integrate เข้า Streamlit
- [ ] เพิ่มรูปภาพตัวอย่างท่ามือทั้ง 26 ตัวอักษร
- [ ] ปรับปรุง UI/UX ใน Streamlit
- [ ] เพิ่มโหมดฝึกสะกดคำ (word spelling)
- [ ] Deploy ขึ้น Streamlit Cloud หรือ Hugging Face Spaces
- [ ] เพิ่มระบบบันทึกและแสดงประวัติการฝึกฝน
- [ ] รองรับภาษามือหลายภาษา (BSL, JSL, ฯลฯ)

## 🤝 การมีส่วนร่วม

ยินดีรับ contributions! กรุณา:
1. Fork โปรเจกต์
2. สร้าง Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add some AmazingFeature'`)
4. Push ไปยัง Branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 ผู้พัฒนา

- **Chonapatcc** - [GitHub](https://github.com/Chonapatcc)

## 🙏 กิตติกรรมประกาศ

- MediaPipe Hands - Google
- TensorFlow.js - Google
- ASL Sign Language Dataset

## 📞 ติดต่อ

หากมีคำถามหรือข้อเสนอแนะ กรุณาติดต่อผ่าน:
- GitHub Issues: [Create Issue](https://github.com/Chonapatcc/Deep_learning_Project/issues)

---

**สร้างด้วย ❤️ เพื่อการเรียนรู้ภาษามือที่เข้าถึงได้สำหรับทุกคน**

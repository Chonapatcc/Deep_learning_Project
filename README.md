# ASL Fingerspelling Coach 🤟

แอปพลิเคชันเว็บสำหรับฝึกทำท่าภาษามือตัวอักษรภาษาอังกฤษ (American Sign Language Fingerspelling) โดยใช้เทคโนโลยี AI และ Computer Vision พร้อมระบบให้ผลตอบรับแบบเรียลไทม์

---

## 🎯 คำอธิบายโครงงาน

### ปัญหาที่แก้ไข
โครงงานนี้พัฒนาขึ้นเพื่อแก้ปัญหา **การขาดผลตอบรับที่ชัดเจนและทันท่วงที (Lack of Immediate Feedback)** ในการเรียนรู้ภาษามือตัวอักษรภาษาอังกฤษด้วยตนเอง ซึ่งส่งผลให้ผู้เรียนไม่สามารถทราบได้ว่าท่าทางที่ฝึกฝนนั้นถูกต้องหรือไม่ นำไปสู่ปัญหา:

- **รูปทรงมือ (Handshape)** ไม่ถูกต้อง: การจัดวางนิ้วผิดเพี้ยนจากท่ามาตรฐาน
- **การเคลื่อนไหว (Movement)** ไม่เหมาะสม: เคลื่อนไหวเร็วหรือช้าเกินไป
- **ตำแหน่งมือ (Location)** ไม่ถูกต้อง: วางมือสูงหรือต่ำเกินไป

### วัตถุประสงค์
พัฒนาแอปพลิเคชันที่ทำหน้าที่เสมือนผู้ช่วยฝึกสอนภาษามือส่วนตัว โดย:

1. ✅ **จดจำท่าภาษามือตัวอักษร A-Z** ด้วยความแม่นยำ ≥ 95%
2. ⚡ **ให้ผลตอบรับแบบเรียลไทม์** ภายใน 0.5 วินาที
3. 🎯 **วิเคราะห์และให้คำแนะนำเฉพาะเจาะจง** สำหรับรูปทรงมือ, การเคลื่อนไหว และตำแหน่งมือ
4. 📈 **สร้างประสบการณ์การเรียนรู้ที่มีประสิทธิภาพ** เพื่อพัฒนาทักษะได้อย่างถูกต้อง

### เทคโนโลยีหลัก
- **MediaPipe Hands**: ตรวจจับมือและ 21 keypoints แบบเรียลไทม์
- **PyTorch/TensorFlow**: โมเดล AI สำหรับจดจำท่าทาง (ResNet-based, ความแม่นยำ 95%+)
- **Streamlit**: Web framework สำหรับ UI/UX
- **Gemini API**: ปรับปรุงข้อความในโหมด Translation (ตัวเลือก)

### คุณสมบัติหลัก
- 📚 **Learning Mode**: เรียนรู้ท่ามือ A-Z พร้อมตัวอย่างและคำแนะนำ
- ✋ **Practice Mode**: ฝึกฝนพร้อม feedback เรียลไทม์
- 🎯 **Test Mode**: ทดสอบความสามารถทั้ง 26 ตัวอักษร
- 🌐 **Translation Mode**: แปลท่ามือเป็นข้อความแบบเรียลไทม์

---

## 📋 ข้อกำหนดระบบ

### ฮาร์ดแวร์
- 💻 คอมพิวเตอร์พร้อมกล้องเว็บแคม
- 🖥️ CPU สมัยใหม่ (ไม่จำเป็นต้องมี GPU)

### ซอฟต์แวร์
- 🐍 Python 3.12 ขึ้นไป
- 🌐 เบราว์เซอร์: Chrome, Edge หรือ Firefox

### สภาพแวดล้อม
- � แสงสว่างเพียงพอและสม่ำเสมอ
- 🎨 พื้นหลังไม่ซับซ้อนจนเกินไป

---

## 🚀 วิธีติดตั้งและรันโค้ด (Reproducible Setup)

### ขั้นตอนที่ 1: Clone Repository

```bash
git clone https://github.com/Chonapatcc/Deep_learning_Project.git
cd Deep_learning_Project
```

---

### ขั้นตอนที่ 2: เตรียม Dataset

**วิธีที่ 1: ใช้ Dataset ที่มีอยู่แล้ว (แนะนำ)**

แตกไฟล์ dataset ที่ดาวน์โหลดมาลงใน `data/asl_dataset/`

```bash
# โครงสร้างที่ถูกต้อง:
data/
└── asl_dataset/
    ├── A/          # รูปภาพตัว A
    ├── B/          # รูปภาพตัว B
    ├── ...
    ├── Z/          # รูปภาพตัว Z
    ├── 0/          # รูปภาพเลข 0
    ├── ...
    └── 9/          # รูปภาพเลข 9
```

**วิธีที่ 2: ดาวน์โหลดจาก Kaggle (ถ้ายังไม่มี)**

```bash
# ติดตั้ง kagglehub
pip install kagglehub

# ดาวน์โหลด dataset (ตัวอย่าง - ปรับ path ตามต้องการ)
python -c "import kagglehub; kagglehub.dataset_download('ayuraj/asl-dataset')"

# ย้ายไฟล์ไปที่ data/asl_dataset/
# (ปรับ path ตามที่ kagglehub ดาวน์โหลดไว้)
```

**ตรวจสอบ Dataset:**

```bash
# ตรวจสอบจำนวนโฟลเดอร์และภาพ
python check_dataset.py
```

**ผลลัพธ์ที่ควรได้:**
```
✅ พบโฟลเดอร์ครบทุกตัวอักษร (A-Z)
📸 รวมทั้งหมด: ~2,500+ ภาพ
```

---

### ขั้นตอนที่ 3: เตรียม Model Weights

**วิธีที่ 1: ใช้ Model ที่มีอยู่แล้ว (Quick Start)**

แตกไฟล์ model weights ลงใน `models/`

```bash
# โครงสร้างที่ถูกต้อง:
models/
├── best_asl_model2.pth      # PyTorch model weights
├── label_encoder2.pkl        # Label encoder
└── asl_processed2.pkl        # (ตัวเลือก) Processed data
```

**วิธีที่ 2: Train Model เอง (Advanced)**

**Option A: Train ใน Google Colab**

1. เปิด `src/colab/train_model.ipynb` ใน Google Colab
2. Upload dataset ไปที่ Colab หรือ mount Google Drive
3. แก้ path ในโค้ดให้ตรงกับที่เก็บ dataset
4. รัน cells ตามลำดับ
5. ดาวน์โหลด model weights ที่ได้กลับมาวางใน `models/`

---

### ขั้นตอนที่ 4: ตั้งค่า Gemini API Key (ตัวเลือก - สำหรับ Translation Mode)

**4.1 รับ API Key ฟรี:**
- ไปที่ https://makersuite.google.com/app/apikey
- Sign in ด้วย Google Account
- คลิก "Create API Key"
- คัดลอก API Key

**4.2 สร้างไฟล์ `.env`:**

```bash
# Windows (Command Prompt)
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

**4.3 แก้ไขไฟล์ `.env`:**

```env
# เปิดไฟล์ .env และใส่ API Key
GEMINI_API_KEY=your_actual_api_key_here
```

**ตัวอย่างที่ถูกต้อง:**
```env
GEMINI_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**⚠️ หมายเหตุ:** ถ้าไม่ต้องการใช้ Translation Mode สามารถข้ามขั้นตอนนี้ได้

---

### ขั้นตอนที่ 5: ติดตั้ง Dependencies

```bash
# ติดตั้ง Python packages ทั้งหมด
pip install -r requirements.txt
```

**หมายเหตุ:** การติดตั้งอาจใช้เวลา 5-10 นาที

**ตรวจสอบการติดตั้ง:**

```bash
# ตรวจสอบ Python version
python --version
# ต้องเป็น Python 3.12 หรือสูงกว่า

# ตรวจสอบ Streamlit
streamlit --version

# ตรวจสอบ PyTorch/TensorFlow
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

### ขั้นตอนที่ 6: รัน Application

```bash
streamlit run app.py
```

**เมื่อรันสำเร็จ คุณจะเห็น:**

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

✅ เปิดเบราว์เซอร์ที่ `http://localhost:8501`

---

## 📊 โครงสร้างโปรเจกต์

```
Deep_learning_Project/
├── 📂 data/                      # Dataset
│   ├── asl_dataset/              # รูปภาพ A-Z, 0-9
│   │   ├── A/*.jpg
│   │   ├── B/*.jpg
│   │   └── ...
│   └── README.md
│
├── 📂 models/                    # Model weights
│   ├── asl_processed2.pkl
│   ├── best_asl_model2.pth      # PyTorch model
│   ├── label_encoder2.pkl       # Label encoder
│   └── README.md
│
├── 📂 src/                       # Source code
│   ├── config.py                # การตั้งค่าหลัก
│   ├── classifier.py            # Model architecture (ResNet-based)
│   ├── dataset.py               # Dataset handler
│   │
│   ├── controllers/
│   │   ├── trainer.py          # Training logic
│   │   ├── evaluator.py        # Evaluation logic
│   │   └── predictor.py        # Prediction logic
│   │
│   ├── utils/
│   │   ├── model_loader.py     # โหลดโมเดล
│   │   ├── preprocessing.py    # Preprocessing
│   │   └── pytorch_utils/      # PyTorch utilities
│   │       ├── preprocessor.py # MediaPipe landmark extraction
│   │       └── data_handler.py
│   │
│   └── colab/                   # Google Colab notebooks
│       └── train_model.ipynb   # Train model in Colab
│
├── 📂 demos/                     # Demo videos
│   ├── easy.mp4                 # Easy scenario
│   ├── medium.mp4               # Medium scenario
│   └── hard.mp4                 # Hard scenario
│
│
├── 📄 app.py                     # Main Streamlit app
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env
├── 📄 .env.example               # ตัวอย่างไฟล์ .env
└── 📄 README.md                  # ไฟล์นี้
```

---

## 🎓 วิธีใช้งาน

### 1. Learning Mode (โหมดเรียนรู้)
- เลือก "Learning Mode" ในแถบด้านข้าง
- คลิกตัวอักษรที่ต้องการเรียนรู้
- ศึกษาท่ามือและดูภาพตัวอย่าง

### 2. Practice Mode (โหมดฝึกฝน)
- เลือก "Practice Mode"
- เปิดกล้อง
- วางมือในกรอบ
- ทำท่ามือตามตัวอย่าง
- รับ feedback ทันที

### 3. Test Mode (โหมดทดสอบ)
- เลือก "Test Mode"
- คลิก "เริ่มทำแบบทดสอบ"
- ทำท่ามือตามที่กำหนด
- ดูผลการทดสอบ

### 4. Translation Mode (โหมดแปลภาษา)
- เลือก "Translation Mode"
- ทำท่ามือเป็นคำ/ประโยค
- ระบบจะแปลและปรับปรุงข้อความอัตโนมัติ

---

## 🐛 การแก้ปัญหา

### ❌ Dataset ไม่พบ
```bash
# ตรวจสอบว่ามีโฟลเดอร์ data/asl_dataset/
```

### ❌ Model ไม่พบ
```bash
# ตรวจสอบว่ามีไฟล์ models/best_asl_model2.pth
dir models\      # Windows
ls models/       # Mac/Linux
```

### ❌ กล้องไม่ทำงาน
- อนุญาตการเข้าถึงกล้องในเบราว์เซอร์
- ปิดแอปอื่นที่ใช้กล้อง (Zoom, Teams, etc.)
- ลองเบราว์เซอร์อื่น (แนะนำ Chrome)

### ❌ Streamlit ไม่ทำงาน
```bash
# ติดตั้ง Streamlit อีกครั้ง
pip install streamlit --upgrade

# หรือใช้ python -m
python -m streamlit run app.py
```

---

## 🤝 การมีส่วนร่วม

ยินดีรับ contributions! กรุณา:
1. Fork โปรเจกต์
2. สร้าง Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add AmazingFeature'`)
4. Push ไปยัง Branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👥 ผู้พัฒนา

- **Chonapatcc** - [GitHub](https://github.com/Chonapatcc)

---

## 🙏 กิตติกรรมประกาศ

- MediaPipe Hands - Google
- PyTorch/TensorFlow - ML Frameworks
- Streamlit - Web Framework
- ASL Dataset - Kaggle Community

---

**สร้างด้วย ❤️ เพื่อการเรียนรู้ภาษามือที่เข้าถึงได้สำหรับทุกคน**

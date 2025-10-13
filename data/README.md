# Data Directory

โฟลเดอร์นี้สำหรับเก็บ ASL Dataset

## 📁 โครงสร้าง (Simplified)

```
data/
├── asl_dataset/           # ⭐ Dataset หลัก (ดาวน์โหลดจาก Kaggle)
│   ├── a/                # ภาพตัวอักษร A
│   ├── b/                # ภาพตัวอักษร B
│   ├── ...               # C-Z
│   └── 0-9/              # ตัวเลข (optional)
│
└── README.md             # ไฟล์นี้
```

**หมายเหตุ**: ไม่ใช้โฟลเดอร์ `raw/` หรือ `processed/` อีกต่อไป เก็บข้อมูลทั้งหมดใน `asl_dataset/` โดยตรง

---

## 🚀 วิธีการใช้งาน

### ตัวเลือก 1: ดาวน์โหลดจาก Kaggle (แนะนำ) ⭐

**ขั้นตอนที่ 1: ติดตั้ง kagglehub**

```bash
pip install kagglehub
```

**ขั้นตอนที่ 2: ตั้งค่า Kaggle API**

1. ไปที่ https://www.kaggle.com/settings
2. ในส่วน 'API' คลิก 'Create New Token'
3. ดาวน์โหลดไฟล์ `kaggle.json`
4. วางไฟล์ไว้ที่:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

**ขั้นตอนที่ 3: ดาวน์โหลด Dataset**

```bash
# ดาวน์โหลดอัตโนมัติ
python download_dataset.py

# หรือตรวจสอบการตั้งค่าก่อน
python download_dataset.py --check
```

---

### ตัวเลือก 2: ใช้ Dataset ที่มีอยู่แล้ว

ถ้าคุณมี dataset อยู่แล้ว:

```bash
# วางโฟลเดอร์ทั้งหมดใน data/asl_dataset/
data/asl_dataset/
├── a/
├── b/
└── ... (ตัวอักษรอื่นๆ)
```

---

## 📊 Dataset Information

**ชื่อ**: ASL Alphabet Dataset  
**แหล่งที่มา**: Kaggle - `ayuraj/asl-dataset`  
**จำนวน**: ~2,500+ ภาพ  
**Classes**: 26 ตัวอักษร (A-Z) + 10 ตัวเลข (0-9, optional)

---

## 🔧 Configuration

Path ที่ตั้งค่าใน `src/config.py`:

```python
class DataConfig:
    DATA_ROOT = "data/"
    DATASET_PATH = "data/asl_dataset"  # เก็บข้อมูลทั้งหมดที่นี่
    KAGGLE_DATASET = "ayuraj/asl-dataset"
```

---

## 🎯 ขั้นตอนหลังดาวน์โหลด

```bash
# 1. ตรวจสอบ dataset
python check_dataset.py

# 2. ประมวลผล (ถ้าต้องการ train model)
python preprocess_dataset.py

# 3. รัน application
streamlit run app.py
```

---

## 📝 Scripts ที่เกี่ยวข้อง

| Script | คำอธิบาย |
|--------|----------|
| `download_dataset.py` | ดาวน์โหลด dataset จาก Kaggle |
| `check_dataset.py` | ตรวจสอบ dataset |
| `preprocess_dataset.py` | ประมวลผล dataset |

---

## 🔍 Troubleshooting

### ❌ "No module named 'kagglehub'"

```bash
pip install kagglehub
```

### ❌ "Could not find kaggle.json"

ตั้งค่า Kaggle API ตามขั้นตอนด้านบน

### ❌ "Dataset not found"

```bash
# ดาวน์โหลดใหม่
python download_dataset.py

# หรือวาง dataset เองใน data/asl_dataset/
```

---

**อัพเดท**: October 14, 2025  
**โครงสร้างใหม่**: ใช้ `asl_dataset/` เท่านั้น (ไม่มี raw/processed)

### 2. ประมวลผล Dataset

**ตอนนี้คุณพร้อมประมวลผลข้อมูลแล้ว!**

```python
# ใช้ preprocessor เพื่อแปลงภาพเป็น landmarks
from src.utils.pytorch_utils.preprocessor import ASLDataPreprocessor
from src.utils.pytorch_utils.data_handler import DataHandler
from sklearn.preprocessing import LabelEncoder

# สร้าง preprocessor
preprocessor = ASLDataPreprocessor()

# ประมวลผล dataset (ใช้ค่า default path)
X, y = preprocessor.process_dataset(
    dataset_path='data/asl_dataset',  # โฟลเดอร์ที่มีอยู่
    augment=True,
    augment_factor=2,
    filter_alphabet_only=True  # เฉพาะ A-Z ไม่รวม 0-9
)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# บันทึกข้อมูลที่ประมวลผลแล้ว
DataHandler.save_processed_data(X, y_encoded, 'data/processed/asl_processed.pkl')
DataHandler.save_label_encoder(le, 'data/processed/label_encoder.pkl')

print(f"✅ ประมวลผล {len(X)} samples เรียบร้อย!")
print(f"✅ {len(set(y))} classes: {sorted(set(y))}")
```

**หมายเหตุ:**
- ใช้ `filter_alphabet_only=True` ถ้าต้องการเฉพาะ A-Z
- ใช้ `filter_alphabet_only=False` ถ้าต้องการรวมตัวเลข 0-9

### 3. โหลดข้อมูลที่ประมวลผลแล้ว

```python
from src.utils.pytorch_utils.data_handler import DataHandler

# โหลดข้อมูล
X, y = DataHandler.load_processed_data('data/processed/processed_data.pkl')
label_encoder = DataHandler.load_label_encoder('data/processed/label_encoder.pkl')
```

## 📊 รูปแบบข้อมูล

### Raw Data (ภาพ)
- **Format**: .jpg, .png, .jpeg
- **ขนาดแนะนำ**: 200x200 หรือ 224x224 pixels
- **จำนวน**: อย่างน้อย 100 ภาพต่อตัวอักษร
- **Background**: ควรมีหลากหลาย

### Processed Data (Landmarks)
- **Format**: NumPy array
- **Shape**: (n_samples, 63)
  - 21 landmarks × 3 coordinates (x, y, z)
- **Normalized**: ค่า 0-1

## 🔧 Configuration

ตั้งค่า paths ใน `src/config.py`:

```python
class DataConfig:
    DATA_ROOT = "data/"
    DATA_RAW = "data/asl_dataset/"      # โฟลเดอร์ที่มีอยู่
    DATA_PROCESSED = "data/processed/"
    DATASET_PATH = "data/asl_dataset"   # ใช้โฟลเดอร์นี้
    PROCESSED_DATA_PATH = "data/processed/asl_processed.pkl"
```

## 🚀 Quick Start

**ทดสอบโหลดข้อมูล:**

```python
from pathlib import Path

# ตรวจสอบว่า dataset พร้อมใช้งาน
dataset_path = Path('data/asl_dataset')
letters = sorted([f.name for f in dataset_path.iterdir() if f.is_dir()])
print(f"พบ {len(letters)} folders: {letters}")

# นับจำนวนภาพในแต่ละ folder
for letter in letters[:3]:  # แสดง 3 ตัวแรก
    count = len(list((dataset_path / letter).glob('*.*')))
    print(f"{letter.upper()}: {count} images")
```

**Output ที่คาดหวัง:**
```
พบ 36 folders: ['0', '1', ..., 'a', 'b', ..., 'z']
A: 70 images
B: 70 images
C: 70 images
```

## 📝 หมายเหตุ

- ⚠️ ไฟล์ `.pkl` อาจมีขนาดใหญ่ (100-500 MB)
- 💡 ควรเพิ่ม `data/raw/` และ `data/processed/` ใน `.gitignore`
- 🔄 สามารถใช้ data augmentation เพื่อเพิ่มจำนวนข้อมูล

## 🚀 ตัวอย่างการใช้งานครบวงจร

```python
from src.utils.pytorch_utils.preprocessor import ASLDataPreprocessor
from src.utils.pytorch_utils.data_handler import DataHandler
from sklearn.preprocessing import LabelEncoder

# 1. ประมวลผล dataset
preprocessor = ASLDataPreprocessor()
X, y = preprocessor.process_dataset('data/raw/asl_dataset')

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. บันทึก
DataHandler.save_processed_data(X, y_encoded, 'data/processed/processed_data.pkl')
DataHandler.save_label_encoder(le, 'data/processed/label_encoder.pkl')

print(f"✅ ประมวลผล {len(X)} samples เรียบร้อย!")
print(f"✅ {len(set(y))} classes: {sorted(set(y))}")
```

---

**สร้างโดย**: ASL Recognition Project  
**อัพเดทล่าสุด**: October 14, 2025

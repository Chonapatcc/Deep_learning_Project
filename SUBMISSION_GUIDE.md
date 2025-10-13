# รายละเอียดข้อกำหนดการส่งงาน (Submission Guide)

เอกสารนี้สรุปสิ่งที่ผู้ตรวจจะต้องการเห็น พร้อม mapping เข้ากับโปรเจกต์นี้ และขั้นตอนรันแบบ reproducible บนเครื่อง Windows (CMD).

---

## 1) Repository Structure (โครงสร้างที่ชัดเจน)

โครงสร้างในโปรเจกต์นี้ตรงตามข้อกำหนดดังนี้:

```
Deep_learning_Project/
├─ data/                 # ชุดข้อมูล หรือสคริปต์ดาวน์โหลดข้อมูล
│  └─ asl_dataset/       # โครงสร้าง A-Z, 0-9 (ถูก .gitignore เฉพาะเนื้อหา)
├─ src/                  # โค้ดหลักสำหรับ preprocessing / training / evaluation / inference
│  ├─ config.py          # การตั้งค่ากลาง (เส้นทางข้อมูล, โมเดล, พรีโปรเซส)
│  ├─ classifier.py      # สถาปัตยกรรมโมเดล PyTorch (landmark-based)
│  ├─ dataset.py         # ASLDataset (PyTorch)
│  └─ controllers/
│     ├─ trainer.py      # Trainer class: วงจร train/validate + early stopping
│     └─ evaluator.py    # Evaluator class: ประเมินผล/รายงาน
├─ models/               # ไฟล์ weight ของโมเดลและ label encoder
│  ├─ best_asl_model2.pth
│  └─ label_encoder2.pkl
├─ demos/                # วิดีโอ demo 3 scenarios (easy.mp4, medium.mp4, hard.mp4)
│  └─ README.md
├─ app.py                # Streamlit demo app (learning/practice/test/translation)
├─ requirements.txt      # Dependencies
└─ README.md             # คำอธิบายโครงงาน วิธีติดตั้ง วิธีรัน
```

หมายเหตุ:
- Dataset ดาวน์โหลด/วางไว้ใน `data/asl_dataset/` (โครงสร้าง A-Z, 0-9)
- Model weights อยู่ใน `models/`

---

## 2) Reproducibility (ทำซ้ำผลลัพธ์ได้)

มี pipeline เครื่องมือครบตั้งแต่ preprocessing → training → evaluation และสามารถตั้งค่า seed เพื่อให้ผลใกล้เคียงกันทุกครั้ง

### 2.1 เตรียมสภาพแวดล้อม

```bat
:: Windows CMD
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 ตรวจสอบ/เตรียม Dataset

- วาง dataset โครงสร้างแบบ: `data/asl_dataset/A/*.jpg|.jpeg|.png`, ..., `data/asl_dataset/Z/`, `0-9/`
- สำหรับตรวจสอบอย่างเร็ว (นับจำนวนโฟลเดอร์/ภาพ):

```bat
python - <<PY
import os, glob
root = r"data/asl_dataset"
classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print("classes:", len(classes))
count = sum(len(glob.glob(os.path.join(root, c, "*.jp*g"))) + len(glob.glob(os.path.join(root, c, "*.png"))) for c in classes)
print("images:", count)
PY
```

หรือใช้ไฟล์ในโค้ดนี้โดยตรงใน app (Learning/Practice) เพื่อยืนยันการอ่านภาพได้

### 2.3 Preprocessing (Landmark extraction)

ใช้คลาส `ASLDataPreprocessor` ใน `src/utils/pytorch_utils/preprocessor.py` เพื่อดึง landmark (21 จุด × 3 = 63 features) พร้อมการ normalize/augment อย่างง่าย

ตัวอย่างสคริปต์รันแบบรวดเร็ว (สร้างใน notebook/script ได้):

```python
from src.utils.pytorch_utils.preprocessor import ASLDataPreprocessor
from src.config import DataConfig

pre = ASLDataPreprocessor(min_detection_confidence=0.3)
X, y = pre.process_dataset(dataset_path=DataConfig.DATASET_PATH, augment=True, augment_factor=2, filter_alphabet_only=True)
pre.close()

print(X.shape, y.shape)  # (N, 63), (N,)
```

บันทึก X, y เป็นไฟล์ .npz หรือ pickle เพื่อ reuse ในการ train ได้ตามสะดวก

### 2.4 Training

ใช้ `Trainer` ใน `src/controllers/trainer.py`

Pseudo-usage (ภายในสคริปต์ train.py ที่คุณสร้าง):

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.classifier import ASLClassifier
from src.dataset import ASLDataset
from src.controllers.trainer import Trainer

# Reproducibility
import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load preprocessed data (X, y)
X = np.load('X.npy')  # ตัวอย่าง
y = np.load('y.npy')

# Dataset & split
full_ds = ASLDataset(X, y)
N = len(full_ds)
train_len = int(0.8*N)
val_len = int(0.1*N)
test_len = N - train_len - val_len
train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)

# Model
model = ASLClassifier(input_size=63, num_classes=26)
trainer = Trainer(model, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=1e-3)

history = trainer.train(train_loader, val_loader, num_epochs=50, save_path='models/best_asl_model2.pth', patience=10)
```

หมายเหตุ: หากต้องการสคริปต์พร้อมรัน สามารถสร้าง `scripts/train.py` และคัดลอกโค้ดด้านบนได้เลย

### 2.5 Evaluation

ใช้ `Evaluator` ใน `src/controllers/evaluator.py`

```python
import torch
from torch.utils.data import DataLoader
from src.classifier import ASLClassifier
from src.controllers.evaluator import Evaluator

# Load model
model = ASLClassifier(input_size=63, num_classes=26)
ckpt = torch.load('models/best_asl_model2.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Evaluate
test_loader = DataLoader(test_ds, batch_size=64)
evaluator = Evaluator(model, device='cpu')
acc, loss, preds, labels = evaluator.evaluate(test_loader)
print('Test Acc:', acc, 'Loss:', loss)

# Optional: report per-class metrics
from sklearn.preprocessing import LabelEncoder
classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
evaluator.print_classification_report(labels, preds, classes)
```

### 2.6 Demo mode (เร็ว/เบาเครื่อง)

- ใช้ `app.py` เปิดโหมด Learning/Practice/Test แทนการ train ครบ เพื่อ demo เร็ว
- ใช้ชุดน้ำหนัก `models/best_asl_model2.pth` ที่มีอยู่แล้ว

รัน:

```bat
streamlit run app.py
```

---

## 3) Slide (หัวข้อที่ควรครอบคลุม)

โครงสำหรับสไลด์ (PDF/PPT) ที่แนะนำ:

### 3.1 Project Title & Team Members
- ชื่อโครงการ: ASL Fingerspelling Trainer
- สมาชิกทีม: ชื่อ/รหัส/หน้าที่ + รูปถ่าย

### 3.2 Problem & Objectives
- ปัญหา: ผู้เรียน/ผู้ใช้ทั่วไปต้องการฝึกภาษามือ A-Z ด้วย feedback แบบ real-time และใช้งานง่าย
- วัตถุประสงค์: ระบบสาธิต/ฝึก/ทดสอบ ASL ที่ทำงานได้บนกล้องเว็บแคม, มีคะแนนความมั่นใจ, มีโหมดเรียน/ฝึก/สอบ
- ผู้ใช้เป้าหมาย: ผู้เริ่มต้น, ครูผู้สอน, งานสาธิตการรับรู้ท่าทาง

### 3.3 Dataset Details
- แหล่งที่มา: Kaggle (เช่น ayuraj/asl-dataset) + อธิบายโครงสร้างโฟลเดอร์ A-Z, 0-9
- แนวทางการเก็บข้อมูลเพิ่ม (ถ้าทำ): ถ่ายรูปมือในสภาพแสงต่างๆ, มุม/ระยะต่างๆ
- ตัวอย่างข้อมูล: ภาพจาก A, B, C… พร้อม landmark visualization (จาก MediaPipe)
- ความเหมาะสม: ชุดข้อมูลจัดโครงสร้างชัดเจน, ครอบคลุมตัวอักษร (และตัวเลข)
- ข้อจำกัด: จำนวนภาพต่อคลาสไม่มาก, คุณภาพภาพหลากหลาย, ไม่มีวิดีโอ temporal
- ผลกระทบ: ความแม่นยำตกในบางท่า/บางสภาพแสง; เสนอวิธีแก้ เช่น augmentation/เก็บเพิ่ม

### 3.4 Methodology & Model Choice
- เลือกใช้ landmark-based classifier (63 ฟีเจอร์จาก MediaPipe) เพราะเบา/เร็ว/เหมาะสำหรับ real-time
- baseline: Logistic/MLP/RandomForest vs. ResNet-based classifier (ตามสถาปัตยกรรมที่ใช้อยู่)
- ตรรกะ: เชื่อมโยงกับวัตถุประสงค์ “ใช้งาน real-time, บนเว็บแคม, คอมพิวเตอร์ทั่วไป”

### 3.5 Training Setup & Hyperparameters
- Split: Train/Val/Test = 80/10/10 (ตาม `src/config.py`)
- Hyperparams สำคัญ: LR=1e-3, batch_size=64, epochs=50, early stopping (patience=10)
- อธิบายเหตุผล: dataset ขนาดปานกลาง → เลือก LR ปลอดภัย, ใช้ ReduceLROnPlateau, EarlyStopping เพื่อลด overfit

### 3.6 Demo Scenarios (อย่างน้อย 3 ระดับ)
- Easy: แสงนิ่ง, มือชัด, ท่ามาตรฐานกลางเฟรม → ควรทายถูกสูง
- Medium: มุมเอียงเล็กน้อย, ระยะห่างต่างกัน, มี noise เล็กน้อย → ยังควรทำงานดี
- Hard: แสงย้อน/มืด, ท่าซับซ้อน, มือเคลื่อนไหวเร็ว → แสดงข้อจำกัด/วิธีปรับปรุง
- ใส่เหตุผลว่าทำไมเลือกแต่ละ scenario → สะท้อนโลกจริง/ความต้องการผู้ใช้

### 3.7 Insights after Demo
- ตัวอย่างที่ดีเกินคาด: ท่าที่ระบบทายถูกแม้มีมุมเอียง
- ตัวอย่างที่ล้มเหลว: ท่าคล้ายกัน (เช่น M/N), แสงแย่, ออกนอกเฟรม
- วิเคราะห์สาเหตุ: คุณภาพข้อมูล/ข้อจำกัดโมเดล/เงื่อนไขเกิน distribution การฝึก

### 3.8 Pros/Cons & Comparison
- Pros: เบา/เร็ว/ง่าย, ทำงาน real-time, พึ่งพาชุด landmark จำนวนน้อย
- Cons: ไวต่อคุณภาพ landmark, ยากเมื่อท่าคล้ายกันมาก, ขาด temporal modeling
- เทียบ baseline/แนวทางอื่น: raw-image CNN/Transformer vs landmark-based (trade-off ความเร็ว/แม่นยำ)

### 3.9 Conclusion & Future Work
- สรุปผลจาก demo: บรรลุวัตถุประสงค์หลักในบริบท real-time demo
- ต่อยอด: เก็บข้อมูลเพิ่ม, ใช้ temporal models (LSTM/Transformer), multi-hand, multi-view, การปรับปรุง UI/UX

---

## 4) คำสั่ง (Windows CMD) แบบรวดเร็ว

```bat
:: 1) สร้างและติดตั้ง environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

:: 2) ตรวจสอบ dataset
:: วางไฟล์ไว้ที่ data\asl_dataset\A..Z
:: เปิดแอปเพื่อทดสอบโหลดภาพและ landmark
streamlit run app.py

:: 3) (ตัวเลือก) รัน preprocessing + train + eval ด้วยสคริปต์ที่คุณสร้าง
python scripts\preprocess.py
python scripts\train.py
python scripts\eval.py
```

หมายเหตุ: โปรเจกต์นี้เตรียมคลาสที่จำเป็นไว้แล้วใน `src/` คุณสามารถสร้างไฟล์ใน `scripts/` เพื่อเรียกใช้งานคลาสเหล่านี้เป็นลำดับขั้น (preprocess → train → eval) ได้ทันที เพื่อความ reproducible 100% ตามเครื่องมือและทรัพยากรของคุณ

---

หากต้องการให้ผม scaffold สคริปต์ใน `scripts/` (preprocess.py, train.py, eval.py) แจ้งได้ จะสร้างให้พร้อมรันทันทีครับ 🙌

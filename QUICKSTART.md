# 🚀 Quick Start Guide

ใช้งานแอป ASL Fingerspelling Trainer ใน 3 ขั้นตอน!

---

## ⚡ 3 ขั้นตอนเริ่มใช้งาน

### 1️⃣ วาง Model File

```bash
# วางไฟล์ model ที่ train แล้วในโฟลเดอร์ models/
# เปลี่ยนชื่อเป็น: ayumi_chan.h5

models/
└── ayumi_chan.h5    # ✅ ต้องมีไฟล์นี้
```

**รูปแบบไฟล์ที่รองรับ:**
- `.h5` - TensorFlow/Keras (แนะนำ)
- `.keras` - Keras format
- `.pt` / `.pth` - PyTorch
- `.onnx` - ONNX Runtime

**วิธีเปลี่ยนชื่อ:**
```bash
# Windows
cd models
rename your_model.h5 ayumi_chan.h5

# Mac/Linux
cd models
mv your_model.h5 ayumi_chan.h5
```

---

### 2️⃣ ตั้งค่า Gemini API Key

**2.1 รับ API Key (ฟรี!):**
1. ไปที่: https://makersuite.google.com/app/apikey
2. Sign in ด้วย Google Account
3. คลิก "Create API Key"
4. คัดลอก API Key

**2.2 สร้างไฟล์ `.env`:**

**Windows (Notepad):**
1. เปิด Notepad
2. พิมพ์: `GEMINI_API_KEY=your_actual_api_key_here`
3. Save As → All Files
4. ตั้งชื่อ: `.env`
5. Save ในโฟลเดอร์โปรเจกต์

**Windows (Command):**
```cmd
echo GEMINI_API_KEY=your_actual_api_key_here > .env
```

**Mac/Linux:**
```bash
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**ตัวอย่างไฟล์ `.env`:**
```env
GEMINI_API_KEY=AIzaSyBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

### 3️⃣ Install & Run!

```bash
# ติดตั้ง dependencies (ครั้งแรกเท่านั้น)
pip install -r requirements.txt

# รันแอป
streamlit run app.py
```

**เมื่อเห็นข้อความนี้:**
```
Local URL: http://localhost:8501
```

✅ **แอปพร้อมใช้งาน!** เปิดเบราว์เซอร์ที่ http://localhost:8501

---

## 📝 Checklist ก่อนเริ่มใช้งาน

- [ ] ติดตั้ง Python 3.8+ แล้ว
- [ ] มีไฟล์ `models/ayumi_chan.h5` (หรือ .keras, .pt, .onnx)
- [ ] มีไฟล์ `.env` พร้อม Gemini API key
- [ ] รัน `pip install -r requirements.txt` สำเร็จ
- [ ] กล้องทำงานปกติ
- [ ] เบราว์เซอร์ Chrome/Edge/Firefox

---

## 🎯 เริ่มใช้งานแอป

### โหมดต่างๆ:

**📚 Learning Mode** - เรียนรู้ท่ามือ A-Z
- เลือกตัวอักษรที่ต้องการเรียนรู้
- ดูภาพตัวอย่างและคำอธิบาย

**✋ Practice Mode** - ฝึกท่ามือ
- เลือกตัวอักษรที่จะฝึก
- เปิดกล้อง
- ทำท่ามือและรับ feedback ทันที

**🎯 Test Mode** - ทดสอบความสามารถ
- ทำแบบทดสอบ A-Z
- ดูคะแนนและผลลัพธ์

**🌐 Translation Mode** - แปลท่ามือเป็นข้อความ
- ทำท่ามือตามคำที่ต้องการ
- ระบบแปลเป็นข้อความอัตโนมัติ
- Gemini AI ช่วยปรับปรุงประโยค

---

## 💡 เคล็ดลับการใช้งาน

### ✅ ทำ:
- ใช้ในที่แสงสว่างเพียงพอ
- วางมือในกรอบสีเขียว
- ทำท่าช้าๆ และชัดเจน
- ให้มือมีขนาดประมาณ 20-30% ของหน้าจอ
- หันมือให้กล้องเห็นท่าทางชัดเจน

### ❌ หลีกเลี่ยง:
- แสงย้อนหลังหรือเงามากเกินไป
- การเคลื่อนไหวเร็วหรือกระตุก
- พื้นหลังซับซ้อนหรือสีคล้ายผิวมือ
- มือเล็กเกินไปหรือออกนอกกรอบ
- ทำหลายท่าพร้อมกัน

---

## 🆘 แก้ปัญหาเร็ว

### Model ไม่โหลด
```bash
# ตรวจสอบชื่อไฟล์
dir models\ayumi_chan.h5  # Windows
ls models/ayumi_chan.h5   # Mac/Linux

# ต้องเห็นไฟล์นี้ในรายการ
```

### Gemini API ไม่ทำงาน
```bash
# ตรวจสอบไฟล์ .env
type .env        # Windows
cat .env         # Mac/Linux

# ควรเห็น: GEMINI_API_KEY=AIzaSy...
```

### กล้องไม่เปิด
1. อนุญาตการเข้าถึงกล้องในเบราว์เซอร์
2. ปิดแอปอื่นที่ใช้กล้อง (Zoom, Teams, etc.)
3. ลองรีเฟรชหน้าเว็บ (F5)

### แอปช้า
1. ปิดแท็บอื่นๆ ในเบราว์เซอร์
2. ใช้ Chrome (performance ดีที่สุด)
3. ปิดโปรแกรมพื้นหลังที่ไม่จำเป็น

---

## 📚 เอกสารเพิ่มเติม

- **README.md** - คู่มือใช้งานแบบเต็ม
- **TRAINING_GUIDE.md** - วิธี train model
- **MULTI_FRAMEWORK_GUIDE.md** - ใช้ PyTorch/ONNX
- **TRANSLATION_MODE_GUIDE.md** - คู่มือโหมด Translation
- **SKELETON_COLOR_GUIDE.md** - ปรับสี skeleton

---

## 🎓 Next Steps

หลังจากใช้งานแอปได้แล้ว:

1. **ฝึกท่ามือ A-Z** ในโหมด Practice
2. **ทดสอบความสามารถ** ในโหมด Test
3. **ลองแปลประโยค** ในโหมด Translation
4. **Train Model ของคุณเอง** ด้วย `train_improved_model.py`
5. **ปรับแต่งสีและ UI** ตามความชอบ

---

**Happy Learning! 🤟**

ถ้ามีปัญหาหรือคำถาม:
- อ่าน [README.md](README.md) สำหรับรายละเอียดเพิ่มเติม
- สร้าง Issue: https://github.com/Chonapatcc/Deep_learning_Project/issues

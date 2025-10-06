# ASL Reference Images

## การเพิ่มรูปภาพท่ามือ

โฟลเดอร์นี้ใช้สำหรับเก็บรูปภาพตัวอย่างท่ามือตัวอักษร A-Z

### รูปแบบไฟล์ที่รองรับ
- PNG (แนะนำ)
- JPG/JPEG
- SVG

### การตั้งชื่อไฟล์
```
A.png, B.png, C.png, ..., Z.png
```

### ข้อกำหนดรูปภาพ
- ขนาดแนะนำ: 300x300 พิกเซล
- พื้นหลังสีขาวหรือโปร่งใส
- แสดงท่ามือชัดเจน
- มุมมองด้านหน้า

### แหล่งรูปภาพ ASL ที่แนะนำ

1. **ASL Sign Language Resources**
   - https://www.lifeprint.com/asl101/fingerspelling/
   - https://www.handspeak.com/

2. **Open Source Datasets**
   - ASL Alphabet Dataset (Kaggle)
   - SignWriting

3. **สร้างเอง**
   - ถ่ายภาพท่ามือเอง
   - ใช้ graphics software สร้าง illustration

### ตัวอย่างการใช้งาน

ในไฟล์ `js/config.js`:
```javascript
const ASL_IMAGES = {
    'A': './assets/asl/A.png',
    'B': './assets/asl/B.png',
    // ...
};
```

---

**หมายเหตุ**: ปัจจุบันมี placeholder SVG สำหรับตัว A เพื่อการทดสอบ
กรุณาแทนที่ด้วยรูปภาพท่ามือจริง

"""
ASL Fingerspelling Trainer - Streamlit Web App
Quick Demo/Prototype for practicing ASL fingerspelling with real-time feedback
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.model_loader import init_mediapipe, load_models
from utils.prediction import predict_letter
from utils.hand_processing import extract_keypoints, is_in_roi, calculate_bbox
from utils.letter_data import get_letter_instructions

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ google-generativeai not installed. Translation mode will be limited.")

# Page configuration
st.set_page_config(
    page_title="ASL Fingerspelling Trainer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .feedback-correct {
        background-color: #50C878;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.2rem;
        text-align: center;
    }
    .feedback-incorrect {
        background-color: #E74C3C;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.2rem;
        text-align: center;
    }
    .feedback-warning {
        background-color: #F39C12;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
mp_hands, mp_drawing, hands = init_mediapipe()

# Initialize session state
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'attempts': 0,
        'correct': 0,
        'total_accuracy': 0
    }

if 'current_letter' not in st.session_state:
    st.session_state.current_letter = 'A'

if 'keypoint_buffer' not in st.session_state:
    st.session_state.keypoint_buffer = []

if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

if 'refined_text' not in st.session_state:
    st.session_state.refined_text = ""

if 'translation_buffer' not in st.session_state:
    st.session_state.translation_buffer = []

# Alphabet
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Load models on startup
MODELS_DATA = load_models()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Fingerspelling Trainer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ฝึกฝนภาษามือตัวอักษรภาษาอังกฤษ A-Z</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "เลือกโหมด",
            ["📚 Learning Mode", "✋ Practice Mode", "🎯 Test Mode", "🌐 Real-time Translation"],
            index=3  # Default to Real-time Translation
        )
        
        st.markdown("---")
        
        if mode == "✋ Practice Mode":
            st.session_state.current_letter = st.selectbox(
                "เลือกตัวอักษร",
                ALPHABET,
                index=ALPHABET.index(st.session_state.current_letter)
            )
            
            if st.button("🔄 Reset Stats"):
                st.session_state.stats = {
                    'attempts': 0,
                    'correct': 0,
                    'total_accuracy': 0
                }
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - ใช้งานในที่มีแสงสว่างดี
        - วางมือให้อยู่ในกรอบ
        - ทำท่าช้าๆ และชัดเจน
        - ให้มือมีขนาดใหญ่พอ
        """)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("ASL Fingerspelling Trainer v1.0\n\nใช้ AI และ Computer Vision ในการตรวจจับและวิเคราะห์ท่ามือแบบ Real-time")
    
    # Main content based on mode
    if mode == "📚 Learning Mode":
        show_learning_mode()
    elif mode == "✋ Practice Mode":
        show_practice_mode()
    elif mode == "🌐 Real-time Translation":
        show_translation_mode()
    else:
        show_test_mode()

def show_learning_mode():
    """Learning Mode - View ASL alphabet reference"""
    st.header("📚 โหมดเรียนรู้ - ดูท่ามือตัวอักษร")
    
    st.info("ℹ️ เลือกตัวอักษรเพื่อดูตัวอย่างท่ามือและคำแนะนำ")
    
    # Create grid for alphabet
    cols_per_row = 6
    rows = [ALPHABET[i:i+cols_per_row] for i in range(0, len(ALPHABET), cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, letter in enumerate(row):
            with cols[idx]:
                if st.button(letter, key=f"learn_{letter}", use_container_width=True):
                    show_letter_detail(letter)

def show_letter_detail(letter):
    """Show detailed information about a letter"""
    st.markdown(f"### ตัวอักษร {letter}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Try to load ASL alphabet reference image
        asl_image_path = f"assets/asl/{letter}.svg"
        if os.path.exists(asl_image_path):
            try:
                st.image(asl_image_path, use_container_width=True)
            except:
                # Fallback to large letter display
                st.markdown(f"<div style='font-size: 8rem; text-align: center; color: #4A90E2;'>{letter}</div>", 
                           unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size: 8rem; text-align: center; color: #4A90E2;'>{letter}</div>", 
                       unsafe_allow_html=True)
        
        # Load images from dataset
        dataset_path = f"datasets/asl_dataset/{letter.lower()}"
        if os.path.exists(dataset_path):
            images = glob.glob(f"{dataset_path}/*.jpeg")[:6]  # Get first 6 images
            
            if images:
                st.markdown("#### ตัวอย่างท่ามือ:")
                # Display images in 2 rows of 3
                for i in range(0, len(images), 3):
                    img_cols = st.columns(3)
                    for j, img_path in enumerate(images[i:i+3]):
                        with img_cols[j]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except:
                                st.error("❌")
            else:
                st.warning(f"⚠️ ไม่พบรูปภาพสำหรับตัวอักษร {letter}")
        else:
            st.warning(f"⚠️ ไม่พบโฟลเดอร์: {dataset_path}")
    
    with col2:
        st.markdown("#### วิธีทำท่า:")
        instructions = get_letter_instructions(letter)
        st.write(instructions)
        
        st.markdown("---")
        st.markdown("#### 💡 คำแนะนำ:")
        st.markdown("""
        - ดูตัวอย่างภาพทั้งหมดให้ครบ
        - สังเกตตำแหน่งนิ้วและมือ
        - ลองทำท่าตามภาพ
        - ฝึกจนชำนาญก่อนทดสอบ
        """)
        
        if st.button("🎯 เริ่มฝึกฝนตัวอักษรนี้", type="primary"):
            st.session_state.current_letter = letter
            st.rerun()

def show_practice_mode():
    """Practice Mode - Real-time practice with feedback"""
    st.header(f"✋ โหมดฝึกฝน - ตัวอักษร {st.session_state.current_letter}")
    
    # Create two columns: Reference images and Practice area
    col_ref, col_practice = st.columns([1, 2])
    
    with col_ref:
        st.markdown("### 📸 ตัวอย่างท่ามือ")
        
        # Load reference images from dataset
        letter = st.session_state.current_letter
        dataset_path = f"datasets/asl_dataset/{letter.lower()}"
        
        if os.path.exists(dataset_path):
            images = glob.glob(f"{dataset_path}/*.jpeg")[:4]  # Show 4 reference images
            
            if images:
                for i in range(0, len(images), 2):
                    img_cols = st.columns(2)
                    for j, img_path in enumerate(images[i:i+2]):
                        with img_cols[j]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except:
                                pass
            else:
                st.info("ไม่พบรูปตัวอย่าง")
        else:
            st.info("ไม่พบโฟลเดอร์ข้อมูล")
        
        # Instructions
        st.markdown("---")
        st.markdown("### 💡 วิธีทำท่า")
        instructions = get_letter_instructions(letter)
        st.info(instructions)
    
    with col_practice:
        # Display stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.stats['attempts']}</div>
                <div class="stat-label">ครั้งที่ลอง</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.stats['correct']}</div>
            <div class="stat-label">ถูกต้อง</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        success_rate = (st.session_state.stats['correct'] / st.session_state.stats['attempts'] * 100) if st.session_state.stats['attempts'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{success_rate:.0f}%</div>
            <div class="stat-label">อัตราความสำเร็จ</div>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("---")
        
        # Camera section
        st.subheader("📷 Camera Feed")
        run_camera = st.checkbox("เปิดกล้อง", value=False)
        
        if run_camera:
            run_webcam_detection()
        
        if st.button("⏭️ ตัวอักษรถัดไป"):
            current_idx = ALPHABET.index(st.session_state.current_letter)
            next_idx = (current_idx + 1) % len(ALPHABET)
            st.session_state.current_letter = ALPHABET[next_idx]
            st.rerun()

def run_webcam_detection():
    """Run webcam with hand detection"""
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการอนุญาตใช้งานกล้อง")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง")
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ ไม่สามารถอ่านข้อมูลจากกล้องได้")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store raw frame for CNN models (before drawing)
        frame_for_cnn = frame.copy()
        st.session_state.frame_buffer.append(frame_for_cnn)
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        
        if results.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.keypoint_buffer.append(keypoints)
                
                # Keep only recent frames
                if len(st.session_state.keypoint_buffer) > 60:
                    st.session_state.keypoint_buffer.pop(0)
                
                # Check bounding box
                bbox = calculate_bbox(hand_landmarks)
                in_roi = is_in_roi(bbox)
                hand_size = max(bbox['width'], bbox['height'])
                
                if not in_roi:
                    feedback_message = "⚠️ มืออยู่นอกพื้นที่ กรุณาวางมือให้อยู่กลางกรอบ"
                    feedback_class = "feedback-warning"
                elif hand_size < 0.2:
                    feedback_message = "🔍 มือเล็กไป กรุณาเข้าใกล้กล้อง"
                    feedback_class = "feedback-warning"
                elif len(st.session_state.keypoint_buffer) >= 15:
                    # Predict
                    predicted_letter, confidence = predict_letter(st.session_state.keypoint_buffer, MODELS_DATA, ALPHABET)
                    
                    if predicted_letter and confidence >= 0.7:
                        is_correct = predicted_letter == st.session_state.current_letter
                        
                        if is_correct:
                            feedback_message = f"✅ ถูกต้อง! {predicted_letter} ({confidence*100:.0f}%)"
                            feedback_class = "feedback-correct"
                            
                            # Update stats
                            st.session_state.stats['attempts'] += 1
                            st.session_state.stats['correct'] += 1
                            st.session_state.stats['total_accuracy'] += confidence * 100
                            
                            # Clear buffer
                            st.session_state.keypoint_buffer = []
                            time.sleep(1)
                        else:
                            feedback_message = f"🔄 ตรวจจับได้: {predicted_letter} ({confidence*100:.0f}%) - เป้าหมาย: {st.session_state.current_letter}"
                            feedback_class = "feedback-incorrect"
                    else:
                        feedback_message = "⏳ กำลังตรวจจับ..."
                        feedback_class = "feedback-warning"
                else:
                    feedback_message = "⏳ กำลังรวบรวมข้อมูล..."
                    feedback_class = "feedback-warning"
                
                # Draw ROI box
                h, w = frame.shape[:2]
                cv2.rectangle(frame, 
                            (int(w*0.2), int(h*0.1)), 
                            (int(w*0.8), int(h*0.8)), 
                            (0, 255, 0), 2)
        else:
            feedback_message = "✋ ไม่พบมือ กรุณาวางมือในกรอบ"
            feedback_class = "feedback-warning"
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Display feedback
        if feedback_message:
            feedback_placeholder.markdown(
                f'<div class="{feedback_class}">{feedback_message}</div>',
                unsafe_allow_html=True
            )
    
    cap.release()

def show_translation_mode():
    """Real-time Translation Mode - Translate ASL to text and refine with Gemini API"""
    st.header("🌐 โหมดแปลภาษาแบบ Real-time")
    
    # Check Gemini API availability
    if not GEMINI_AVAILABLE:
        st.error("❌ ไม่สามารถใช้งานได้ กรุณาติดตั้ง: pip install google-generativeai")
        return
    
    # API Key configuration - Load from .env or user input
    api_key = os.getenv("GEMINI_API_KEY")  # Try to load from .env first
    
    if not api_key:
        # Show instructions if no API key in .env
        st.info("""
        ### 🔑 ต้องการ Gemini API Key
        
        **วิธีที่ 1: ใช้ไฟล์ .env (แนะนำ - ปลอดภัยกว่า)**
        1. สร้างไฟล์ `.env` ในโฟลเดอร์โปรเจกต์
        2. เพิ่มบรรทัด: `GEMINI_API_KEY=your_api_key_here`
        3. Restart แอปพลิเคชัน
        
        **วิธีที่ 2: ใส่โดยตรง (ชั่วคราว)**
        1. ไปที่ [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. สร้าง API Key
        3. ใส่ในช่องด้านล่าง
        """)
        
        api_key = st.text_input(
            "Gemini API Key", 
            type="password", 
            help="ใส่ API Key จาก Google AI Studio (หรือใช้ .env file)",
            placeholder="กรุณาใส่ API Key หรือตั้งค่าใน .env file"
        )
    else:
        # API key loaded from .env
        st.success("✅ โหลด API Key จากไฟล์ .env สำเร็จ!")
    
    if not api_key:
        st.warning("⚠️ กรุณาใส่ Gemini API Key เพื่อใช้งานโหมดนี้")
        return
    
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ Gemini API: {str(e)}")
        return
    
    st.success("✅ เชื่อมต่อ Gemini API สำเร็จ!")
    
    # Translation settings
    col1, col2 = st.columns(2)
    with col1:
        auto_refine = st.checkbox("🔄 Auto-refine ทุกๆ 5 ตัวอักษร", value=True)
    with col2:
        clear_buffer = st.button("🗑️ ล้างข้อความ")
    
    if clear_buffer:
        st.session_state.translated_text = ""
        st.session_state.refined_text = ""
        st.session_state.translation_buffer = []
        st.rerun()
    
    # Display areas
    st.markdown("### 📝 ข้อความที่แปลได้")
    translated_display = st.empty()
    
    st.markdown("### ✨ ข้อความที่ปรับปรุงแล้ว (Gemini)")
    refined_display = st.empty()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ตัวอักษร", len(st.session_state.translated_text))
    with col2:
        st.metric("คำ", len(st.session_state.translated_text.split()))
    with col3:
        refine_button = st.button("✨ Refine ด้วย Gemini", type="primary")
    
    # Display current text
    if st.session_state.translated_text:
        translated_display.markdown(f"""
        <div style='background-color: #E8F4F8; padding: 20px; border-radius: 10px; font-size: 1.2rem;'>
            {st.session_state.translated_text}
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.refined_text:
        refined_display.markdown(f"""
        <div style='background-color: #E8F8E8; padding: 20px; border-radius: 10px; font-size: 1.2rem;'>
            {st.session_state.refined_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Manual refine button
    if refine_button and st.session_state.translated_text:
        with st.spinner("🔄 กำลังปรับปรุงข้อความด้วย Gemini..."):
            try:
                prompt = f"""You are a helpful assistant that refines text. 
                The following text was detected from ASL fingerspelling and may contain errors or be incomplete.
                Please refine it to make it grammatically correct and meaningful.
                Keep the same language (if Thai, output Thai; if English, output English).
                Only return the refined text, nothing else.
                
                Original text: {st.session_state.translated_text}
                
                Refined text:"""
                
                response = model.generate_content(prompt)
                st.session_state.refined_text = response.text.strip()
                st.rerun()
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📹 กล้อง")
    
    # Initialize MediaPipe
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง")
    last_detected_letter = None
    detection_count = 0
    CONFIRMATION_THRESHOLD = 5  # ต้องตรวจจับตัวอักษรเดียวกัน 5 ครั้งติดต่อกัน
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Store raw frame for CNN models (before drawing)
        frame_for_cnn = frame.copy()
        st.session_state.frame_buffer.append(frame_for_cnn)
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.translation_buffer.append(keypoints)
                
                # Keep buffer size manageable
                if len(st.session_state.translation_buffer) > 30:
                    st.session_state.translation_buffer.pop(0)
                
                # Check hand position
                bbox = calculate_bbox(hand_landmarks)
                in_roi = is_in_roi(bbox)
                
                if in_roi and bbox['width'] >= 0.15 and bbox['height'] >= 0.15:
                    if len(st.session_state.translation_buffer) >= 15:
                        # Predict letter
                        predicted_letter, confidence = predict_letter(st.session_state.translation_buffer, MODELS_DATA, ALPHABET)
                        
                        if predicted_letter and confidence >= 0.75:
                            # Confirmation logic
                            if predicted_letter == last_detected_letter:
                                detection_count += 1
                            else:
                                last_detected_letter = predicted_letter
                                detection_count = 1
                            
                            # Add letter when confirmed
                            if detection_count >= CONFIRMATION_THRESHOLD:
                                st.session_state.translated_text += predicted_letter
                                feedback_message = f"✅ เพิ่ม: {predicted_letter}"
                                feedback_class = "feedback-correct"
                                
                                # Auto-refine every 5 characters
                                if auto_refine and len(st.session_state.translated_text) % 5 == 0:
                                    try:
                                        prompt = f"""Refine this text to make it meaningful. Keep it brief.
                                        Original: {st.session_state.translated_text}
                                        Refined:"""
                                        response = model.generate_content(prompt)
                                        st.session_state.refined_text = response.text.strip()
                                    except:
                                        pass
                                
                                # Reset
                                detection_count = 0
                                last_detected_letter = None
                                st.session_state.translation_buffer = []
                                time.sleep(0.5)
                            else:
                                feedback_message = f"🔄 ตรวจจับ: {predicted_letter} ({detection_count}/{CONFIRMATION_THRESHOLD})"
                                feedback_class = "feedback-warning"
                        else:
                            feedback_message = "⏳ กำลังวิเคราะห์..."
                            feedback_class = "feedback-warning"
                    else:
                        feedback_message = "⏳ กำลังรวบรวมข้อมูล..."
                        feedback_class = "feedback-warning"
                else:
                    feedback_message = "⚠️ วางมือให้อยู่ในกรอบและให้มีขนาดที่เหมาะสม"
                    feedback_class = "feedback-warning"
        else:
            feedback_message = "👋 กรุณาแสดงมือต่อกล้อง"
            feedback_class = "feedback-warning"
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB")
        
        # Display feedback
        if feedback_message:
            feedback_placeholder.markdown(
                f'<div class="{feedback_class}">{feedback_message}</div>',
                unsafe_allow_html=True
            )
    
    cap.release()

def show_test_mode():
    """Test Mode - Complete assessment"""
    st.header("🎯 โหมดทดสอบ")
    
    st.info("""
    ### 📋 คำแนะนำ
    - ทดสอบความสามารถทำท่ามือทั้ง 26 ตัวอักษร
    - จับเวลา 15 นาที
    - คะแนนผ่าน: 80% (21/26 ตัว)
    - ทำตามตัวอักษรที่แสดงบนหน้าจอ
    """)
    
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
    
    if not st.session_state.test_started:
        if st.button("🚀 เริ่มทำแบบทดสอบ", type="primary", use_container_width=True):
            st.session_state.test_started = True
            st.session_state.test_answers = []
            st.session_state.test_start_time = time.time()
            st.rerun()
    else:
        # Show test interface
        elapsed_time = int(time.time() - st.session_state.test_start_time)
        remaining_time = max(0, 900 - elapsed_time)  # 15 minutes
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ข้อที่", f"{len(st.session_state.test_answers) + 1}/26")
        with col2:
            st.metric("เวลาที่ใช้", f"{elapsed_time//60:02d}:{elapsed_time%60:02d}")
        with col3:
            st.metric("เวลาคงเหลือ", f"{remaining_time//60:02d}:{remaining_time%60:02d}")
        
        if remaining_time == 0:
            show_test_results()
        else:
            st.markdown("### ทำท่ามือตัวอักษร:")
            current_letter = ALPHABET[len(st.session_state.test_answers)]
            st.markdown(f"<div style='font-size: 8rem; text-align: center; color: #4A90E2;'>{current_letter}</div>", 
                       unsafe_allow_html=True)
            
            if st.button("✅ ยืนยันคำตอบ", type="primary"):
                # Mock answer
                st.session_state.test_answers.append({
                    'question': current_letter,
                    'answer': current_letter,
                    'correct': True
                })
                
                if len(st.session_state.test_answers) >= 26:
                    show_test_results()
                else:
                    st.rerun()

def show_test_results():
    """Show test results"""
    st.success("✅ ทำแบบทดสอบเสร็จสมบูรณ์!")
    
    correct_count = sum(1 for ans in st.session_state.test_answers if ans['correct'])
    total = len(st.session_state.test_answers)
    percentage = (correct_count / total * 100) if total > 0 else 0
    passed = percentage >= 80
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <div style='font-size: 5rem; font-weight: bold;'>{correct_count}</div>
            <div style='font-size: 2rem;'>/ {total}</div>
            <div style='font-size: 1.2rem; margin-top: 10px;'>คะแนนรวม</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        ### 📊 สรุปผล
        - คะแนน: {correct_count}/{total} ({percentage:.0f}%)
        - ผลการทดสอบ: {'✅ ผ่าน' if passed else '❌ ไม่ผ่าน'}
        - เกณฑ์ผ่าน: 80% (21/26)
        """)
    
    if st.button("🔄 ทำแบบทดสอบใหม่"):
        st.session_state.test_started = False
        st.session_state.test_answers = []
        st.rerun()

if __name__ == "__main__":
    main()

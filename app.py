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

# Alphabet and Numbers
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

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
    st.header("📚 โหมดเรียนรู้ - ดูท่ามือตัวอักษรและตัวเลข")
    
    # Check if a character is selected
    if 'selected_learning_char' not in st.session_state:
        st.session_state.selected_learning_char = None
    
    # If character is selected, show fullscreen detail
    if st.session_state.selected_learning_char:
        show_letter_detail(st.session_state.selected_learning_char)
    else:
        # Show selection grid
        st.info("ℹ️ เลือกตัวอักษรหรือตัวเลขเพื่อดูตัวอย่างท่ามือและคำแนะนำให้ดีกว่านี้")
        
        # Tab for Letters and Numbers
        tab1, tab2 = st.tabs(["🔤 ตัวอักษร A-Z", "🔢 ตัวเลข 0-9"])
        
        with tab1:
            # Create grid for alphabet
            st.markdown("### ตัวอย่างให้ ดีกว่านี้ - เลือกตัวอักษรที่ต้องการเรียนรู้")
            cols_per_row = 7
            letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            rows = [letters[i:i+cols_per_row] for i in range(0, len(letters), cols_per_row)]
            
            for row in rows:
                cols = st.columns(cols_per_row)
                for idx, letter in enumerate(row):
                    with cols[idx]:
                        if st.button(letter, key=f"learn_{letter}", use_container_width=True):
                            st.session_state.selected_learning_char = letter
                            st.rerun()
        
        with tab2:
            # Create grid for numbers
            st.markdown("### ตัวอย่างให้ ดีกว่านี้ - เลือกตัวเลขที่ต้องการเรียนรู้")
            cols = st.columns(10)
            numbers = list('0123456789')
            for idx, number in enumerate(numbers):
                with cols[idx]:
                    if st.button(number, key=f"learn_{number}", use_container_width=True):
                        st.session_state.selected_learning_char = number
                        st.rerun()

def show_letter_detail(letter):
    """Show detailed information about a letter or number - FULLSCREEN"""
    char_type = "ตัวเลข" if letter.isdigit() else "ตัวอักษร"
    
    # Back button at top
    if st.button("← กลับไปเลือกตัวอักษร/ตัวเลข", key="back_to_selection"):
        st.session_state.selected_learning_char = None
        st.rerun()
    
    st.markdown(f"# {char_type} {letter} - ตัวอย่างให้ ดีกว่านี้")
    st.markdown("---")
    
    # Fullscreen layout - no columns, stack vertically
    # Large character display
    st.markdown(f"<div style='font-size: 12rem; text-align: center; color: #4A90E2; font-weight: bold; margin: 30px 0;'>{letter}</div>", 
               unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("## 💡 วิธีทำท่า:")
    instructions = get_letter_instructions(letter)
    st.info(instructions)
    
    st.markdown("---")
    
    # Load images from dataset - display fullwidth
    dataset_path = f"datasets/asl_dataset/{letter.lower()}"
    if os.path.exists(dataset_path):
        images = glob.glob(f"{dataset_path}/*.jpeg")[:9]  # Get first 9 images
        
        if images:
            st.markdown("## 📸 ตัวอย่างท่ามือให้ ดีกว่านี้:")
            # Display images in 3 rows of 3 - larger size
            for i in range(0, len(images), 3):
                img_cols = st.columns(3)
                for j, img_path in enumerate(images[i:i+3]):
                    with img_cols[j]:
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True, caption=f"ตัวอย่าง {i+j+1}")
                        except:
                            st.error("❌")
        else:
            st.warning(f"⚠️ ไม่พบรูปภาพสำหรับ{char_type} {letter}")
    else:
        st.warning(f"⚠️ ไม่พบโฟลเดอร์: {dataset_path}")
    
    st.markdown("---")
    
    # Tips section
    st.markdown("## 🎯 คำแนะนำ:")
    st.markdown("""
    - 👀 **ดูตัวอย่างภาพทั้งหมดให้ครบ** - สังเกตละเอียดทุกมุมมอง
    - ✋ **สังเกตตำแหน่งนิ้วและมือ** - ความถูกต้องของแต่ละนิ้ว
    - 🔁 **ลองทำท่าตามภาพ** - ฝึกหัดต่อกล้องจนชำนาญ
    - 🎯 **ฝึกจนชำนาญก่อนทดสอบ** - ความเชี่ยวชาญเป็นกุญแจ
    """)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 เริ่มฝึกฝนตัวอักษรนี้", type="primary", use_container_width=True):
            st.session_state.current_letter = letter
            st.session_state.selected_learning_char = None
            st.rerun()
    with col2:
        if st.button("← กลับไปเลือก", use_container_width=True):
            st.session_state.selected_learning_char = None
            st.rerun()

def show_practice_mode():
    """Practice Mode - Real-time practice with feedback"""
    char_type = "ตัวเลข" if st.session_state.current_letter.isdigit() else "ตัวอักษร"
    st.header(f"✋ โหมดฝึกฝน - {char_type} {st.session_state.current_letter}")
    
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
            attempts = st.session_state.stats.get('attempts', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{attempts}</div>
                <div class="stat-label">ครั้งที่ลอง</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        correct = st.session_state.stats.get('correct', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{correct}</div>
            <div class="stat-label">ถูกต้อง</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        attempts = st.session_state.stats.get('attempts', 0)
        correct = st.session_state.stats.get('correct', 0)
        success_rate = (correct / attempts * 100) if attempts > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{success_rate:.1f}%</div>
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

def run_test_detection(target_letter):
    """Run camera detection for test mode with auto-skip"""
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการอนุญาตใช้งานกล้อง")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง", key="test_stop_camera")
    
    # Detection tracking
    detection_frames = 0
    required_frames = 30  # Need 30 consecutive frames for confirmation
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store frame for CNN models
        frame_for_cnn = frame.copy()
        st.session_state.frame_buffer.append(frame_for_cnn)
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.keypoint_buffer.append(keypoints)
                
                if len(st.session_state.keypoint_buffer) > 60:
                    st.session_state.keypoint_buffer.pop(0)
                
                # Check if enough data
                if len(st.session_state.keypoint_buffer) >= 15:
                    predicted_letter, confidence = predict_letter(
                        st.session_state.keypoint_buffer, 
                        MODELS_DATA, 
                        ALPHABET
                    )
                    
                    if predicted_letter and confidence >= 0.75:
                        if predicted_letter == target_letter:
                            detection_frames += 1
                            progress = int((detection_frames / required_frames) * 100)
                            feedback_message = f"✅ ตรวจจับได้: {predicted_letter} ({confidence*100:.0f}%) - กำลังยืนยัน... {progress}%"
                            feedback_class = "feedback-correct"
                            
                            # If confirmed, auto-skip to next question
                            if detection_frames >= required_frames:
                                st.session_state.test_detected_letter = predicted_letter
                                st.session_state.test_answers.append({
                                    'question': target_letter,
                                    'answer': predicted_letter,
                                    'correct': True
                                })
                                feedback_placeholder.success(f"✅ ถูกต้อง! {predicted_letter} - ไปข้อถัดไปอัตโนมัติ...")
                                cap.release()
                                time.sleep(1)
                                st.rerun()  # Auto-skip to next question
                        else:
                            detection_frames = 0
                            feedback_message = f"❌ ตรวจจับได้: {predicted_letter} ({confidence*100:.0f}%) - ต้องการ: {target_letter}"
                            feedback_class = "feedback-incorrect"
                    else:
                        detection_frames = 0
                        feedback_message = "⏳ กำลังตรวจจับ..."
                        feedback_class = "feedback-warning"
                
                # Draw ROI
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (int(w*0.2), int(h*0.1)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
        else:
            detection_frames = 0
            feedback_message = "✋ ไม่พบมือ - กรุณาวางมือในกรอบ"
            feedback_class = "feedback-warning"
        
        # Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Show feedback
        if feedback_message:
            if feedback_class == "feedback-correct":
                feedback_placeholder.success(feedback_message)
            elif feedback_class == "feedback-incorrect":
                feedback_placeholder.error(feedback_message)
            else:
                feedback_placeholder.info(feedback_message)
    
    cap.release()

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
    
    # Show word formation
    st.markdown("### 💬 คำที่สร้าง")
    if st.session_state.translated_text:
        words = st.session_state.translated_text.split()
        if words:
            st.markdown(" · ".join(words))
        else:
            st.info("เริ่มทำท่ามือเพื่อสร้างคำ...")
    else:
        st.info("เริ่มทำท่ามือเพื่อสร้างคำ...")
    
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
            # Show current question
            current_letter = ALPHABET[len(st.session_state.test_answers)]
            char_type = "ตัวเลข" if current_letter.isdigit() else "ตัวอักษร"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### ทำท่ามือ{char_type}:")
                st.markdown(f"<div style='font-size: 10rem; text-align: center; color: #4A90E2; font-weight: bold; padding: 50px 0;'>{current_letter}</div>", 
                           unsafe_allow_html=True)
                
                # Show instructions only
                st.markdown("#### 💡 วิธีทำท่า:")
                instructions = get_letter_instructions(current_letter)
                st.info(instructions)
            
            with col2:
                st.markdown("### 📷 เปิดกล้องเพื่อจับท่ามือ")
                run_test_camera = st.checkbox("เปิดกล้อง", value=False, key="test_camera")
                
                if run_test_camera:
                    st.info(f"📌 ทำท่ามือ{char_type} {current_letter} และกดยืนยันเมื่อพร้อม")
                    run_test_detection(current_letter)
                else:
                    st.warning("⚠️ เปิดกล้องเพื่อเริ่มทำแบบทดสอบ")
                
                st.info("📹 เปิดกล้องและทำท่ามือ - ระบบจะข้ามข้อถัดไปอัตโนมัติเมื่อตรวจจับถูกต้อง")
                
                # Manual skip (if needed)
                if st.button("⏭️ ข้ามข้อนี้ (ไม่ทำหรือไม่เปิดกล้อง)", use_container_width=True):
                    # Mark as incorrect if skipped
                    st.session_state.test_answers.append({
                        'question': current_letter,
                        'answer': None,
                        'correct': False
                    })
                    
                    total_questions = len([c for c in ALPHABET if c.isalpha()]) + len([c for c in ALPHABET if c.isdigit()])
                    if len(st.session_state.test_answers) >= total_questions:
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

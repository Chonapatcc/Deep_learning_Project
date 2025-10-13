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
import io
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src import config
from src.config import (
    PreprocessConfig, HandDetectionConfig, InferenceConfig,
    ModelConfig, PracticeModeConfig, TestModeConfig, UIConfig, DataConfig,
    get_preprocess_function, get_resize_dimensions, get_color_conversion,
    THEMES
)

# Import utility functions
from src.utils.model_loader import init_mediapipe, load_models
from src.utils.prediction import predict_letter
from src.utils.hand_processing import extract_keypoints, calculate_bbox
from src.utils.letter_data import get_letter_instructions
from src.utils.confirmation import ConfirmationManager

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


# Page configuration
st.set_page_config(
    page_title="ASL Fingerspelling Trainer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_theme_css(theme_name='light'):
    """Generate dynamic CSS based on selected theme"""
    theme = THEMES.get(theme_name, THEMES['light'])
    
    return f"""
    <style>
    /* Theme Variables */
    :root {{
        --primary-color: {theme['primary']};
        --secondary-color: {theme['secondary']};
        --background-color: {theme['background']};
        --text-color: {theme['text']};
        --card-bg: {theme['card_bg']};
        --success-color: {theme['success']};
        --warning-color: {theme['warning']};
        --error-color: {theme['error']};
    }}
    
    /* Apply theme */
    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
        transition: all 0.3s ease;
    }}
    
    .main-header {{
        font-size: 1.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
    }}
    .subtitle {{
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
    }}
    .stat-card {{
        background: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        color: var(--text-color);
        text-align: center;
        border: 2px solid var(--primary-color);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    .stat-value {{
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
        animation: pulse 2s ease-in-out infinite;
    }}
    .stat-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    .feedback-correct {{
        background-color: var(--success-color);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1rem;
        text-align: center;
        animation: slideInBounce 0.5s ease;
    }}
    .feedback-confirming {{
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(46, 125, 50, 0.5);
    }}
    
    .feedback-confirming::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        transition: width 0.1s ease-out;
        z-index: 0;
        box-shadow: 2px 0 10px rgba(76, 175, 80, 0.6);
    }}
    
    .feedback-confirming-text {{
        position: relative;
        z-index: 1;
    }}
    .feedback-incorrect {{
        background-color: var(--error-color);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1rem;
        text-align: center;
        animation: shake 0.5s ease;
    }}
    .feedback-warning {{
        background-color: var(--warning-color);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 1rem;
        text-align: center;
        animation: fadeIn 0.3s ease;
    }}
    
    /* Enhanced Animations */
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideInBounce {{
        0% {{
            opacity: 0;
            transform: translateY(-50px) scale(0.8);
        }}
        50% {{
            transform: translateY(10px) scale(1.05);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}
    
    @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-10px); }}
        75% {{ transform: translateX(10px); }}
    }}
    
    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.8;
        }}
    }}
    
    @keyframes glow {{
        0%, 100% {{
            box-shadow: 0 0 5px var(--primary-color);
        }}
        50% {{
            box-shadow: 0 0 20px var(--primary-color);
        }}
    }}
    
    /* Add smooth transitions to buttons */
    .stButton > button {{
        transition: all 0.3s ease;
        animation: fadeInUp 0.4s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Animate checkboxes */
    .stCheckbox {{
        animation: fadeIn 0.3s ease;
    }}
    
    /* Animate images */
    img {{
        animation: fadeIn 0.5s ease;
        transition: transform 0.3s ease;
    }}
    
    img:hover {{
        transform: scale(1.02);
    }}
    </style>
    """

# Custom CSS - Apply selected theme
st.markdown(generate_theme_css(st.session_state.get('selected_theme', 'light')), unsafe_allow_html=True)

# Original CSS (remove old static CSS)
st.markdown("""
    <style>
    /* Additional custom styles */
    </style>
""", unsafe_allow_html=True)


# Initialize MediaPipe
mp_hands, mp_drawing, hands = init_mediapipe()

# Initialize session state
if 'selected_theme' not in st.session_state:
    st.session_state.selected_theme = 'light'

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

# Initialize confirmation managers
if 'practice_confirmation' not in st.session_state:
    st.session_state.practice_confirmation = ConfirmationManager(required_duration=1.5)

if 'translation_confirmation' not in st.session_state:
    st.session_state.translation_confirmation = ConfirmationManager(required_duration=1.5)

# Alphabet - Only A-Z letters (removed 0-9 for simplicity)
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Load models on startup
MODELS_DATA = load_models()


def predict_asl(frame, hand_landmarks, keypoint_buffer, alphabet):
    """
    Unified prediction function that works with both TensorFlow and PyTorch models
    
    Args:
        frame: Current camera frame (for PyTorch landmark extraction)
        hand_landmarks: MediaPipe hand landmarks
        keypoint_buffer: Buffer of keypoints (for TensorFlow)
        alphabet: List of possible characters
    
    Returns:
        tuple: (predicted_letter, confidence)
    """
    if MODELS_DATA is None:
        return None, 0.0
    
    model_type = MODELS_DATA.get('model_type')
    
    if model_type == 'pytorch_landmark':
        # PyTorch: Use Predictor.predict_frame() directly
        # It handles MediaPipe extraction internally
        predictor = MODELS_DATA['predictor']
        try:
            predicted_letter, confidence = predictor.predict_frame(frame)
            # Ensure confidence is never None
            if confidence is None:
                confidence = 0.0
            return predicted_letter, confidence
        except Exception as e:
            # Log error silently in backend
            import logging
            logging.error(f"Prediction error: {e}")
            return None, 0.0
    
    elif model_type == 'cnn':
        # TensorFlow: Use existing predict_letter() function
        return predict_letter(
            keypoint_buffer,
            MODELS_DATA,
            alphabet,
            landmarks=hand_landmarks
        )
    
    else:
        # Log error silently in backend
        import logging
        logging.error(f"Unknown model type: {model_type}")
        return None, 0.0


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Fingerspelling Trainer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">ฝึกฝนภาษามือตัวอักษรภาษาอังกฤษ A-Z</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        
        st.markdown("---")
        
        # Theme Selector
        st.markdown("### 🎨 ธีม")
        
        theme_options = {theme_key: theme_data['name'] 
                        for theme_key, theme_data in THEMES.items()}
        
        selected_theme = st.selectbox(
            "เลือกธีมที่ต้องการ",
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=list(theme_options.keys()).index(st.session_state.selected_theme),
            key='theme_selector'
        )
        
        # Update theme if changed
        if selected_theme != st.session_state.selected_theme:
            st.session_state.selected_theme = selected_theme
            st.rerun()
        
        # Show color preview
        theme_data = THEMES[st.session_state.selected_theme]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background-color: {theme_data['primary']}; 
                        color: white; padding: 8px; border-radius: 5px; text-align: center; font-size: 0.8rem;">
                สีหลัก
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background-color: {theme_data['secondary']}; 
                        color: white; padding: 8px; border-radius: 5px; text-align: center; font-size: 0.8rem;">
                สีรอง
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        mode = st.radio(
            "เลือกโหมด",
            ["📚 Learning Mode", "✋ Practice Mode", "🎯 Test Mode", "🌐 Real-time Translation"],
            index=3  # Default to Real-time Translation
        )
        
        st.markdown("---")
        
        if mode == "✋ Practice Mode":
            # Store previous letter to detect changes
            prev_letter = st.session_state.current_letter
            
            selected_letter = st.selectbox(
                "เลือกตัวอักษร",
                ALPHABET,
                index=ALPHABET.index(st.session_state.current_letter),
                key='letter_selector'
            )
            
            # Update and rerun if changed
            if selected_letter != prev_letter:
                st.session_state.current_letter = selected_letter
                # Clear buffers when switching letters
                st.session_state.keypoint_buffer = []
                st.session_state.frame_buffer = []
                st.rerun()
            
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
        st.markdown("### 📖 เกี่ยวกับ")
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
        # Show selection grid - Only letters A-Z
        st.info("ℹ️ เลือกตัวอักษรเพื่อดูตัวอย่างท่ามือและคำแนะนำ")
        
        # Create grid for alphabet only
        st.markdown("### เลือกตัวอักษรที่ต้องการเรียนรู้")
        cols_per_row = 7
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        rows = [letters[i:i+cols_per_row] for i in range(0, len(letters), cols_per_row)]
        
        for row in rows:
            cols = st.columns(cols_per_row)
            for idx, letter in enumerate(row):
                with cols[idx]:
                    if st.button(letter, key=f"learn_{letter}", width='stretch'):
                        st.session_state.selected_learning_char = letter
                        st.rerun()

def show_letter_detail(letter):
    """Show detailed information about a letter - FULLSCREEN"""
    
    # Initialize image index in session state
    if 'learning_image_index' not in st.session_state:
        st.session_state.learning_image_index = 0
    
    # Back button at top
    if st.button("← กลับไปเลือกตัวอักษร", key="back_to_selection"):
        st.session_state.selected_learning_char = None
        st.session_state.learning_image_index = 0  # Reset index
        st.rerun()
    
    # Compact header
    st.markdown(f"## ตัวอักษร {letter} - ตัวอย่างท่ามือ")
    
    # Large character display at the top
    st.markdown(f"<div style='font-size: 3.5rem; text-align: center; color: #4A90E2; font-weight: bold; margin: 10px 0 20px 0;'>{letter}</div>", 
               unsafe_allow_html=True)
    
    # Two column layout: Image on left, Instructions on right
    col_img, col_info = st.columns([3, 2])
    
    with col_img:
        # Load images from dataset
        dataset_path = os.path.join(DataConfig.DATASET_PATH, letter.upper())
        if os.path.exists(dataset_path):
            images = glob.glob(f"{dataset_path}/*.jpg") + glob.glob(f"{dataset_path}/*.jpeg") + glob.glob(f"{dataset_path}/*.png")
            
            if images:
                # Ensure index is within bounds
                if st.session_state.learning_image_index >= len(images):
                    st.session_state.learning_image_index = 0
                
                # Display current image - smaller for better fit
                current_image_path = images[st.session_state.learning_image_index]
                try:
                    img = Image.open(current_image_path)
                    # Resize to smaller size
                    img.thumbnail((400, 400))
                    st.image(img, width=400)
                except:
                    st.error("❌ ไม่สามารถโหลดรูปภาพได้")
                
                # Navigation buttons - compact
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                
                with nav_col1:
                    if st.button("⬅️", width='stretch',
                                disabled=(st.session_state.learning_image_index == 0),
                                help="ภาพก่อนหน้า"):
                        st.session_state.learning_image_index -= 1
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"<div style='text-align: center; padding: 5px;'><b>{st.session_state.learning_image_index + 1}/{len(images)}</b></div>", 
                               unsafe_allow_html=True)
                
                with nav_col3:
                    if st.button("➡️", width='stretch',
                                disabled=(st.session_state.learning_image_index == len(images) - 1),
                                help="ภาพถัดไป"):
                        st.session_state.learning_image_index += 1
                        st.rerun()
            else:
                st.warning(f"⚠️ ไม่พบรูปภาพ")
        else:
            st.warning(f"⚠️ ไม่พบโฟลเดอร์")
    
    with col_info:
        # Instructions
        st.markdown("### 💡 วิธีทำท่า")
        instructions = get_letter_instructions(letter)
        st.info(instructions)
        
        # Tips
        st.markdown("### 🎯 คำแนะนำ")
        st.success("""
        - วางมือให้อยู่ตรงกลางกรอบภาพ
        - ทำท่าช้าๆ และชัดเจน
        - ศึกษาจากตัวอย่างภาพหลายๆ มุม
        - ฝึกฝนซ้ำๆ จนชำนาญ
        """)

def show_practice_mode():
    """Practice Mode - Real-time practice with feedback"""
    st.header(f"✋ โหมดฝึกฝน - ตัวอักษร {st.session_state.current_letter}")
    
    # Initialize practice start time if not exists
    if 'practice_start_time' not in st.session_state:
        st.session_state.practice_start_time = time.time()
    
    # Create placeholders for real-time stats updates
    stats_placeholder = st.empty()
    
    # Function to update stats display
    def update_stats_display():
        elapsed_seconds = int(time.time() - st.session_state.practice_start_time)
        elapsed_minutes = elapsed_seconds // 60
        elapsed_secs = elapsed_seconds % 60
        
        attempts = st.session_state.stats.get('attempts', 0)
        correct = st.session_state.stats.get('correct', 0)
        success_rate = (correct / attempts * 100) if attempts > 0 else 0
        
        stats_placeholder.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
            <div class="stat-card" style="flex: 1;">
                <div class="stat-value">{attempts}</div>
                <div class="stat-label">ครั้งที่ลอง</div>
            </div>
            <div class="stat-card" style="flex: 1;">
                <div class="stat-value">{correct}</div>
                <div class="stat-label">ถูกต้อง</div>
            </div>
            <div class="stat-card" style="flex: 1;">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">อัตราความสำเร็จ</div>
            </div>
            <div class="stat-card" style="flex: 1;">
                <div class="stat-value">{elapsed_minutes:02d}:{elapsed_secs:02d}</div>
                <div class="stat-label">เวลาที่ใช้</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Initial display
    update_stats_display()
    
    # Two-column layout: Example Image on left | Instructions + Camera on right
    col_ref, col_inst = st.columns([1, 1])
    
    with col_ref:
        st.markdown("### 📸 ตัวอย่าง")
        
        # Load reference images from dataset - show only 1 image
        letter = st.session_state.current_letter
        dataset_path = os.path.join(DataConfig.DATASET_PATH, letter.upper())
        
        if os.path.exists(dataset_path):
            images = glob.glob(f"{dataset_path}/*.jpg") + glob.glob(f"{dataset_path}/*.jpeg") + glob.glob(f"{dataset_path}/*.png")
            
            if images:
                # Show only 1 image to save space - use caching to prevent MediaFileStorageError
                try:
                    # Read image as bytes and display directly to avoid caching issues
                    with open(images[0], 'rb') as img_file:
                        img_bytes = img_file.read()
                    img = Image.open(io.BytesIO(img_bytes))
                    # Resize to smaller size to fit better
                    img.thumbnail((300, 300))
                    st.image(img, width=300)
                except Exception as e:
                    st.warning("ไม่สามารถโหลดรูปได้")
            else:
                st.info("ไม่พบรูปตัวอย่าง")
        else:
            st.info("ไม่พบโฟลเดอร์ข้อมูล")
    
    with col_inst:
        # Instructions
        st.markdown("### 💡 วิธีทำท่า")
        instructions = get_letter_instructions(letter)
        st.info(instructions)
        
        # Camera section - below instructions in same column
        st.markdown("### 📹 กล้อง")
        run_camera = st.checkbox("เปิดกล้อง", value=True, key="practice_camera_checkbox")
        
        if run_camera:
            run_webcam_detection(update_stats_display, camera_width=1280)

def run_test_detection(target_letter, update_timer_callback=None):
    """Run camera detection for test mode with auto-skip"""
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Full resolution for test mode
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการอนุญาตใช้งานกล้อง")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง", key="test_stop_camera")
    
    # Detection tracking
    detection_frames = 0
    required_frames = 30  # Need 30 consecutive frames for confirmation
    
    while not stop_button:
        # Update timer in real-time if callback provided
        if update_timer_callback:
            elapsed_time, remaining_time = update_timer_callback()
            
            # Check if time is up
            if remaining_time == 0:
                cap.release()
                st.rerun()
                return
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB once (optimized)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store frame for CNN models (avoid unnecessary copy)
        st.session_state.frame_buffer.append(frame)
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        skeleton_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                skeleton_detected = True
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.keypoint_buffer.append(keypoints)
                
                if len(st.session_state.keypoint_buffer) > 60:
                    st.session_state.keypoint_buffer.pop(0)
                
                # Check if enough data
                if len(st.session_state.keypoint_buffer) >= 15:
                    predicted_letter, confidence = predict_asl(
                        frame,
                        hand_landmarks,
                        st.session_state.keypoint_buffer,
                        ALPHABET
                    )
                    
                    if predicted_letter and confidence is not None and confidence >= 0.75:
                        if predicted_letter == target_letter:
                            detection_frames += 1
                            progress = int((detection_frames / required_frames) * 100)

                            feedback_message = f"✅ ตรวจจับได้: {predicted_letter} ({confidence*100:.0f}%) - กำลังยืนยัน... {progress}%"
                            feedback_class = "feedback-confirming"
                            feedback_progress = progress
                            
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
                            feedback_progress = 0
                    else:
                        detection_frames = 0
                        feedback_message = "⏳ กำลังตรวจจับ..."
                        feedback_class = "feedback-warning"
                        feedback_progress = 0
        else:
            detection_frames = 0
            feedback_message = "✋ ไม่พบมือ - กรุณาแสดงมือในกล้อง"
            feedback_class = "feedback-warning"
            feedback_progress = 0
        
        # Add skeleton detection indicator
        if skeleton_detected:
            cv2.putText(frame, "Hand Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display - larger size
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", width=1280)
        
        # Show feedback with progress bar
        if feedback_message:
            if feedback_class == "feedback-confirming":
                feedback_placeholder.markdown(
                    f'''<div class="feedback-confirming">
                        <div class="feedback-confirming-text">{feedback_message}</div>
                        <div style="position: absolute; left: 0; top: 0; width: {feedback_progress}%; height: 100%; 
                             background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%); 
                             border-radius: 10px; z-index: 0;
                             box-shadow: 2px 0 10px rgba(76, 175, 80, 0.6);"></div>
                    </div>''',
                    unsafe_allow_html=True
                )
            elif feedback_class == "feedback-correct":
                feedback_placeholder.success(feedback_message)
            elif feedback_class == "feedback-incorrect":
                feedback_placeholder.error(feedback_message)
            else:
                feedback_placeholder.info(feedback_message)
    
    cap.release()

def run_webcam_detection(update_stats_callback=None, camera_width=1280):
    """Run webcam with hand detection"""
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    # Set camera resolution based on width parameter
    camera_height = int(camera_width * 9 / 16)  # Maintain 16:9 aspect ratio
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการอนุญาตใช้งานกล้อง")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง")
    
    # Frame-based confirmation (like test mode)
    detection_frames = 0
    required_frames = 30  # Need 30 consecutive frames for confirmation
    
    while not stop_button:
        # Update stats in real-time if callback provided
        if update_stats_callback:
            update_stats_callback()
        
        ret, frame = cap.read()
        if not ret:
            st.error("❌ ไม่สามารถอ่านข้อมูลจากกล้องได้")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB once for MediaPipe (more efficient)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store raw frame for CNN models (avoid unnecessary copy)
        st.session_state.frame_buffer.append(frame)
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        skeleton_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                skeleton_detected = True
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.keypoint_buffer.append(keypoints)
                
                # Keep only recent frames
                if len(st.session_state.keypoint_buffer) > 60:
                    st.session_state.keypoint_buffer.pop(0)
                
                # Check bounding box (only for size check)
                bbox = calculate_bbox(hand_landmarks)
                hand_size = max(bbox['width'], bbox['height'])
                
                if hand_size < 0.2:
                    feedback_message = "🔍 มือเล็กไป กรุณาเข้าใกล้กล้อง"
                    feedback_class = "feedback-warning"
                    detection_frames = 0
                elif len(st.session_state.keypoint_buffer) >= 15:
                    # Predict
                    predicted_letter, confidence = predict_asl(
                        frame,
                        hand_landmarks,
                        st.session_state.keypoint_buffer,
                        ALPHABET
                    )
                    
                    if predicted_letter and confidence is not None and confidence >= 0.75:
                        is_correct = predicted_letter == st.session_state.current_letter
                        
                        if is_correct:
                            # Correct letter detected
                            detection_frames += 1
                            progress = int((detection_frames / required_frames) * 100)
                            
                            if detection_frames >= required_frames:
                                # Confirmed correct answer
                                feedback_message = f"✅ ถูกต้อง! {predicted_letter} ({confidence*100:.0f}%)"
                                feedback_class = "feedback-correct"
                                
                                # Update stats
                                st.session_state.stats['attempts'] += 1
                                st.session_state.stats['correct'] += 1
                                st.session_state.stats['total_accuracy'] += confidence * 100
                                
                                # Clear buffer and reset
                                st.session_state.keypoint_buffer = []
                                detection_frames = 0
                                time.sleep(0.5)
                            else:
                                # Confirming
                                feedback_message = f"🎯 กำลังยืนยัน: {predicted_letter} ({progress}%)"
                                feedback_class = "feedback-confirming"
                                feedback_progress = progress
                        else:
                            # Wrong letter
                            detection_frames = 0
                            feedback_message = f"❌ ตรวจจับได้: {predicted_letter} ({confidence*100:.0f}%) - ต้องการ: {st.session_state.current_letter}"
                            feedback_class = "feedback-incorrect"
                            feedback_progress = 0
                    else:
                        detection_frames = 0
                        feedback_message = "⏳ กำลังตรวจจับ..."
                        feedback_class = "feedback-warning"
                        feedback_progress = 0
                else:
                    detection_frames = 0
                    feedback_message = "⏳ กำลังรวบรวมข้อมูล..."
                    feedback_class = "feedback-warning"
                    feedback_progress = 0
        else:
            detection_frames = 0
            feedback_message = "✋ ไม่พบมือ กรุณาแสดงมือในกล้อง"
            feedback_class = "feedback-warning"
            feedback_progress = 0
        
        # Add skeleton detection indicator
        if skeleton_detected:
            cv2.putText(frame, "Hand Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame with dynamic width based on parameter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", width=1280)
        
        # Display feedback with progress bar
        if feedback_message:
            if feedback_class == "feedback-confirming":
                feedback_placeholder.markdown(
                    f'''<div class="feedback-confirming" style="--progress: {feedback_progress}%;">
                        <div class="feedback-confirming-text">{feedback_message}</div>
                        <div style="position: absolute; left: 0; top: 0; width: {feedback_progress}%; height: 100%; 
                             background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%); 
                             border-radius: 10px; z-index: 0;
                             box-shadow: 2px 0 10px rgba(76, 175, 80, 0.6);"></div>
                    </div>''',
                    unsafe_allow_html=True
                )
            else:
                feedback_placeholder.markdown(
                    f'<div class="{feedback_class}">{feedback_message}</div>',
                    unsafe_allow_html=True
                )
    
    cap.release()

def run_translation_detection(update_translation_callback=None):
    """Run camera detection for real-time translation mode - optimized for performance"""
    hands, mp_drawing, mp_hands = init_mediapipe()
    
    FRAME_WINDOW = st.image([])
    feedback_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        st.error("❌ ไม่สามารถเปิดกล้องได้")
        return
    
    stop_button = st.button("⏹️ หยุดกล้อง")
    
    # Frame-based confirmation (like test mode)
    detection_frames = 0
    required_frames = 30  # Need 30 consecutive frames for confirmation
    last_confirmed_letter = None  # Track to avoid duplicates
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB once for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store raw frame for CNN models (before any processing)
        st.session_state.frame_buffer.append(frame.copy())
        if len(st.session_state.frame_buffer) > 30:
            st.session_state.frame_buffer.pop(0)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        feedback_message = ""
        feedback_class = "feedback-warning"
        skeleton_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                skeleton_detected = True
                
                # Extract keypoints
                keypoints = extract_keypoints(hand_landmarks)
                st.session_state.translation_buffer.append(keypoints)
                
                # Keep buffer size manageable
                if len(st.session_state.translation_buffer) > 60:
                    st.session_state.translation_buffer.pop(0)
                
                # Check hand size
                bbox = calculate_bbox(hand_landmarks)
                hand_size = max(bbox['width'], bbox['height'])
                
                if hand_size < 0.2:
                    feedback_message = "🔍 มือเล็กไป กรุณาเข้าใกล้กล้อง"
                    feedback_class = "feedback-warning"
                    detection_frames = 0
                    feedback_progress = 0
                elif len(st.session_state.translation_buffer) >= 15:
                    # Predict letter
                    predicted_letter, confidence = predict_asl(
                        frame,
                        hand_landmarks,
                        st.session_state.translation_buffer,
                        ALPHABET
                    )
                    
                    if predicted_letter and confidence is not None and confidence >= 0.75:
                        # Check if same as last confirmed (avoid duplicates)
                        if predicted_letter == last_confirmed_letter:
                            detection_frames += 1
                            progress = int((detection_frames / required_frames) * 100)
                            
                            if detection_frames >= required_frames:
                                # Confirmed - add letter
                                st.session_state.translated_text += predicted_letter
                                feedback_message = f"✅ เพิ่ม: {predicted_letter}"
                                feedback_class = "feedback-correct"
                                feedback_progress = 100
                                
                                # Update displays in real-time
                                if update_translation_callback:
                                    update_translation_callback()
                                
                                # Reset and wait for hand to be removed
                                detection_frames = 0
                                last_confirmed_letter = None
                                st.session_state.translation_buffer = []
                                time.sleep(0.5)
                            else:
                                # Confirming same letter
                                feedback_message = f"🎯 กำลังยืนยัน: {predicted_letter} ({progress}%)"
                                feedback_class = "feedback-confirming"
                                feedback_progress = progress
                        else:
                            # New letter detected
                            last_confirmed_letter = predicted_letter
                            detection_frames = 1
                            feedback_message = f"🎯 ตรวจจับ: {predicted_letter} ({confidence*100:.0f}%)"
                            feedback_class = "feedback-warning"
                            feedback_progress = 0
                    else:
                        detection_frames = 0
                        last_confirmed_letter = None
                        feedback_message = "⏳ กำลังวิเคราะห์..."
                        feedback_class = "feedback-warning"
                        feedback_progress = 0
                else:
                    detection_frames = 0
                    last_confirmed_letter = None
                    feedback_message = "⏳ กำลังรวบรวมข้อมูล..."
                    feedback_class = "feedback-warning"
                    feedback_progress = 0
        else:
            detection_frames = 0
            last_confirmed_letter = None
            feedback_message = "👋 กรุณาแสดงมือต่อกล้อง"
            feedback_class = "feedback-warning"
            feedback_progress = 0
        
        # Add skeleton detection indicator (text only, no drawing)
        if skeleton_detected:
            cv2.putText(frame, "Hand Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame - convert once for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", width=1280)
        
        # Display feedback with progress bar
        if feedback_message:
            if feedback_class == "feedback-confirming":
                feedback_placeholder.markdown(
                    f'''<div class="feedback-confirming">
                        <div class="feedback-confirming-text">{feedback_message}</div>
                        <div style="position: absolute; left: 0; top: 0; width: {feedback_progress}%; height: 100%; 
                             background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%); 
                             border-radius: 10px; z-index: 0;
                             box-shadow: 2px 0 10px rgba(76, 175, 80, 0.6);"></div>
                    </div>''',
                    unsafe_allow_html=True
                )
            else:
                feedback_placeholder.markdown(
                    f'<div class="{feedback_class}">{feedback_message}</div>',
                    unsafe_allow_html=True
                )
    
    cap.release()

def show_translation_mode():
    """Real-time Translation Mode - Translate ASL to text and refine with Gemini API"""
    st.header("🌐 โหมดแปลภาษา Real-time")
    
    # Check Gemini API availability
    if not GEMINI_AVAILABLE:
        st.error("❌ โหมดนี้ต้องการการติดตั้งเพิ่มเติม กรุณาติดต่อผู้ดูแลระบบ")
        return
    
    # API Key configuration - compact
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.info("🔑 กรุณาใส่ API Key ด้านล่าง")
        api_key = st.text_input("API Key", type="password", placeholder="กรุณาใส่ API Key")

    if not api_key:
        st.warning("⚠️ กรุณาใส่ API Key เพื่อใช้งานโหมดนี้")
        return
    
    # Configure Gemini with token limits
    try:
        genai.configure(api_key=api_key)
        # Configure with 500 token limits for input and output
        generation_config = {
            "max_output_tokens": 500,
            "temperature": 0.7,
        }
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อได้ กรุณาตรวจสอบ API Key")
        return
    
    # Compact controls in one row
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 2])
    
    with ctrl_col1:
        clear_buffer = st.button("🗑️ ล้าง", width='stretch')
    
    with ctrl_col2:
        st.metric("ตัวอักษร", len(st.session_state.translated_text))
    
    with ctrl_col3:
        st.metric("คำ", len(st.session_state.translated_text.split()))
    
    with ctrl_col4:
        refine_button = st.button("✨ ปรับปรุงข้อความ", type="primary", width='stretch')
    
    if clear_buffer:
        st.session_state.translated_text = ""
        st.session_state.refined_text = ""
        st.session_state.translation_buffer = []
        st.rerun()
    
    # Compact display areas - side by side
    col_text, col_refined = st.columns(2)
    
    with col_text:
        st.markdown("### 📝 ข้อความ")
        translated_display = st.empty()
    
    with col_refined:
        st.markdown("### ✨ ปรับปรุงแล้ว")
        refined_display = st.empty()
    
    # Word display placeholder
    word_display = st.empty()
    
    # Function to update displays in real-time
    def update_translation_displays():
        # Update translated text
        if st.session_state.translated_text:
            translated_display.markdown(f"""
            <div style='background-color: #E8F4F8; padding: 10px; border-radius: 8px;'>
                {st.session_state.translated_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Update word display - compact
        if st.session_state.translated_text:
            words = st.session_state.translated_text.split()
            if words:
                word_display.info(f"💬 คำล่าสุด: **{words[-1]}**")
            else:
                word_display.info("💬 เริ่มทำท่ามือเพื่อสร้างคำ")
        else:
            word_display.info("💬 เริ่มทำท่ามือเพื่อสร้างคำ")
    
    # Initial display
    update_translation_displays()
    
    # Manual refine button - handle refined text display
    if st.session_state.refined_text:
        refined_display.markdown(f"""
        <div style='background-color: #E8F8E8; padding: 10px; border-radius: 8px;'>
            {st.session_state.refined_text}
        </div>
        """, unsafe_allow_html=True)
    
    if refine_button and st.session_state.translated_text:
        with st.spinner("🔄 กำลังปรับปรุงข้อความด้วย Gemini..."):
            try:
                # Truncate text if too long to stay within 500 token limit
                text_to_refine = st.session_state.translated_text[:400]  # Limit input length
                
                # Improved prompt to avoid recitation issues and return only refined text
                prompt = f"""Improve the following text by correcting grammar and making it more readable. Maintain the original meaning and language. Return ONLY the refined text without any explanations, introductions, or additional commentary.

Text: {text_to_refine}

Refined text:"""
                
                response = model.generate_content(prompt)
                
                # Check if response has valid content
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    
                    # Check finish reason
                    if candidate.finish_reason == 1:  # STOP - successful completion
                        if hasattr(response, 'text') and response.text:
                            st.session_state.refined_text = response.text.strip()
                            st.rerun()
                        else:
                            st.error("❌ ไม่ได้รับคำตอบจาก API")
                    elif candidate.finish_reason == 2:  # RECITATION
                        st.warning("⚠️ ข้อความไม่สามารถปรับปรุงได้ กรุณาลองแก้ไขข้อความหรือเขียนใหม่")
                    elif candidate.finish_reason == 3:  # SAFETY
                        st.warning("⚠️ เนื้อหาถูกบล็อกโดยระบบความปลอดภัย")
                    elif candidate.finish_reason == 4:  # MAX_TOKENS
                        st.warning("⚠️ ข้อความยาวเกินไป กรุณาลดความยาวข้อความ")
                    else:
                        st.error(f"❌ การปรับปรุงหยุดโดยไม่คาดคิด (รหัส: {candidate.finish_reason})")
                else:
                    st.error("❌ ไม่ได้รับการตอบกลับจาก API")
                    
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                st.info("💡 กรุณาตรวจสอบ: API Key, การเชื่อมต่ออินเทอร์เน็ต, หรือโควต้าของ API")
    
    # Camera section - compact
    st.markdown("### 📹 กล้อง")
    
    # Use dedicated translation detection function
    run_translation_detection(update_translation_displays)

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
        if st.button("🚀 เริ่มทำแบบทดสอบ", type="primary", width='stretch'):
            st.session_state.test_started = True
            st.session_state.test_answers = []
            st.session_state.test_start_time = time.time()
            # Randomize alphabet order for test
            import random
            st.session_state.test_alphabet = list(ALPHABET)
            random.shuffle(st.session_state.test_alphabet)
            st.rerun()
    else:
        # Create placeholders for real-time updates
        timer_placeholder = st.empty()
        
        # Function to update timer display
        def update_timer_display():
            elapsed_time = int(time.time() - st.session_state.test_start_time)
            remaining_time = max(0, 900 - elapsed_time)  # 15 minutes
            
            col1, col2, col3 = timer_placeholder.columns(3)
            with col1:
                st.metric("ข้อที่", f"{len(st.session_state.test_answers) + 1}/26")
            with col2:
                # Real-time elapsed timer
                st.metric("เวลาที่ใช้", f"{elapsed_time//60:02d}:{elapsed_time%60:02d}")
            with col3:
                # Real-time remaining timer
                st.metric("เวลาคงเหลือ", f"{remaining_time//60:02d}:{remaining_time%60:02d}")
            
            return elapsed_time, remaining_time
        
        # Initial display
        elapsed_time, remaining_time = update_timer_display()
        
        if remaining_time == 0:
            show_test_results()
        elif len(st.session_state.test_answers) >= len(ALPHABET):
            # All questions completed
            show_test_results()
        else:
            # Show current question - compact layout
            # Use randomized alphabet order
            current_letter = st.session_state.test_alphabet[len(st.session_state.test_answers)]
            
            # Compact two-column: Character + Instructions | Camera
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### ทำท่าตัวอักษร:")
                # Larger character display (no instructions)
                st.markdown(f"<div style='font-size: 5rem; text-align: center; color: #4A90E2; font-weight: bold; padding: 40px 0;'>{current_letter}</div>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📷 เปิดกล้องเพื่อจับท่ามือ")
                run_test_camera = st.checkbox("เปิดกล้อง", value=True, key="test_camera")
                
                if run_test_camera:
                    st.info(f"📌 ทำท่ามือตัวอักษร {current_letter} และกดยืนยันเมื่อพร้อม")
                    run_test_detection(current_letter, update_timer_display)
                else:
                    st.warning("⚠️ เปิดกล้องเพื่อเริ่มทำแบบทดสอบ")
                
                st.info("📹 เปิดกล้องและทำท่ามือ - ระบบจะข้ามข้อถัดไปอัตโนมัติเมื่อตรวจจับถูกต้อง")
                
                # Manual skip (if needed)
                if st.button("⏭️ ข้ามข้อนี้ (ไม่ทำหรือไม่เปิดกล้อง)", width='stretch'):
                    # Mark as incorrect if skipped
                    st.session_state.test_answers.append({
                        'question': current_letter,
                        'answer': None,
                        'correct': False
                    })
                    
                    # Total questions is now just 26 letters (A-Z)
                    if len(st.session_state.test_answers) >= len(ALPHABET):
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
            <div style='font-size: 2.5rem; font-weight: bold;'>{correct_count}</div>
            <div style='font-size: 1.2rem;'>/ {total}</div>
            <div style='font-size: 1rem; margin-top: 10px;'>คะแนนรวม</div>
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

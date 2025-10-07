"""
Model loading utilities for ASL Fingerspelling Trainer
Supports: TensorFlow/Keras (.h5, .keras), PyTorch (.pt, .pth, landmark-based), ONNX (.onnx)
"""

import streamlit as st
import mediapipe as mp
import os
import pickle
import numpy as np

# Import configuration
import config
ModelConfig = config.ModelConfig

# Try to import TensorFlow/Keras
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import PyTorch ASL classes
try:
    from pytorch_asl.models.classifier import ASLClassifier
    from pytorch_asl.controllers.predictor import Predictor
    PYTORCH_ASL_AVAILABLE = True
except ImportError as e:
    PYTORCH_ASL_AVAILABLE = False
    PYTORCH_ASL_IMPORT_ERROR = str(e)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try to import sklearn for label encoder
try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe Hands"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return hands, mp_drawing, mp_hands


@st.cache_resource
def load_pytorch_landmark_model(model_path, encoder_path, device='auto'):
    """
    Load PyTorch landmark-based ASL model (from pytorch_asl package)
    
    Returns:
        Predictor instance or None if failed
    """
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch not installed")
        st.info("📦 Install with: `pip install torch torchvision`")
        st.code("pip install torch torchvision", language="bash")
        return None
    
    if not PYTORCH_ASL_AVAILABLE:
        st.error("❌ PyTorch ASL package import failed")
        if 'PYTORCH_ASL_IMPORT_ERROR' in globals():
            st.error(f"Import error: {PYTORCH_ASL_IMPORT_ERROR}")
        st.info("📁 Check that pytorch_asl/ folder exists with all required files")
        st.info("💡 Try: `pip install mediapipe scikit-learn`")
        return None
    
    try:
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load predictor
        predictor = Predictor(model_path, encoder_path, device=device)
        
        return predictor
        
    except Exception as e:
        st.error(f"❌ Failed to load PyTorch landmark model: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


@st.cache_resource
def load_model(model_name=None):
    """
    Load a deep learning model by name
    Supports: TensorFlow (.h5, .keras), PyTorch (.pt, .pth), ONNX (.onnx)
    
    Args:
        model_name: Name of model file (e.g., 'resnet50_improved', 'model.pt')
                   If None, will try to load from config or find first available model
    
    Returns:
        dict with 'model', 'label_encoder', 'model_name', 'model_type', 'framework'
        or None if failed
    """
    # Get model paths to try
    if model_name:
        # User specified model name
        if not any(model_name.endswith(ext) for ext in ['.h5', '.keras', '.pt', '.pth', '.onnx']):
            # Try all extensions
            model_paths = [
                f"models/{model_name}.h5",
                f"models/{model_name}.keras",
                f"models/{model_name}.pt",
                f"models/{model_name}.pth",
                f"models/{model_name}.onnx"
            ]
        else:
            model_paths = [f"models/{model_name}"]
    else:
        # Auto-detect: try common model names in priority order
        model_paths = [
            # TensorFlow/Keras models
            "models/ayumi_chan.h5",
            "models/resnet50_app2_2.h5",
            "models/resnet50.h5",
            "models/trained_model_10epochs.h5",
            "models/best_transfer_CNN.keras",
            "models/asl_cnn_model.h5",
            "models/asl_cnn_model.keras",
            # PyTorch models
            "models/resnet50_improved.pt",
            "models/resnet50_improved.pth",
            "models/resnet50.pt",
            "models/resnet50.pth",
            "models/asl_model.pt",
            "models/asl_model.pth",
            # ONNX models
            "models/resnet50_improved.onnx",
            "models/resnet50.onnx",
            "models/asl_model.onnx"
        ]
    
    # Try to load model
    loaded_model = None
    loaded_path = None
    framework = None
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
        
        model_basename = os.path.basename(model_path)
        
        try:
            # Determine model type by extension
            if model_path.endswith(('.h5', '.keras')):
                # TensorFlow/Keras model
                if not TF_AVAILABLE:
                    st.warning(f"⚠️ TensorFlow not installed. Cannot load {model_basename}")
                    continue
                
                loaded_model = keras.models.load_model(model_path)
                framework = 'tensorflow'
                st.success(f"✅ โหลด TensorFlow Model ({model_basename}) สำเร็จ!")
                loaded_path = model_path
                break
                
            elif model_path.endswith(('.pt', '.pth')):
                # PyTorch model
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ PyTorch not installed. Cannot load {model_basename}")
                    continue
                
                # Load PyTorch model
                loaded_model = torch.load(model_path, map_location='cpu')
                
                # If it's a state dict, need to know the architecture
                if isinstance(loaded_model, dict):
                    st.warning(f"⚠️ {model_basename} is a state_dict. Need model architecture to load.")
                    continue
                
                # Set to eval mode
                if isinstance(loaded_model, nn.Module):
                    loaded_model.eval()
                
                framework = 'pytorch'
                st.success(f"✅ โหลด PyTorch Model ({model_basename}) สำเร็จ!")
                loaded_path = model_path
                break
                
            elif model_path.endswith('.onnx'):
                # ONNX model
                if not ONNX_AVAILABLE:
                    st.warning(f"⚠️ ONNX Runtime not installed. Cannot load {model_basename}")
                    continue
                
                loaded_model = ort.InferenceSession(model_path)
                framework = 'onnx'
                st.success(f"✅ โหลด ONNX Model ({model_basename}) สำเร็จ!")
                loaded_path = model_path
                break
                
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลด {model_basename}: {e}")
    
    if loaded_model is None:
        # List available models
        available_models = []
        if os.path.exists("models"):
            available_models = [f for f in os.listdir("models") 
                              if f.endswith(('.h5', '.keras', '.pt', '.pth', '.onnx')) 
                              and not f.startswith('.')]
        
        # Check which frameworks are available
        frameworks_status = []
        if TF_AVAILABLE:
            frameworks_status.append("✅ TensorFlow (.h5, .keras)")
        else:
            frameworks_status.append("❌ TensorFlow (.h5, .keras) - run: pip install tensorflow")
        
        if TORCH_AVAILABLE:
            frameworks_status.append("✅ PyTorch (.pt, .pth)")
        else:
            frameworks_status.append("❌ PyTorch (.pt, .pth) - run: pip install torch")
        
        if ONNX_AVAILABLE:
            frameworks_status.append("✅ ONNX (.onnx)")
        else:
            frameworks_status.append("❌ ONNX (.onnx) - run: pip install onnxruntime")
        
        st.error(f"""❌ ไม่พบ Deep Learning Model!
        
📁 Models ที่มีในโฟลเดอร์:
{chr(10).join(['  - ' + f for f in available_models]) if available_models else '  (ไม่พบไฟล์ model)'}

🔧 Frameworks ที่รองรับ:
{chr(10).join(['  ' + f for f in frameworks_status])}

💡 วิธีแก้ไข:
  1. Train model ใหม่: python train_improved_model.py
  2. หรือระบุชื่อ model ที่มีอยู่ในโฟลเดอร์ models/
  3. ติดตั้ง framework ที่ต้องการ (ดูด้านบน)
  4. ตรวจสอบว่าไฟล์ไม่เสียหาย
        """)
        return None
    
    # Load label encoder
    label_encoder = None
    # Try different label file extensions
    label_extensions = ['.pkl', '_labels.pkl']
    base_path = loaded_path.rsplit('.', 1)[0]  # Remove extension
    
    for ext in label_extensions:
        label_encoder_path = base_path + ext
        if os.path.exists(label_encoder_path):
            try:
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                st.info(f"📋 โหลด Label Encoder: {len(label_encoder.classes_)} classes")
                break
            except Exception as e:
                st.warning(f"⚠️ ไม่สามารถโหลด label encoder: {e}")
    
    # Create default label encoder if not found
    if label_encoder is None and SKLEARN_AVAILABLE:
        st.info("📋 สร้าง Default Label Encoder (0-9, A-Z)")
        label_encoder = LabelEncoder()
        label_encoder.fit(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    
    return {
        'model': loaded_model,
        'label_encoder': label_encoder,
        'model_name': os.path.basename(loaded_path),
        'model_type': 'cnn',  # For backward compatibility
        'framework': framework  # 'tensorflow', 'pytorch', or 'onnx'
    }


@st.cache_resource
def load_models():
    """
    Load model based on ModelConfig.MODEL_TYPE
    Supports both TensorFlow (image-based CNN) and PyTorch (landmark-based)
    
    Returns dict with appropriate structure:
    - PyTorch: {'model_type': 'pytorch_landmark', 'predictor': Predictor, 'label_encoder': LabelEncoder}
    - TensorFlow: {'model_type': 'cnn', 'cnn_model': model, 'label_encoder': LabelEncoder, 'framework': str}
    """
    try:
        model_type = ModelConfig.MODEL_TYPE.lower()
    except AttributeError as e:
        st.error(f"⚠️ Configuration error: {e}")
        st.info("💡 Make sure config.py has ModelConfig class with MODEL_TYPE attribute")
        # Try to reimport config
        import importlib
        importlib.reload(config)
        model_type = config.ModelConfig.MODEL_TYPE.lower()
    
    if model_type == 'pytorch':
        st.info("🔧 Loading PyTorch landmark-based model...")
        
        # Load PyTorch landmark model using Predictor
        predictor = load_pytorch_landmark_model(
            ModelConfig.PYTORCH_MODEL_PATH,
            ModelConfig.PYTORCH_LABEL_ENCODER_PATH,
            device=ModelConfig.PYTORCH_DEVICE
        )
        
        if predictor is None:
            st.error("❌ Failed to load PyTorch model")
            return None
        
        st.success(f"✅ PyTorch model loaded successfully ({len(predictor.label_encoder.classes_)} classes)")
        
        return {
            'model_type': 'pytorch_landmark',
            'predictor': predictor,
            'label_encoder': predictor.label_encoder,
            'framework': 'pytorch',
            'ml_model': None  # For backward compatibility
        }
    
    elif model_type == 'tensorflow':
        st.info("🔧 Loading TensorFlow CNN model...")
        
        # Load TensorFlow CNN model using existing load_model()
        result = load_model(ModelConfig.TF_MODEL_PATH)
        
        if result is None:
            st.error("❌ Failed to load TensorFlow model")
            return None
        
        st.success(f"✅ TensorFlow model loaded successfully")
        
        # Convert to legacy format for backward compatibility
        return {
            'cnn_model': result['model'],
            'label_encoder': result['label_encoder'],
            'model_type': 'cnn',
            'ml_model': None,  # No longer supported
            'framework': result.get('framework', 'tensorflow')
        }
    
    else:
        st.error(f"❌ Unknown MODEL_TYPE: {ModelConfig.MODEL_TYPE}")
        st.info("💡 Set ModelConfig.MODEL_TYPE to 'tensorflow' or 'pytorch' in config.py")
        return None


def predict_with_model(model_data, input_data):
    """
    Universal prediction function that works with any framework
    
    Args:
        model_data: Dict returned from load_model() containing 'model' and 'framework'
        input_data: Preprocessed input (numpy array, shape depends on model)
    
    Returns:
        numpy array of predictions (probabilities for each class)
    """
    if model_data is None:
        raise ValueError("Model data is None. Please load a model first.")
    
    model = model_data['model']
    framework = model_data.get('framework', 'tensorflow')
    
    try:
        if framework == 'tensorflow':
            # TensorFlow/Keras prediction
            predictions = model.predict(input_data, verbose=0)
            return predictions
            
        elif framework == 'pytorch':
            # PyTorch prediction
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            # Convert numpy to torch tensor
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            else:
                input_tensor = input_data
            
            # Move to same device as model
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                
                # Convert to numpy
                if isinstance(output, torch.Tensor):
                    predictions = output.cpu().numpy()
                else:
                    predictions = output
            
            return predictions
            
        elif framework == 'onnx':
            # ONNX prediction
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not available")
            
            # Get input name
            input_name = model.get_inputs()[0].name
            
            # Prepare input
            if isinstance(input_data, np.ndarray):
                input_data = input_data.astype(np.float32)
            
            # Run inference
            outputs = model.run(None, {input_name: input_data})
            predictions = outputs[0]
            
            return predictions
            
        else:
            raise ValueError(f"Unknown framework: {framework}")
            
    except Exception as e:
        st.error(f"❌ Prediction error ({framework}): {e}")
        raise


def get_model_input_shape(model_data):
    """
    Get expected input shape for a model
    
    Args:
        model_data: Dict returned from load_model()
    
    Returns:
        Tuple of input shape (e.g., (224, 224, 3))
    """
    if model_data is None:
        return None
    
    model = model_data['model']
    framework = model_data.get('framework', 'tensorflow')
    
    try:
        if framework == 'tensorflow':
            # TensorFlow/Keras
            return model.input_shape[1:]  # Skip batch dimension
            
        elif framework == 'pytorch':
            # PyTorch - try to get from first layer
            if hasattr(model, 'input_shape'):
                return model.input_shape
            # Default for image models
            return (224, 224, 3)
            
        elif framework == 'onnx':
            # ONNX
            input_shape = model.get_inputs()[0].shape
            # Remove batch dimension and convert -1 to actual size
            shape = tuple([s if isinstance(s, int) and s > 0 else 224 for s in input_shape[1:]])
            return shape
            
    except Exception as e:
        st.warning(f"⚠️ Could not determine input shape: {e}")
        return (224, 224, 3)  # Default
    
    return (224, 224, 3)  # Default

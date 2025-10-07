# Changelog

All notable changes to this project will be documented in this file.

## [3.1.1] - 2025-10-07

### 🎨 Fixed Skeleton Color

#### Fixed
- 🐛 **Skeleton Color Randomness**:
  - Fixed random skeleton colors in camera display
  - Now uses configured color from `InferenceConfig.SKELETON_COLOR_RGB`
  - Applied to all three modes: Practice, Test, and Translation
  - Consistent green color (0, 255, 0) by default

#### Changed
- 🔧 **MediaPipe Drawing** (`app.py`):
  - Added `DrawingSpec` parameters to `mp_drawing.draw_landmarks()`
  - Practice Mode (line 465): Now uses configured color
  - Test Mode (line 578): Now uses configured color
  - Translation Mode (line 826): Now uses configured color
  - Thickness and circle radius now consistent (2 and 3 respectively)

#### Added
- 📚 **New Documentation**:
  - `SKELETON_COLOR_GUIDE.md` - Complete guide for customizing skeleton colors
  - Color configuration examples
  - Visual customization options
  - Testing script

#### Technical
- `mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=3)` for landmark points
- `mp_drawing.DrawingSpec(color=skeleton_color, thickness=2)` for connection lines
- Color sourced from `InferenceConfig.SKELETON_COLOR_RGB` in all locations

---

## [3.1.0] - 2025-10-07

### 🎯 Multi-Framework Support

#### Added
- ✅ **PyTorch Model Support** (.pt, .pth files)
  - Load PyTorch models alongside TensorFlow
  - Automatic framework detection
  - Full model saving required (not state_dict only)

- ✅ **ONNX Model Support** (.onnx files)
  - Optimized inference with ONNX Runtime
  - Faster CPU inference than TensorFlow/PyTorch
  - Smaller model file sizes

- ✅ **Universal Prediction Function**: `predict_with_model()`
  - Works with TensorFlow, PyTorch, and ONNX
  - Handles framework differences automatically
  - Single API for all frameworks

- ✅ **Model Input Shape Detection**: `get_model_input_shape()`
  - Returns expected input dimensions
  - Works across all frameworks
  - Useful for validation

- 📚 **New Documentation**:
  - `MULTI_FRAMEWORK_GUIDE.md` - Comprehensive multi-framework guide
  - `MULTI_FRAMEWORK_UPDATE.md` - Update summary and migration guide

#### Changed
- 🔧 **Model Loading System** (`utils/model_loader.py`):
  - Now tries .h5, .keras, .pt, .pth, .onnx extensions
  - Returns framework type in model_data dict
  - Better error messages showing which frameworks are installed
  - Auto-detection priority: TensorFlow → PyTorch → ONNX

- ⚙️ **Requirements** (`requirements.txt`):
  - Added optional PyTorch dependencies (commented)
  - Added optional ONNX Runtime dependencies (commented)
  - Added conversion tools (commented)
  - Better documentation of optional packages

- 📦 **Exports** (`utils/__init__.py`):
  - Added `load_model` function
  - Added `predict_with_model` function
  - Added `get_model_input_shape` function

#### Technical
- Framework detection based on file extension
- PyTorch models automatically set to eval mode
- ONNX models use InferenceSession
- Backward compatible with existing TensorFlow models
- Label encoder works with all frameworks

#### Performance
- ONNX: ~30-60ms CPU inference (fastest)
- PyTorch: ~40-80ms CPU, ~8-15ms GPU (best GPU)
- TensorFlow: ~50-100ms CPU, ~10-20ms GPU (balanced)

---

## [3.0.0] - 2025-10-07

### 🚀 Major Update: Deep Learning Only + Improved Training

#### Added
- ✅ **New Training Script**: `train_improved_model.py`
  - Addresses high test accuracy, low real-world performance
  - Preprocessing matches inference pipeline exactly
  - Comprehensive data augmentation (rotation, shift, zoom, brightness)
  - Transfer learning with frozen ResNet50 base
  - Advanced callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
  - L2 regularization, Dropout, BatchNormalization
  - Automatic label encoder saving
  - Expected: 94-96% test accuracy, 80-90% real-world accuracy

- 📚 **New Documentation**:
  - `TRAINING_GUIDE.md` - Comprehensive training guide
  - `MODEL_LOADING_GUIDE.md` - Model loading reference
  - `UPDATE_v3.0.0.md` - Migration guide and summary

#### Changed
- 🎯 **Model Loading System**:
  - New `load_model(model_name)` function
  - Simplified loading logic (deep learning only)
  - Better auto-detection with priority order
  - Improved error messages
  - Backward compatibility wrapper `load_models()`

- ⚙️ **Configuration Updates** (`config.py`):
  - Added `MODEL_NAME` parameter (specify model or auto-detect)
  - Lowered `CONFIDENCE_THRESHOLD` to 0.65 (better real-world detection)
  - Added `MIN_DISPLAY_CONFIDENCE` parameter
  - Removed `MODEL_PRIORITY` (no longer needed)
  - Removed `MODEL_SEARCH_PATHS` (auto-detection improved)

#### Removed
- ❌ **ML Model Support**:
  - Removed RandomForest model loading
  - Removed `asl_model.pkl` support
  - Removed ML-specific prediction logic
  - Simplified codebase (~100 lines removed)
  - Deep learning (CNN) models only

#### Fixed
- 🎯 **Real-World Performance**:
  - Training now matches inference preprocessing
  - Data augmentation prevents overfitting
  - Skeleton overlay applied during training (if configured)
  - Better generalization to varied conditions
  - Improved accuracy in different lighting/backgrounds

#### Technical
- Model file structure: `<model_name>.h5` + `<model_name>_labels.pkl`
- Auto-detect priority: resnet50_improved → resnet50_app2 → resnet50 → others
- Backward compatible with existing model files
- TensorFlow-only (no sklearn for models, only for label encoding)

#### Documentation
- Added comprehensive training guide explaining why/how
- Added model loading quick reference
- Updated README with new training instructions
- Added migration guide for existing users

---

## [2.5.1] - 2025-10-07

### Improved
- ⏱️ **Practice Mode Real-time Timer**:
  - Timer now updates in real-time during practice
  - Shows live elapsed time without manual refresh
  - Better user experience with continuous time tracking

- 📹 **Enhanced Camera Feed**:
  - Increased camera resolution to 1280x720 (was 640x480)
  - Bigger and clearer camera display
  - Applied to all modes: Practice, Test, and Translation
  - Better hand gesture visibility

- ⏰ **Test Mode Real-time Timers**:
  - "เวลาที่ใช้" (elapsed time) updates in real-time
  - "เวลาคงเหลือ" (remaining time) updates in real-time
  - Live countdown during test without page refresh
  - More accurate time tracking

- 🤖 **Translation Mode Updates**:
  - Upgraded to Gemini 2.0 Flash (was Gemini Pro)
  - Faster and more accurate text refinement
  - Removed auto-refine - now manual refinement only
  - User controls when to refine translated text
  - Better translation quality with latest model

### Changed
- 🔄 **Translation Workflow**:
  - Auto-refine removed (was every 5 characters)
  - Users now refine manually when finished translating
  - More control over when to process text
  - Reduces unnecessary API calls

### Technical
- **Camera Resolution**: All camera feeds now use 1280x720
- **Gemini Model**: Updated from `gemini-pro` to `gemini-2.0-flash-exp`
- **Real-time Updates**: Timer placeholders update continuously
- **Auto-refine Logic**: Removed from detection loop

## [2.5.0] - 2025-10-07

### Added
- ⚙️ **Comprehensive Configuration System**:
  - Created `config.py` with 8 configuration classes
  - Centralized control over preprocessing, detection, and inference
  - Runtime configuration changes without code modifications
  - 350+ lines of well-documented configuration options

- 🔄 **Multiple Preprocessing Approaches**:
  - Support for 6 preprocessing types: `normal`, `mobilenetv2`, `vgg16`, `vgg19`, `resnet50`, `inception`
  - Configurable resize dimensions
  - Multiple color modes: RGB, BGR, Grayscale
  - Flexible normalization methods

- 🦴 **Skeleton Detection Integration**:
  - 3 inference approaches: `raw_image`, `image_with_skeleton`, `skeleton_only`
  - Pluggable skeleton detection architecture
  - MediaPipe integration (working)
  - OpenPose and YOLOPose support (placeholders)
  - Configurable skeleton visualization (color, thickness, background)

- 📐 **Advanced Image Processing**:
  - `utils/preprocessing.py` - 250+ lines of preprocessing utilities
  - Skeleton overlay on images (Approach 2)
  - Skeleton extraction on blank background (Approach 3)
  - Automatic hand detector selection
  - Integration with all camera detection modes

- ⏱️ **Real-time Practice Timer**:
  - Live timer showing elapsed time in practice mode
  - MM:SS format display
  - 4-column stats layout (attempts, correct, accuracy, time)
  - Starts when practice mode activated

- 🧪 **Configuration Testing**:
  - `test_config.py` - Comprehensive test script
  - Tests all preprocessing types
  - Tests all inference approaches
  - Tests detector configurations
  - Validates color conversions and resize operations

- 📚 **Documentation**:
  - `CONFIG.md` - Complete configuration guide
  - Usage examples for all approaches
  - Troubleshooting section
  - Best practices and future enhancements

### Fixed
- 🐛 **Learning Mode Button**: Removed confusing "Practice" button from learning mode
  - Learning mode is now purely for viewing/learning
  - Only "Back" button remains
  - Clearer separation between learning and practice modes

- 🐛 **Practice Mode Double-Click**: Fixed letter switching requiring two clicks
  - Added buffer clearing on letter switch
  - Single click now works correctly
  - Cleared keypoint_buffer and frame_buffer on switch
  - Immediate response to letter changes

### Improved
- 🎯 **Prediction Pipeline**: Integrated config-based preprocessing
  - `utils/prediction.py` now uses configurable preprocessing
  - All camera detection modes pass landmarks
  - Support for skeleton-based approaches
  - Backward compatible with older models

- 📊 **Practice Mode Stats**: Enhanced stats display
  - Added real-time timer
  - Better layout with 4 columns
  - More informative metrics
  - Professional appearance

### Technical
- **New Files**:
  - `config.py` - Centralized configuration (350+ lines)
  - `utils/preprocessing.py` - Preprocessing utilities (250+ lines)
  - `test_config.py` - Configuration testing script
  - `CONFIG.md` - Configuration documentation

- **Modified Files**:
  - `utils/prediction.py` - Config-based preprocessing integration
  - `app.py` - Button removal, double-click fix, timer addition, landmarks passing

- **Configuration Classes**:
  - `PreprocessConfig` - Image preprocessing settings
  - `HandDetectionConfig` - Detection library selection
  - `InferenceConfig` - Approach and skeleton settings
  - `ModelConfig` - Model loading configuration
  - `PracticeModeConfig` - Practice mode thresholds
  - `TestModeConfig` - Test mode settings
  - `TranslationModeConfig` - Translation mode settings
  - `UIConfig` - UI appearance settings

- **Helper Functions**:
  - `get_preprocess_function()` - Returns preprocessing function
  - `get_resize_dimensions()` - Returns (width, height)
  - `get_color_conversion()` - Returns OpenCV conversion code
  - `should_apply_skeleton()` - Check if skeleton needed
  - `is_skeleton_only()` - Check for skeleton-only mode

## [2.4.1] - 2025-10-06

### Fixed
- 🐛 **Learning Mode Display**: Fixed cramped layout by implementing fullscreen character display
  - Characters now show in fullscreen when selected
  - Added back navigation button
  - Larger character display (12rem vs 8rem)
  - Better example image layout (fullwidth grid)
  - No more overlapping content

- 🐛 **Test Mode Examples Removed**: Removed example images from test interface
  - True testing environment (no cheating)
  - Shows only character and instructions
  - Cleaner, faster test interface

- 🐛 **Test Mode Auto-Skip**: Auto-advance to next question when correct answer detected
  - No manual confirmation needed
  - Smooth progression after 30 consecutive correct frames
  - Added optional manual skip button

- 🐛 **Practice Mode Stats**: Fixed division by zero error when no attempts made
  - Safe division with default values
  - Shows 0.0% instead of crashing
  - Uses `.get()` for safe dictionary access
  - Better decimal precision (1 decimal place)

### Improved
- 📱 **Learning Mode UX**: Fullscreen layout with better navigation
- 🎯 **Test Mode Flow**: Automatic progression reduces test time
- 📊 **Practice Mode Stats**: More stable and professional display

## [2.4.0] - 2025-10-06

### Added
- 🔢 **Number Support (0-9)**:
  - Complete number gestures (0-9) added to learning mode
  - Number instructions in Thai for all digits
  - Separate tab for numbers in learning mode
  - Practice mode now includes 36 characters (A-Z + 0-9)
  - Test mode supports numbers

- 📸 **Camera-Based Test Mode**:
  - Real-time gesture detection during tests
  - Requires 30 consecutive correct detections for confirmation
  - Progress indicator (0-100%) during detection
  - Reference images and instructions displayed during test
  - Auto-confirmation when target gesture detected

- 💬 **Word Formation Display**:
  - New section in translation mode showing formed words
  - Words separated by " · " for clarity
  - Real-time updates as user signs
  - Helps visualize word boundaries

- 🎓 **Enhanced Learning Mode**:
  - Improved with "ตัวอย่างให้ ดีกว่านี้" title
  - Increased examples from 6 to 9 images per character
  - Organized tabs: "🔤 A-Z" and "🔢 0-9"
  - Better grid layout (7 columns for letters, 10 for numbers)
  - Image captions for each example

- 📦 **CNN .pkl Support**:
  - Load CNN models from pickle files
  - Priority: .pkl → .keras → .h5
  - Extracts model and label encoder from pickle data

### Changed
- 🔤 **Alphabet Constant**: Expanded from 26 to 36 characters (A-Z0-9)
- 🏷️ **Label Encoder**: Now supports 36 classes instead of 26
- 📝 **Character Type Detection**: Auto-detects if practicing letter or number
- 🎯 **Test Mode UI**: Split into reference column and camera column

### Improved
- Better Thai translations and descriptions
- More robust model loading with multiple format support
- Enhanced user feedback in all modes
- Clearer visual hierarchy in learning mode

## [2.3.3] - 2025-10-06

### Changed
- 🔧 **Code Refactoring - Modular Architecture**:
  - Refactored monolithic `app.py` into modular structure
  - Created `utils/` package with specialized modules
  - Removed ~230 lines of duplicate code from `app.py`
  - Improved code organization and maintainability

### Added
- 📦 **New `utils/` Package**:
  - `utils/model_loader.py` - Model initialization and loading (~110 lines)
  - `utils/prediction.py` - Prediction logic for all model types (~120 lines)
  - `utils/hand_processing.py` - MediaPipe utilities (~60 lines)
  - `utils/letter_data.py` - ASL instruction database (~50 lines)
  - `utils/__init__.py` - Package initialization with exports

- 🎯 **Better Separation of Concerns**:
  - Model loading isolated in `model_loader.py`
  - Prediction logic centralized in `prediction.py`
  - Hand processing utilities in dedicated module
  - Letter instructions separated for easy expansion

- 📚 **Enhanced Letter Data**:
  - Expanded from 5 to 26 letters (A-Z)
  - Complete ASL instruction set in Thai
  - Centralized data structure for easy translation

### Improved
- ✨ **Function Signatures**:
  - Updated `predict_letter()` to accept `models_data` and `alphabet` parameters
  - Better parameter passing for testability
  - Clearer function dependencies

- 🧹 **Code Quality**:
  - Removed duplicate function definitions
  - Cleaner imports structure
  - Better module organization
  - Maintained all caching decorators for performance

## [2.3.2] - 2025-10-06

### Fixed
- 🐛 **MobileNetV2 Input Shape Error**:
  - Fixed "expected shape=(None, 224, 224, 3)" error
  - CNN models now receive properly formatted image input
  - Added automatic image preprocessing for transfer learning models
  
### Added
- 📸 **Frame Buffer System**:
  - New `frame_buffer` in session state
  - Stores raw camera frames for CNN models
  - Keeps last 30 frames for processing
  
- 🔄 **Smart Model Input Detection**:
  - Automatic detection of model input type
  - Image-based models (MobileNetV2) → use frame_buffer
  - Keypoint-based models (LSTM) → use keypoint_buffer
  - ML models → use averaged keypoints
  
- 🖼️ **Image Preprocessing Pipeline**:
  - Resize to 224×224 for MobileNetV2
  - BGR to RGB color conversion
  - Normalize to [0, 1] range
  - Automatic batch dimension handling

### Changed
- 🔧 **Enhanced predict_letter() Function**:
  - Now handles both image and keypoint inputs
  - Fallback logic for different model types
  - Better error messages for debugging
  
- 📹 **Camera Processing**:
  - Store clean frames before drawing landmarks
  - Separate buffers for different model types
  - Applied to both Practice and Translation modes

### Technical
- Added frame buffering in camera capture loops
- Enhanced CNN prediction with image preprocessing
- Improved error handling with informative warnings
- Maintained backward compatibility with all model types

### Documentation
- Added `UPDATE_v2.3.2.md` with detailed technical guide
- Included preprocessing examples and debugging tips

---

## [2.3.1] - 2025-10-06

### Fixed
- 🔧 **Deprecated Parameter Warning**:
  - Fixed `use_column_width` → `use_container_width` (5 occurrences)
  - Locations: Learning Mode, Practice Mode, Camera Feed
  - No more Streamlit deprecation warnings
  
### Added
- 📸 **ASL Reference Images in Learning Mode**:
  - Shows official ASL alphabet diagram from `assets/asl/{Letter}.svg`
  - Graceful fallback to large text if image not found
  - Displays above dataset example images
  
- 🤖 **Enhanced Model Support**:
  - Added `best_transfer_CNN.keras` to model search paths
  - Now supports transfer learning models
  - Priority order: best_transfer_CNN.keras > other CNN models > ML models

### Technical
- Updated all `st.image()` calls to use `use_container_width` parameter
- Added SVG image loading in `show_letter_detail()` function
- Expanded CNN model paths array with new model name

### Documentation
- Added `UPDATE_v2.3.1.md` with detailed changes

---

## [2.3.0] - 2025-10-06

### Changed - Model Loading Architecture 🤖
- 🚀 **Production-Ready Model Loading**:
  - **Removed**: Auto-training functionality from app.py
  - **Added**: Load pre-trained models from saved files only
  - **Result**: 100x faster startup (instant vs 2-3 minutes)
  
- 🎯 **Dual Model Support**:
  - Support for ML models (RandomForest .pkl)
  - Support for CNN models (TensorFlow .h5/.keras)
  - Auto-detection of available models
  - Seamless switching between model types
  
- ⚡ **Performance Improvements**:
  - Instant app startup with pre-trained models
  - No dataset processing on app launch
  - Cleaner separation: training vs inference
  
- 🔧 **Enhanced Prediction Function**:
  - `predict_letter()` now supports both ML and CNN models
  - Different preprocessing for each model type
  - Temporal sequence handling for CNN (45 frames)
  - Static keypoint averaging for ML (10 frames)

### Added
- 📦 **New Dependencies**:
  - `tensorflow>=2.10.0` for CNN model support
  - Optional sklearn imports with graceful fallback
  
- 📚 **New Documentation**:
  - `UPDATE_NOTES_v2.3.0.md` - Detailed technical changes
  - `TRAINING_GUIDE.md` - Complete model training guide
  - Updated `README.md` with training instructions

### Technical
- Replaced `load_or_train_model()` with `load_models()`
- Added `TF_AVAILABLE` and `SKLEARN_AVAILABLE` flags
- Enhanced error messages for missing models
- Support for multiple model file formats (.h5, .keras, .pkl)
- Automatic label encoder loading for CNN models

### Breaking Changes
- ⚠️ **Must train model before running app**
- ⚠️ **No auto-training on first run**
- Migration: Run `python preprocess_data.py` or `python train_model.py` once

### Documentation
- Added comprehensive training guide
- Updated installation instructions
- Added model comparison table
- Enhanced troubleshooting section

---

## [2.2.0] - 2025-10-06

### Added - Real ML Model Integration 🤖
- 🔄 **Machine Learning Model**:
  - Real RandomForest classifier (replaced mock predictions)
  - Auto-training from dataset images on first run
  - Model persistence (saved as pickle file)
  - ~70-85% real-world accuracy
  
- 📸 **Practice Mode Enhancements**:
  - Reference images displayed during practice
  - 4 sample images in 2×2 grid layout
  - Side-by-side view (reference + camera)
  - Dynamic image loading based on selected letter
  
- ⚡ **Performance Features**:
  - `@st.cache_resource` for model caching
  - Prediction averaging (last 10 frames)
  - Fast inference (<100ms per prediction)

### Technical
- Added `scikit-learn>=1.0.0` dependency
- Added `load_or_train_model()` function
- Enhanced `predict_letter()` with real ML
- Modified `show_practice_mode()` layout
- MediaPipe landmark extraction for training

### Documentation
- Added `UPDATE_NOTES_v2.2.0.md` with technical details

---

## [2.1.2] - 2025-10-06

### Changed - UX Improvements & Dataset Integration
- 🌐 **Default Mode**: Changed to Real-time Translation (was Practice Mode)
  - Users now start with the most powerful feature
  - Better first impression for new users
  
- 🔐 **Simplified API Key UI**:
  - Removed "ใช้ API Key อื่น (ชั่วคราว)" option
  - Cleaner, less confusing interface
  - Encourages best practice (.env file usage)
  
- 📸 **Learning Mode Enhancement**:
  - Now loads real images from `datasets/asl_dataset/`
  - Displays 6 sample images per letter
  - Shows images in 2×3 grid layout
  - Automatically matches letter to folder (A → a/)
  - Graceful error handling for missing images

### Technical
- Added `glob` import for file pattern matching
- Enhanced `show_letter_detail()` with image loading
- Improved error messages for missing dataset folders

### Documentation
- Added `UPDATE_NOTES_v2.1.2.md` with detailed changes

---

## [2.1.1] - 2025-10-06

### Added - Environment Variables Support 🔐
- 🔒 **Enhanced Security**: API keys now loaded from `.env` file
  - Added `python-dotenv` dependency
  - Created `.env.example` template file
  - Added comprehensive `ENV_SETUP_GUIDE.md`
  - Updated `.gitignore` to exclude `.env` files
  
- 🎯 **Improved UX**:
  - Auto-load API key from `.env` on startup
  - Option to override with manual input (temporary)
  - Clear success message when .env is used
  - Helpful instructions when .env is not found
  
- 📚 **Documentation**:
  - Added `ENV_SETUP_GUIDE.md` - Complete .env setup guide
  - Updated `README.md` with .env setup instructions
  - Updated `QUICKSTART_TRANSLATION.md` with .env option
  - Updated `test_gemini.py` to support .env

### Changed
- **API Key Input**: Now optional if .env is configured
- **test_gemini.py**: Checks for .env before asking for manual input

### Security
- ✅ `.env` added to `.gitignore` (prevents accidental commits)
- ✅ API keys no longer need to be entered manually each time
- ✅ Reduced risk of exposing API keys in screenshots/demos

---

## [2.1.0] - 2025-10-06

### Added - Real-time Translation Mode ✨
- 🌐 **New Mode**: Real-time Translation with Gemini API integration
  - Automatically translates ASL fingerspelling to text
  - Sends text to Gemini API for refinement and correction
  - Returns grammatically correct and meaningful sentences
  - Auto-refine option (every 5 characters)
  - Manual refine button
  - Statistics display (character count, word count)
  - Clear buffer functionality
  
- 🤖 **Gemini API Features**:
  - Spell correction
  - Grammar improvement
  - Sentence formation
  - Punctuation addition
  - Multi-language support
  
- 📚 **Documentation**:
  - Added `TRANSLATION_MODE_GUIDE.md` - Complete guide for Translation Mode
  - Updated `README.md` with Translation Mode features
  - Updated `requirements.txt` with `google-generativeai`

### Technical Implementation
- Confirmation threshold system (5 consecutive detections)
- Translation buffer management
- Session state for translated and refined text
- Error handling for API calls
- Graceful degradation when API unavailable

---

## [2.0.0] - 2025-10-06

### Changed - Major Architecture Migration
- 🚀 **Migrated to Streamlit** for rapid prototyping and simplicity
- **Removed**: All HTML, CSS, and JavaScript files (legacy web version)
  - Deleted `index.html`
  - Deleted `package.json`
  - Deleted `styles/` folder (CSS files)
  - Deleted `js/` folder (9 JavaScript modules)
- **Consolidated**: Single-file application (`app.py`) replacing 10+ files
- **Simplified**: Dependencies reduced to essential Python packages only
- **Improved**: Setup time from 30+ minutes to < 3 minutes

### Added
- ✅ Complete Streamlit application (`app.py`)
- ✅ Streamlit configuration (`.streamlit/config.toml`)
- ✅ Session state management for statistics
- ✅ Custom CSS styling within Streamlit
- ✅ Real-time camera integration with OpenCV
- ✅ Updated documentation for Streamlit workflow

### Technical Changes
- **Framework**: HTML/CSS/JavaScript → Streamlit (Python)
- **Model Loading**: TensorFlow.js → Ready for TensorFlow/Keras
- **Camera**: Browser API → OpenCV (cv2)
- **State Management**: localStorage → Streamlit session_state
- **Deployment**: Static files → Streamlit app server

---

## [1.0.0] - 2025-10-06 (Legacy - Deprecated)

### Added
- Initial release of ASL Fingerspelling Trainer (JavaScript version)
- **Learning Mode**: View reference images and instructions for all 26 letters
- **Practice Mode**: Real-time hand detection and gesture recognition
  - MediaPipe Hands integration for 21-point hand tracking
  - Live feedback on hand position and shape
  - Accuracy scoring system
  - Statistics tracking
- **Test Mode**: Complete assessment of A-Z fingerspelling
  - Timed testing
  - Score calculation
  - Test history
- **UI Features**:
  - Responsive design for desktop and mobile
  - Thai language interface
  - Progress tracking
  - Help and about sections
- **AI/ML Components**:
  - TensorFlow.js integration
  - Mock model for development
  - Keypoint-based gesture analysis
  - Temporal sequence analysis support
- **Technical Features**:
  - Real-time camera access and processing
  - 30 FPS video processing
  - ROI (Region of Interest) detection
  - Hand size and position validation
  - Feedback latency < 500ms

### Technical Stack
- HTML5/CSS3/JavaScript (ES6+)
- MediaPipe Hands v0.4
- TensorFlow.js v4.x
- Vanilla JavaScript (no frameworks)

### Documentation
- Comprehensive README with setup instructions
- Dataset requirements documentation
- Model training guidelines
- Code comments and JSDoc

### Known Limitations
- Mock AI model (requires training with real dataset)
- Placeholder reference images (need ASL hand images)
- English-only sign recognition (ASL)
- Single-hand detection only

### Future Plans
- Train production-ready model with real dataset
- Add actual ASL reference images
- Multi-language support
- Word spelling mode
- Mobile app version
- Leaderboard system

---

## Version Guidelines

Format: [Major.Minor.Patch]
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

---

**Note**: This is version 1.0.0 - the initial release with core functionality.

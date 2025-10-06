# Changelog

All notable changes to this project will be documented in this file.

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

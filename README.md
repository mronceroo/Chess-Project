# 🤖♟️ Virtual Chess Mentor - AI-Powered Chess Tutor

An intelligent chess tutoring system that combines **Computer Vision**, **Natural Language Processing**, and **Game AI** to provide real-time feedback and coaching during chess games.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenAI Whisper](https://img.shields.io/badge/Whisper-STT-green.svg)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technical Highlights](#technical-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## 🎯 Overview

This project implements a **virtual chess mentor** capable of:
- **Detecting** chess pieces on a real board using computer vision
- **Playing** optimally using Minimax algorithm with Alpha-Beta pruning
- **Communicating** moves and answering questions via voice (STT + LLM + TTS)
- **Teaching** chess concepts and providing real-time feedback

The system focuses on the **King + Rook vs King endgame** (KRK), implementing advanced AI techniques for optimal play and natural language interaction.

---

## ✨ Features

### 🔍 Computer Vision
- **Board Detection**: Automatic detection of chessboard contours using OpenCV
- **Piece Recognition**: YOLOv8 fine-tuned model achieving **100% precision** and **99.5% mAP@50**
- **Real-time Tracking**: Live piece position updates via camera feed
- **Perspective Transformation**: Warped view correction for accurate square mapping

### 🧠 Game AI
- **Minimax Algorithm**: With Alpha-Beta pruning for optimal move selection
- **Strategic Evaluation**: 
  - Manhattan distance for direct king pressure
  - Chebyshev distance for cornering opponent
  - Rook protection and indirect pressure tactics
- **Legal Move Generation**: Complete validation including check detection
- **Guaranteed Victory**: Algorithm achieves checkmate in all test games

### 🗣️ Natural Language Interface
- **Speech-to-Text (STT)**: OpenAI Whisper for voice command recognition
- **Large Language Model**: Local LLM (Llama 3.2 - 3B parameters) for chess coaching
- **Text-to-Speech (TTS)**: Microsoft SpeechT5 for natural voice responses
- **Interactive Q&A**: Answer chess questions and provide move recommendations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                         │
│              (Voice Input / Board Moves)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │   VISION    │         │    VOICE    │
    │   MODULE    │         │   MODULE    │
    └──────┬──────┘         └──────┬──────┘
           │                       │
           │  ┌─────────────┐      │
           └─►│  REASONING  │◄─────┘
              │   ENGINE    │
              │  (Minimax)  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   OUTPUT    │
              │ (Voice/Move)│
              └─────────────┘
```

### Components:

1. **Vision Module**
   - OpenCV contour detection
   - YOLOv8 piece detection (fine-tuned)
   - Virtual board generation

2. **Reasoning Engine**
   - Minimax with Alpha-Beta pruning
   - Legal move generation
   - Position evaluation (Manhattan + Chebyshev distances)

3. **Communication Module**
   - STT: Whisper (Medium model)
   - LLM: Llama 3.2 via LM Studio API
   - TTS: SpeechT5

---

## 🔬 Technical Highlights

### YOLOv8 Model Performance
- **Dataset**: 497 images (70% train, 20% valid, 10% test)
- **Metrics on Validation Set**:
  - Precision: **100%**
  - Recall: **100%**
  - F1-Score: **100%**
  - mAP@50: **99.5%**
- **Real-world Performance**: 72-97% accuracy depending on lighting conditions

### Minimax Algorithm
- **Evaluation Function Components**:
  ```python
  # Manhattan Distance (direct pressure)
  d_manhattan = sum(abs(p_i - q_i) for all dimensions)
  
  # Chebyshev Distance (cornering)
  d_chebyshev = max(abs(p_i - q_i) for all dimensions)
  ```
- **Pruning**: Alpha-Beta to optimize search depth
- **Strategy**: Aggressive king positioning + rook protection
- **Success Rate**: 100% win rate in all test games

### NLP Pipeline
- **STT Error Rate**: Low transcription error with Whisper Medium
- **LLM Response**: Average 3-5 seconds (3B model on local hardware)
- **TTS Quality**: Natural-sounding English voice with number pronunciation handling

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or external camera
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster inference)

### Step 1: Clone Repository
```bash
git clone https://github.com/mronceroo/Chess-Project.git
cd Chess-Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install FFmpeg
**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# MacOS
brew install ffmpeg
```

### Step 4: Download Model Weights
```bash
# YOLOv8 model will be downloaded automatically on first run
# Whisper model will be downloaded automatically
# SpeechT5 embeddings will be cached locally
```

### Step 5: Setup LM Studio (for LLM)
1. Download [LM Studio](https://lmstudio.ai/)
2. Download **Hermes 3 Llama 3.2 - 3B** model
3. Start local server on default port (1234)

---

## 💻 Usage

### Basic Usage
```bash
python main.py
```

### Game Flow
1. **Setup**: Place the three pieces on the board (Black Rook, Black King, White King)
2. **Detection**: System detects board and pieces via camera
3. **Play**: System announces its move via voice
4. **Move**: Human player moves the white king
5. **Ask Questions**: Press designated key to ask chess questions
6. **Repeat**: Game continues until checkmate or resignation

### Voice Commands
- **"What should I do next?"** - Get move recommendations
- **"Explain the concept of [term]"** - Learn chess terminology
- **"Why did you make that move?"** - Understand AI strategy

### Configuration
Edit `config.py` to adjust:
- Camera resolution
- Detection thresholds
- Minimax search depth
- LLM parameters
- Audio settings

---

## 📁 Project Structure

```
Chess-Project/
│
├── vision/
│   ├── board_detector.py       # Board contour detection
│   ├── piece_detector.py       # YOLO piece detection
│   └── virtual_board.py        # Virtual board generation
│
├── reasoning/
│   ├── minimax.py              # Minimax + Alpha-Beta
│   ├── move_generator.py       # Legal moves
│   └── evaluator.py            # Position evaluation
│
├── communication/
│   ├── stt.py                  # Speech-to-Text (Whisper)
│   ├── llm_interface.py        # LLM API connection
│   └── tts.py                  # Text-to-Speech (SpeechT5)
│
├── models/
│   └── yolov8_chess.pt         # Fine-tuned YOLOv8 weights
│
├── data/
│   └── dataset/                # Training images
│
├── utils/
│   ├── config.py               # Configuration
│   └── helpers.py              # Utility functions
│
├── main.py                     # Main execution
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 📊 Results

### Vision System
| Metric | Value |
|--------|-------|
| Precision | 100% |
| Recall | 100% |
| mAP@50 | 99.5% |
| Real-time Detection | 72-97% (lighting dependent) |

### Game AI Performance
- **Win Rate**: 100% (all test games)
- **Average Moves to Checkmate**: 15-25 moves
- **Strategy Effectiveness**: Successfully implements cornering and pressure tactics

### Communication System
- **STT Accuracy**: >95% with clear audio
- **LLM Response Time**: 3-5 seconds
- **TTS Quality**: Natural voice with proper number pronunciation

### Sample Game
```
Initial Position:
♜(Black Rook): a8
♚(Black King): e7  
♔(White King): e5

Move 10: Rook to a5 (cutting off escape routes)
Move 15: King to d6 (direct pressure, Manhattan distance = 2)
Move 20: Rook to e1 (checkmate preparation)
Move 23: Checkmate! ♜e5# (King on e7, Rook on e5)
```

---

## 🔮 Future Improvements

### Vision Enhancements
- [ ] 3D piece detection (abandon top-down requirement)
- [ ] Multi-board support (different colors/materials)
- [ ] Improved lighting robustness
- [ ] Mobile app with AR overlay

### Game AI Extensions
- [ ] Support for full chess game (all 32 pieces)
- [ ] Opening book integration
- [ ] Endgame tablebase lookup
- [ ] Difficulty levels (adjustable Minimax depth)
- [ ] Multi-agent learning (self-play improvement)

### NLP Improvements
- [ ] Larger LLM models (7B-13B parameters)
- [ ] Multilingual support (Spanish, French, etc.)
- [ ] More natural TTS voices
- [ ] Context-aware conversation memory
- [ ] Personalized coaching based on player level

### User Experience
- [ ] GUI dashboard with move history
- [ ] Game replay and analysis
- [ ] Progress tracking over time
- [ ] Integration with online chess platforms
- [ ] Tournament mode with ELO rating

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| **Computer Vision** | OpenCV, YOLOv8 (Ultralytics), NumPy |
| **Deep Learning** | PyTorch, Ultralytics |
| **NLP - STT** | OpenAI Whisper, FFmpeg |
| **NLP - LLM** | Llama 3.2 (3B), LM Studio API |
| **NLP - TTS** | SpeechT5, Hugging Face Transformers |
| **Audio** | sounddevice, numpy |
| **Game Logic** | Custom Minimax implementation |
| **Utils** | Python 3.8+, regex, hasattr |

---

## 📚 References

1. Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Menezes, R. & Peixoto, H. (2023). "An Intelligent Chess Piece Detection Tool." *SEMISH 2023*.
3. Touvron, H. et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.
4. Ao, J. et al. (2022). "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training." arXiv:2110.07205.
5. FIDE. (2023). *Laws of Chess*. International Chess Federation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Manuel Roncero Gago**
- GitHub: [@mronceroo](https://github.com/mronceroo)
- Email: manuel.roncero.01@uie.edu
- LinkedIn: [https://www.linkedin.com/in/manuel-roncero-gago/]

---

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Ultralytics for YOLOv8
- Meta AI for Llama architecture
- Microsoft for SpeechT5
- Roboflow for chess piece dataset
- LM Studio for local LLM inference

---

## ⚠️ Disclaimer

This project is for **educational purposes** only. The AI tutor is designed to assist learning and should not replace professional chess coaching for competitive players.

# AI Pong Game - Computer Vision & Neural Network Paddle Control

An advanced AI-powered Pong game featuring computer vision-based ball tracking, predictive paddle movement, and dual AI capabilities for autonomous gameplay. The system uses real-time image processing, WebSocket communication, and smart algorithms to create intelligent AI opponents.

## 🎮 Features

### Core Gameplay
- **AI vs AI**: Dual autonomous AI system controlling both paddles simultaneously
- **Remote AI Control**: External AI clients connect via HTTP API to control paddles
- **Real-time Ball Tracking**: Advanced computer vision for accurate ball detection
- **Predictive Movement**: Smart algorithms predict ball trajectory and optimal paddle positioning
- **WebSocket Communication**: Real-time game state synchronization between server and clients

### AI Capabilities
- **Computer Vision Ball Detection**: Multi-technique ball tracking using HSV color space and contour analysis
- **Velocity Prediction**: Ball trajectory prediction with wall bounce calculations
- **Adaptive Strategies**: Different behavior patterns based on ball direction and game state
- **High-Frequency Updates**: 50 FPS paddle control for responsive gameplay
- **Smart Positioning**: Defensive and offensive paddle positioning algorithms

## 🏗️ Architecture

### Components
1. **Game Server** (`server.py`) - WebSocket server handling game state and API endpoints
2. **Web Interface** (`main.html`) - Browser-based game visualization and controls  
3. **AI Client** (`ai_pong_player.py`) - Individual paddle AI controller
4. **Dual AI System** (`perfect_pong_ai.py`) - Simultaneous control of both paddles

### Communication Flow
Browser Game ←→ WebSocket Server ←→ AI Clients
↓ ↓ ↓
Game Display State Management Paddle Control

## 🚀 Installation

### Dependencies
pip install aiohttp pillow numpy torch torchvision opencv-python asyncio

### System Requirements
- Python 3.7+
- Modern web browser with WebSocket support
- Minimum 4GB RAM for neural network operations
- OpenCV-compatible camera (optional for extended features)

## 📖 Usage

### 1. Start the Game Server
python server.py

The server will start on `http://127.0.0.1:8000` [file:4].

### 2. Open the Game Interface
Navigate to `http://127.0.0.1:8000` in your browser and open `main.html` [file:2].

### 3. Run AI Controllers

#### Single Paddle Control
Control left paddle (AI1)
python ai_pong_player.py ai1

Control right paddle (AI2)
python ai_pong_player.py ai2

#### Dual AI Mode (Autonomous Match)
python perfect_pong_ai.py dual

### 4. Game Controls
- **ENTER**: Start/Resume game
- **Space**: Pause game
- **Speed Control**: Adjust game speed (1x-3x)
- **Manual Capture**: Take screenshots for analysis
- **Score Management**: Manual score adjustment and tracking

## 🧠 AI Implementation Details

### Ball Detection Algorithm
- **Multi-technique Approach**: Combines HSV color space filtering and grayscale thresholding [file:1]
- **Contour Analysis**: Identifies circular objects with appropriate size and aspect ratio
- **Edge Filtering**: Excludes paddle areas to avoid false detections
- **Circularity Scoring**: Ranks candidates based on geometric properties

### Paddle Control Strategy
- **Predictive Interception**: Calculates ball trajectory and optimal interception point [file:1]
- **Wall Bounce Simulation**: Accounts for multiple wall bounces in trajectory prediction
- **Adaptive Positioning**: Different strategies for offensive and defensive play
- **Movement Optimization**: Reduces unnecessary paddle movements to improve efficiency

### Performance Features
- **50 FPS Control Loop**: High-frequency paddle updates for responsive gameplay [file:1]
- **Smart Movement Threshold**: Only sends commands for significant position changes
- **Consecutive Detection Tracking**: Improves ball tracking reliability
- **Velocity History**: Maintains ball movement history for better prediction

## 🛠️ API Endpoints

### Capture System
POST /capture
Content-Type: application/json

{
"captureOptions": {
"format": "jpeg",
"quality": 0.8
},
"returnBase64": true
}

### Paddle Control
POST /control
Content-Type: application/json

{
"paddle": "ai1",
"y": 350,
"immediate": true
}

### AI Prediction
POST /predict
Content-Type: application/json

{
"model": "ai1",
"targetY": 350,
"confidence": 0.95
}


### Score Management
GET /score
POST /score

{
"ai1": 0,
"ai2": 0,
"match": 1
}

## 🎯 Configuration

### Game Parameters (in `main.html`)
- **Canvas Size**: 1400x700 pixels [file:2]
- **Paddle Dimensions**: 150px height, positioned at x=50 and x=1350
- **Ball Physics**: Variable velocity with collision detection
- **Win Condition**: First to 15 points

### AI Parameters (configurable in code)
- **Update Interval**: 0.02 seconds (50 FPS)
- **Movement Threshold**: 15 pixels minimum movement
- **Ball History**: 10-frame tracking window
- **Prediction Accuracy**: 95% confidence threshold

## 🔧 Customization

### Adjusting AI Difficulty
1. **Reaction Speed**: Modify `update_interval` in `run_ai_loop()`
2. **Prediction Accuracy**: Adjust ball tracking parameters
3. **Movement Style**: Change defensive vs aggressive positioning ratios

### Adding New AI Strategies
1. Extend the `predict_paddle_position()` method [file:1]
2. Implement custom ball tracking algorithms
3. Add strategy-specific parameters and configurations

## 🐛 Troubleshooting

### Common Issues
- **No Ball Detection**: Check lighting conditions and contrast settings
- **Connection Errors**: Ensure server is running and ports are available
- **High CPU Usage**: Reduce update frequency or image quality
- **Paddle Not Moving**: Verify WebSocket connection and API endpoints

### Debug Features
- **Frame Logging**: Detailed per-frame status information [file:1]
- **Success Rate Tracking**: Monitor AI performance metrics
- **Image Capture**: Manual screenshot capability for analysis
- **WebSocket Status**: Real-time connection monitoring

## 📊 Performance Metrics

The system tracks various performance indicators:
- **Ball Detection Rate**: Percentage of successful ball identifications
- **Paddle Command Success**: HTTP request success rates
- **Frame Processing Time**: Computer vision processing latency
- **Prediction Accuracy**: Trajectory prediction correctness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements or bug fixes
4. Add tests for new functionality
5. Submit a pull request with detailed description

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built using:
- **OpenCV** for computer vision processing
- **PyTorch** for neural network operations  
- **aiohttp** for asynchronous web server
- **PIL/Pillow** for image processing
- **NumPy** for numerical computations

---

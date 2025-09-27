#!/usr/bin/env python3

"""
AI Pong Client - High-accuracy CNN-LSTM model for Pong paddle control
Optimized for real-time paddle movement with frequent updates
Usage: python ai_pong_player.py [paddle_name]
Dependencies: pip install aiohttp pillow numpy torch torchvision opencv-python
"""

import asyncio
import aiohttp
import json
import base64
import numpy as np
import time
import sys
import cv2
from io import BytesIO
from PIL import Image

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class SmartBallTracker:
    """Advanced ball tracking with improved accuracy"""
    
    def __init__(self):
        self.ball_history = []
        self.velocity_history = []
        self.last_ball_pos = None
        self.ball_velocity = (0, 0)
        self.consecutive_detections = 0
    
    def detect_ball(self, img_array):
        """Enhanced ball detection using multiple techniques"""
        try:
            if len(img_array.shape) == 3:
                # Convert to HSV for better ball detection
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                # Create mask for white objects (ball)
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 30, 255])
                white_mask = cv2.inRange(hsv, lower_white, upper_white)
                
                # Also try grayscale thresholding
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                
                # Combine masks
                combined_mask = cv2.bitwise_or(white_mask, thresh)
            else:
                gray = img_array.copy()
                _, combined_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ball_candidates = []
            h, w = img_array.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 800:  # Ball size range
                    # Get bounding box
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (should be roughly square for ball)
                    aspect_ratio = w_box / h_box if h_box > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:
                        
                        # Calculate center
                        center_x = x + w_box // 2
                        center_y = y + h_box // 2
                        
                        # Avoid edges (likely paddles)
                        edge_margin = 60
                        if (edge_margin < center_x < w - edge_margin and
                            edge_margin < center_y < h - edge_margin):
                            
                            # Calculate circularity
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                ball_candidates.append((center_x, center_y, area, circularity))
            
            if ball_candidates:
                # Choose the best candidate (most circular with good size)
                best_candidate = max(ball_candidates, 
                    key=lambda x: x[3] * min(x[2] / 50.0, 1.0))  # Balance circularity and size
                
                ball_pos = (best_candidate[0], best_candidate[1])
                
                # Update velocity if we have previous position
                if self.last_ball_pos:
                    self.ball_velocity = (
                        ball_pos[0] - self.last_ball_pos[0],
                        ball_pos[1] - self.last_ball_pos[1]
                    )
                
                self.last_ball_pos = ball_pos
                self.consecutive_detections += 1
                
                # Add to history
                self.ball_history.append(ball_pos)
                if len(self.ball_history) > 10:
                    self.ball_history.pop(0)
                
                return ball_pos
            else:
                self.consecutive_detections = 0
                return None
                
        except Exception as e:
            print(f"❌ Ball detection error: {e}")
            return None
    
    def predict_paddle_position(self, ball_pos, canvas_width, canvas_height, paddle_name):
        """Predict optimal paddle position with advanced strategy"""
        if not ball_pos:
            return canvas_height // 2
        
        ball_x, ball_y = ball_pos
        
        # Determine paddle position
        if paddle_name == "ai1":  # Left paddle
            paddle_x = 50
            target_side = "left"
        else:  # Right paddle
            paddle_x = canvas_width - 50
            target_side = "right"
        
        # Strategy based on ball direction and position
        if len(self.ball_history) >= 2:
            # Calculate ball velocity from recent history
            recent_pos = self.ball_history[-2:]
            vx = recent_pos[1][0] - recent_pos[0][0]
            vy = recent_pos[1][1] - recent_pos[0][1]
            
            # Check if ball is moving towards our paddle
            moving_towards_paddle = False
            if target_side == "left" and vx < 0:
                moving_towards_paddle = True
            elif target_side == "right" and vx > 0:
                moving_towards_paddle = True
            
            if moving_towards_paddle and abs(vx) > 0.5:
                # Predict interception point
                time_to_paddle = abs(ball_x - paddle_x) / abs(vx)
                predicted_y = ball_y + vy * time_to_paddle
                
                # Handle wall bounces
                bounces = 0
                while (predicted_y < 0 or predicted_y > canvas_height) and bounces < 5:
                    if predicted_y < 0:
                        predicted_y = -predicted_y
                    elif predicted_y > canvas_height:
                        predicted_y = 2 * canvas_height - predicted_y
                    bounces += 1
                
                return predicted_y
            else:
                # Ball moving away or slow - defensive position
                # Move towards center but follow ball slightly
                center_y = canvas_height // 2
                ball_influence = 0.3  # How much ball position affects paddle
                return center_y + ball_influence * (ball_y - center_y)
        else:
            # Not enough history - just follow ball
            return ball_y
        
        return canvas_height // 2

class PongAIClient:
    def __init__(self, server_url="http://127.0.0.1:8000", paddle_name="ai1"):
        self.server_url = server_url
        self.paddle_name = paddle_name
        self.running = False
        self.session = None
        
        # Ball tracker
        self.ball_tracker = SmartBallTracker()
        
        # Paddle tracking
        self.current_paddle_y = None
        self.last_command_y = None
        self.paddle_movement_threshold = 15  # Minimum movement to send command
        
        print(f"🚀 Initialized Smart AI client for {paddle_name}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_game_image(self):
        """Capture base64 image from the game server"""
        try:
            payload = {
                "captureOptions": {
                    "format": "jpeg",
                    "quality": 0.8
                },
                "returnBase64": True
            }
            
            async with self.session.post(
                f"{self.server_url}/capture",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=2.0)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "success" and "imageData" in data:
                        return data["imageData"]
                    elif data.get("ok") and "result" in data:
                        return data["result"].get("image_base64")
                    elif "imageData" in data:
                        return data["imageData"]
                    elif "image_base64" in data:
                        return data["image_base64"]
                return None
        except Exception as e:
            return None

    async def send_paddle_command(self, target_y):
        """Send paddle movement command to server with movement tracking"""
        try:
            # Only send command if significant movement or first command
            if (self.last_command_y is None or 
                abs(target_y - self.last_command_y) >= self.paddle_movement_threshold):
                
                payload = {
                    "paddle": self.paddle_name,
                    "y": int(target_y),
                    "immediate": True
                }
                
                async with self.session.post(
                    f"{self.server_url}/control",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=1.0)
                ) as resp:
                    success = resp.status == 200
                    if success:
                        self.last_command_y = target_y
                        self.current_paddle_y = target_y
                    return success
            else:
                # No significant movement needed
                return True
                
        except Exception as e:
            return False

    async def get_score(self):
        """Get current score information"""
        try:
            async with self.session.get(f"{self.server_url}/score") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                return None
        except:
            return None

    def process_image_with_smart_ai(self, base64_image):
        """Process image using smart ball tracking and prediction"""
        try:
            # Decode image
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            img_array = np.array(image)
            
            canvas_height, canvas_width = img_array.shape[:2]
            
            # Detect ball position
            ball_pos = self.ball_tracker.detect_ball(img_array)
            
            # Predict optimal paddle position
            predicted_y = self.ball_tracker.predict_paddle_position(
                ball_pos, canvas_width, canvas_height, self.paddle_name
            )
            
            # Ensure within valid bounds
            paddle_height = 120
            min_y = paddle_height // 2
            max_y = canvas_height - paddle_height // 2
            predicted_y = max(min_y, min(max_y, predicted_y))
            
            return int(predicted_y), ball_pos
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return 350, None

    async def run_ai_loop(self, update_interval=0.02):  # 50 FPS for more responsive movement
        """Main AI loop with frequent updates and better tracking"""
        print(f"🤖 Starting AI loop for paddle: {self.paddle_name}")
        print("🧠 Using Smart Ball Tracking + Predictive AI")
        print("📡 Connecting to game server...")
        
        self.running = True
        consecutive_failures = 0
        frame_count = 0
        last_success_time = time.time()
        last_score_check = time.time()
        
        while self.running:
            try:
                frame_count += 1
                
                # Get current game image
                base64_image = await self.get_game_image()
                
                if base64_image:
                    consecutive_failures = 0
                    last_success_time = time.time()
                    
                    # Process with smart AI
                    target_y, ball_pos = self.process_image_with_smart_ai(base64_image)
                    
                    # Send paddle command (only if significant movement)
                    success = await self.send_paddle_command(target_y)
                    
                    # More frequent logging - every 15 frames
                    if frame_count % 15 == 0:
                        status = "✅" if success else "⚠️"
                        ball_info = f"Ball: {ball_pos}" if ball_pos else "Ball: Not detected"
                        movement = f"Move: {abs(target_y - (self.last_command_y or target_y)):.0f}px" if self.last_command_y else "Move: Initial"
                        print(f"{status} Frame {frame_count}: 🧠 Target Y:{target_y} ({movement}) | {ball_info}")
                        
                        # Check score periodically
                        if time.time() - last_score_check > 5:  # Every 5 seconds
                            score = await self.get_score()
                            if score:
                                print(f"📊 Score - AI1: {score.get('ai1', 0)} | AI2: {score.get('ai2', 0)}")
                            last_score_check = time.time()
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 1:
                        print(f"⚠️ No image received (failure #{consecutive_failures})")
                    
                    if consecutive_failures >= 50 or (time.time() - last_success_time) > 10:
                        print("😴 Too many failures, sleeping longer...")
                        await asyncio.sleep(2)
                        consecutive_failures = 0
                
                await asyncio.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping AI client...")
                break
            except Exception as e:
                print(f"❌ Error in AI loop: {e}")
                await asyncio.sleep(1)

    def stop(self):
        self.running = False

async def main():
    paddle_name = sys.argv[1] if len(sys.argv) > 1 else "ai1"
    
    if paddle_name not in ["ai1", "ai2"]:
        print("❌ Invalid paddle name. Use 'ai1' or 'ai2'")
        print("Usage: python ai_pong_player.py [ai1|ai2]")
        sys.exit(1)
    
    print(f"🚀 Starting High-Performance Pong AI for paddle: {paddle_name}")
    print("🧠 Smart Ball Tracking + Predictive Movement + Frequent Updates")
    print("🎯 Make sure the game server is running on http://127.0.0.1:8000")
    print("🎮 Set the game to 'Remote AI' mode and press ENTER to start!")
    print()
    print("⌨️  Press Ctrl+C to stop")
    
    async with PongAIClient(paddle_name=paddle_name) as client:
        try:
            await client.run_ai_loop(update_interval=0.02)  # 50 FPS
        except KeyboardInterrupt:
            print("👋 AI client stopped by user")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

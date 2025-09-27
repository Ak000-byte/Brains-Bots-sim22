#!/usr/bin/env python3

"""
DUAL AI PONG CLIENT - Controls BOTH ai1 and ai2 paddles simultaneously
Perfect working version with correct canvas dimensions for both paddles
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

class DualPongAI:
    def __init__(self, server_url="http://127.0.0.1:8000"):
        self.server_url = server_url
        self.session = None

        # Correct canvas dimensions from HTML
        self.canvas_width = 1400
        self.canvas_height = 700

        # Ball tracking (shared between both paddles)
        self.ball_history = []
        self.last_ball_pos = None

        print("🎯 DUAL AI PONG CLIENT - Controls BOTH paddles!")
        print(f"📐 Canvas: {self.canvas_width}x{self.canvas_height}")
        print(f"🎮 AI1 (Left) + AI2 (Right) working together")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def control_paddle(self, paddle_name, target_y):
        """Control any paddle using /control endpoint"""
        try:
            payload = {
                "paddle": paddle_name,
                "y": float(target_y),
                "immediate": True
            }

            async with self.session.post(
                f"{self.server_url}/control",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=1.0)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("ok", False)
                return False

        except Exception as e:
            return False

    async def send_ai_prediction(self, paddle_name, target_y, confidence=0.95):
        """Send AI prediction for any paddle"""
        try:
            payload = {
                "model": paddle_name,
                "targetY": float(target_y),
                "confidence": confidence,
                "immediate": True
            }

            async with self.session.post(
                f"{self.server_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=1.0)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("ok", False)
                return False

        except Exception as e:
            return False

    async def get_game_image(self):
        """Get game image using /capture endpoint"""
        try:
            payload = {
                "captureOptions": {"format": "jpeg", "quality": 0.8},
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
                    if data.get("ok") and "result" in data:
                        result = data["result"]
                        return result.get("image_base64")
                return None

        except Exception as e:
            return None

    def detect_ball(self, base64_image):
        """Ball detection - same as working version"""
        try:
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]

            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            img_array = np.array(image)

            # Convert to HSV for better detection
            if len(img_array.shape) == 3:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                # White ball detection
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 30, 255])
                white_mask = cv2.inRange(hsv, lower_white, upper_white)

                # Also try grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

                # Combine masks
                combined_mask = cv2.bitwise_or(white_mask, thresh)
            else:
                gray = img_array.copy()
                _, combined_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h, w = img_array.shape[:2]
            ball_candidates = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 800:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect_ratio = w_box / h_box if h_box > 0 else 0

                    if 0.5 < aspect_ratio < 2.0:
                        center_x = x + w_box // 2
                        center_y = y + h_box // 2

                        # Avoid edges (paddles)
                        edge_margin = 80
                        if (edge_margin < center_x < w - edge_margin and
                            20 < center_y < h - 20):

                            # Calculate circularity
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                ball_candidates.append((center_x, center_y, area, circularity))

            if ball_candidates:
                # Choose most circular candidate
                best = max(ball_candidates, key=lambda x: x[3] * min(x[2] / 50.0, 1.0))
                ball_pos = (best[0], best[1])

                # Update history
                self.ball_history.append(ball_pos)
                if len(self.ball_history) > 10:
                    self.ball_history.pop(0)

                self.last_ball_pos = ball_pos
                return ball_pos

            return None

        except Exception as e:
            return None

    def predict_paddle_position(self, ball_pos, paddle_name):
        """Smart paddle positioning for specified paddle"""
        if not ball_pos:
            return self.canvas_height // 2

        ball_x, ball_y = ball_pos

        # Determine paddle side
        if paddle_name == "ai1":  # Left paddle
            paddle_x = 50
            target_side = "left"
        else:  # Right paddle (ai2)
            paddle_x = self.canvas_width - 50
            target_side = "right"

        # Advanced prediction if we have history
        if len(self.ball_history) >= 3:
            # Calculate velocity from recent positions
            recent = self.ball_history[-3:]
            vx = (recent[2][0] - recent[0][0]) / 2.0
            vy = (recent[2][1] - recent[0][1]) / 2.0

            # Check if ball is moving towards our paddle
            moving_towards = False
            if target_side == "left" and vx < 0:
                moving_towards = True
            elif target_side == "right" and vx > 0:
                moving_towards = True

            if moving_towards and abs(vx) > 1.0:
                # Predict interception
                time_to_paddle = abs(ball_x - paddle_x) / abs(vx)
                predicted_y = ball_y + vy * time_to_paddle

                # Handle wall bounces
                for _ in range(5):
                    if predicted_y < 0:
                        predicted_y = -predicted_y
                    elif predicted_y > self.canvas_height:
                        predicted_y = 2 * self.canvas_height - predicted_y
                    else:
                        break

                return predicted_y
            else:
                # Defensive position
                center_y = self.canvas_height // 2
                return center_y + 0.4 * (ball_y - center_y)

        # Basic strategy: follow ball with center bias
        center_y = self.canvas_height // 2
        return 0.7 * ball_y + 0.3 * center_y

    async def run_dual_ai(self):
        """Main AI loop controlling BOTH paddles"""
        print("🚀 Starting DUAL AI - controlling BOTH ai1 and ai2!")
        print("🎮 AI vs AI autonomous match!")
        print("📐 Full canvas range: 75px to 625px for both paddles")
        print()

        frame_count = 0
        ai1_success = 0
        ai2_success = 0

        while frame_count < 2000:  # Longer for AI vs AI match
            frame_count += 1

            # Get game image
            base64_image = await self.get_game_image()
            if not base64_image:
                if frame_count % 10 == 0:
                    print(f"❌ Frame {frame_count}: No image from /capture")
                await asyncio.sleep(0.1)
                continue

            # Detect ball
            ball_pos = self.detect_ball(base64_image)

            # Calculate target positions for BOTH paddles
            ai1_target_y = self.predict_paddle_position(ball_pos, "ai1")
            ai2_target_y = self.predict_paddle_position(ball_pos, "ai2")

            # Apply bounds for both
            paddle_height = 150
            min_y = paddle_height // 2  # 75
            max_y = self.canvas_height - paddle_height // 2  # 625

            ai1_target_y = max(min_y, min(max_y, ai1_target_y))
            ai2_target_y = max(min_y, min(max_y, ai2_target_y))

            # Control BOTH paddles simultaneously
            ai1_control_task = self.control_paddle("ai1", ai1_target_y)
            ai1_predict_task = self.send_ai_prediction("ai1", ai1_target_y)
            ai2_control_task = self.control_paddle("ai2", ai2_target_y)
            ai2_predict_task = self.send_ai_prediction("ai2", ai2_target_y)

            # Execute all paddle commands in parallel
            results = await asyncio.gather(
                ai1_control_task, ai1_predict_task, 
                ai2_control_task, ai2_predict_task,
                return_exceptions=True
            )

            # Count successes
            if any([results[0], results[1]]):  # ai1 success
                ai1_success += 1
            if any([results[2], results[3]]):  # ai2 success  
                ai2_success += 1

            # Enhanced logging every 15 frames
            if frame_count % 5 == 0:
                if ball_pos:
                    ball_info = f"Ball({ball_pos[0]:.0f},{ball_pos[1]:.0f})"
                else:
                    ball_info = "Ball: Not found"

                ai1_rate = (ai1_success / frame_count) * 100
                ai2_rate = (ai2_success / frame_count) * 100

                print(f"🎯 Frame {frame_count}: {ball_info}")
                print(f"   🔵 AI1 -> Y:{int(ai1_target_y)} | Success: {ai1_rate:.1f}%")  
                print(f"   🔴 AI2 -> Y:{int(ai2_target_y)} | Success: {ai2_rate:.1f}%")

            await asyncio.sleep(0.08)  # Slightly faster for dual control

        print(f"\n🏁 DUAL AI MATCH COMPLETED!")
        print(f"🔵 AI1 Success Rate: {(ai1_success/frame_count)*100:.1f}%")
        print(f"🔴 AI2 Success Rate: {(ai2_success/frame_count)*100:.1f}%")
        print("🎮 Both paddles covered full canvas range!")

async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dual"

    if mode == "dual":
        print("🎮 DUAL AI PONG CLIENT - AI vs AI AUTONOMOUS MATCH")
        print("=" * 60)
        print("🔵 AI1 (Left paddle) - Autonomous control")
        print("🔴 AI2 (Right paddle) - Autonomous control") 
        print("🎯 Both using perfect working algorithm with full range")
        print()
        print("⚠️ Make sure server is running: python server.py")
        print("🌐 And game is loaded in browser at http://127.0.0.1:8000")
        print("🎮 Set game mode to allow both AI paddles")
        print()
        print("⌨️ Press Ctrl+C to stop")
        print()

        async with DualPongAI() as client:
            try:
                await client.run_dual_ai()
            except KeyboardInterrupt:
                print("\n🛑 Dual AI stopped by user")
            except Exception as e:
                print(f"💥 Error: {e}")

    elif mode in ["ai1", "ai2"]:
        # Single paddle mode - import from working version
        from working_fixed_pong_ai import WorkingFixedPongAI

        print(f"🎯 SINGLE AI MODE - Controlling {mode.upper()}")
        print("Using working_fixed_pong_ai.py logic...")

        async with WorkingFixedPongAI(paddle_name=mode) as client:
            try:
                await client.run_working_fixed_ai()
            except KeyboardInterrupt:
                print(f"\n🛑 {mode.upper()} AI stopped by user")
            except Exception as e:
                print(f"💥 Error: {e}")
    else:
        print("❌ Invalid mode. Use:")
        print("  python dual_pong_ai.py dual    # Control both paddles")
        print("  python dual_pong_ai.py ai1     # Control left paddle only")
        print("  python dual_pong_ai.py ai2     # Control right paddle only")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

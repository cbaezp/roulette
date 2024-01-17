import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from roulette import RouletteWheelTracker, RouletteBallTracker, VideoProcessor


class RouletteDataCollector(RouletteWheelTracker, RouletteBallTracker):
    def __init__(self, wheel_tracker_params, ball_tracker_params):
        RouletteWheelTracker.__init__(self, **wheel_tracker_params)
        RouletteBallTracker.__init__(self, **ball_tracker_params)

        self.data = []
        self.prev_ball_info = {"position": None, "time": None, "speed": 0}
        self.prev_wheel_info = {"position": None, "time": None, "speed": 0}

    def process_frame(self, frame):
        current_time = datetime.now()

       
        RouletteWheelTracker.process_frame(self, frame)
        RouletteBallTracker.process_frame(self, frame)

        
        ball_position = self.trajectory[-1] if self.trajectory else None
        ball_speed, ball_acceleration = self.calculate_motion_metrics(
            ball_position, current_time, self.prev_ball_info
        )

        # Wheel green zero position | only for European roulette. The American roulette may have two green zeros 
        wheel_position = self.find_green_zero_position(frame)
        wheel_speed, wheel_acceleration = self.calculate_motion_metrics(
            wheel_position, current_time, self.prev_wheel_info
        )

        
        self.data.append(
            {
                "timestamp": current_time,
                "ball_position_x": ball_position[0] if ball_position else None,
                "ball_position_y": ball_position[1] if ball_position else None,
                "ball_speed": ball_speed,
                "ball_acceleration": ball_acceleration,
                "wheel_position_x": wheel_position[0] if wheel_position else None,
                "wheel_position_y": wheel_position[1] if wheel_position else None,
                "wheel_speed": wheel_speed,
                "wheel_acceleration": wheel_acceleration,
            }
        )

       
        self.update_previous_info(ball_position, current_time, ball_speed, "ball")
        self.update_previous_info(wheel_position, current_time, wheel_speed, "wheel")

    def find_green_zero_position(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

    def calculate_motion_metrics(self, current_position, current_time, previous_info):
        if (
            not current_position
            or not previous_info["position"]
            or not previous_info["time"]
        ):
            return 0, 0

        time_diff = (current_time - previous_info["time"]).total_seconds()
        if time_diff <= 0:
            return 0, 0

        distance = np.linalg.norm(
            np.array(current_position) - np.array(previous_info["position"])
        )
        speed = distance / time_diff
        acceleration = (
            (speed - previous_info["speed"]) / time_diff if time_diff > 0 else 0
        )

        return speed, acceleration

    def update_previous_info(self, position, time, speed, info_type):
        if info_type == "ball":
            self.prev_ball_info = {"position": position, "time": time, "speed": speed}
        elif info_type == "wheel":
            self.prev_wheel_info = {"position": position, "time": time, "speed": speed}

    def save_data_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def reset_data(self):
        self.data = []
        self.prev_ball_info = {"position": None, "time": None, "speed": 0}
        self.prev_wheel_info = {"position": None, "time": None, "speed": 0}


wheel_tracker_params = {
    "center_adjust_x": 20,
    "center_adjust_y": -20,
    "distance_adjustment": 30,
    "fraction": 1 / 100,
}

ball_tracker_params = {
    "model_path": "runs/detect/train2/weights/last.pt",
    "threshold": 0.5,
    "show_bb": True,
    "display_score": True,
    "track_trajectory": True,
    "max_trajectory_length": 50,
}

data_collector = RouletteDataCollector(wheel_tracker_params, ball_tracker_params)
video_processor = VideoProcessor(
    "videos/roulette_test.mp4", "output_video.mp4", [data_collector]
)
video_processor.process_video()
data_collector.save_data_to_csv("roulette_data.csv")

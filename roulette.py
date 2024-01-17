import cv2
import numpy as np
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self, video_path, output_path, trackers):
        self.video_path = video_path
        self.output_path = output_path
        self.trackers = trackers

        self.cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, 30.0, (int(self.cap.get(3)), int(self.cap.get(4)))
        )

    def process_video(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                for tracker in self.trackers:
                    tracker.process_frame(frame)
                self.out.write(frame)
                cv2.imshow("Processed Frame", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


class RouletteWheelTracker:
    def __init__(
        self,
        center_adjust_x=20,  # adjust automatically detected center if needed
        center_adjust_y=-20,
        distance_adjustment=30,  # distance between numbers drawn and the roulette's center
        fraction=1 / 100,  # number spacing
    ):
        self.center_adjust_x = center_adjust_x
        self.center_adjust_y = center_adjust_y
        self.distance_adjustment = distance_adjustment
        self.fraction = fraction

        self.roulette_numbers = [
            0,
            32,
            15,
            19,
            4,
            21,
            2,
            25,
            17,
            34,
            6,
            27,
            13,
            36,
            11,
            30,
            8,
            23,
            10,
            5,
            24,
            16,
            33,
            1,
            20,
            14,
            31,
            9,
            22,
            18,
            29,
            7,
            28,
            12,
            35,
            3,
            26,
        ]
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        self.fixed_distance_to_center = None
        self.default_angle_per_slot = (2 * np.pi) / 37
        self.spacing_adjustment = self.default_angle_per_slot * self.fraction

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        if contours:
            self.process_contours(contours, frame)

    def process_contours(self, contours, frame):
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            self.draw_wheel_center(frame, cx, cy)
            self.draw_numbers(frame, cx, cy)

    def draw_wheel_center(self, frame, cx, cy):
        wheel_center_x = frame.shape[1] // 2 + self.center_adjust_x
        wheel_center_y = frame.shape[0] // 2 + self.center_adjust_y
        cv2.circle(frame, (wheel_center_x, wheel_center_y), 5, (0, 255, 0), -1)

        if self.fixed_distance_to_center is None:
            self.fixed_distance_to_center = (
                np.sqrt((cx - wheel_center_x) ** 2 + (cy - wheel_center_y) ** 2)
                - self.distance_adjustment
            )

    def draw_numbers(self, frame, cx, cy):
        wheel_center_x = frame.shape[1] // 2 + self.center_adjust_x
        wheel_center_y = frame.shape[0] // 2 + self.center_adjust_y
        starting_angle = np.arctan2(cy - wheel_center_y, cx - wheel_center_x)
        angle_per_slot = self.default_angle_per_slot + self.spacing_adjustment

        for i, number in enumerate(self.roulette_numbers):
            angle = starting_angle + i * angle_per_slot
            num_x = int(wheel_center_x + np.cos(angle) * self.fixed_distance_to_center)
            num_y = int(wheel_center_y + np.sin(angle) * self.fixed_distance_to_center)
            cv2.putText(
                frame,
                str(number),
                (num_x, num_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )


class RouletteBallTracker:
    def __init__(
        self,
        model_path,
        threshold=0.8,
        show_bb=False,
        display_score=False,
        track_trajectory=True,
        max_trajectory_length=50,
    ):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.show_bb = show_bb
        self.display_score = display_score
        self.track_trajectory = track_trajectory
        self.max_trajectory_length = max_trajectory_length
        self.trajectory = []

    def process_frame(self, frame):
        results = self.model(frame)[0]

        # Find the detection with the highest score above the threshold
        best_score = self.threshold
        best_result = None
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > best_score:
                best_score = score
                best_result = result

        # Process only the best result
        if best_result:
            x1, y1, x2, y2, score, class_id = best_result
            ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.update_trajectory(ball_position)

            label = results.names[int(class_id)].upper()
            if self.display_score:
                label = f"{label}: {score:.2f}"

            if self.show_bb:
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4
                )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        if self.track_trajectory:
            self.draw_trajectory(frame)

    def update_trajectory(self, position):
        int_position = (int(position[0]), int(position[1]))
        self.trajectory.append(int_position)

        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)

    def draw_trajectory(self, frame):
        for i in range(1, len(self.trajectory)):
            cv2.line(
                frame, self.trajectory[i - 1], self.trajectory[i], (0, 255, 255), 2
            )



if __name__ == "__main__":
    VIDEO_PATH = "videos/roulette_test.mp4"
    OUTPUT_PATH = "output_video.mp4"
    roulette_tracker = RouletteWheelTracker()
    yolo_tracker = RouletteBallTracker(
        model_path="runs/detect/train2/weights/last.pt",
        threshold=0.5,
        display_score=True,
        track_trajectory=False,
    )
    video_processor = VideoProcessor(
        VIDEO_PATH, OUTPUT_PATH, [roulette_tracker, yolo_tracker]
    )
    video_processor.process_video()

import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class RexzeaCasualFaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
        
        self.COLORS = {
            'primary': (255, 200, 0),   
            'secondary': (0, 255, 255),   
            'accent': (0, 165, 255),      
            'warning': (0, 0, 255),       
            'success': (0, 255, 0),       
            'white': (255, 255, 255),     
            'black': (0, 0, 0)            
        }

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.6
        self.THICKNESS = 2

        self.show_mesh = True
        self.show_contours = True
        self.show_metrics = True

    def draw_fancy_rectangle(self, img, pt1, pt2, color, thickness=1, r=10, d=5):
        x1, y1 = pt1
        x2, y2 = pt2

        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def draw_metrics_panel(self, frame, metrics):
        height, width = frame.shape[:2]
        panel_width = 250
        panel_height = 150

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     self.COLORS['black'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        self.draw_fancy_rectangle(frame, (10, 10), (panel_width, panel_height), 
                                self.COLORS['primary'], 2)

        cv2.putText(frame, "Face Analysis Metrics", (20, 35),
                    self.FONT, self.FONT_SCALE, self.COLORS['primary'], 
                    self.THICKNESS)
        
        y_offset = 60
        for label, value in metrics.items():
            cv2.putText(frame, f"{label}: {value}", (20, y_offset),
                       self.FONT, self.FONT_SCALE, self.COLORS['white'], 1)
            y_offset += 25
            

    def draw_status_indicator(self, frame, text, position, is_active):
        color = self.COLORS['success'] if is_active else self.COLORS['warning']

        cv2.circle(frame, position, 8, color, -1)

        cv2.putText(frame, text, (position[0] + 15, position[1] + 5),
                    self.FONT, self.FONT_SCALE, color, 1)

    def calculate_eye_aspect_ratio(self, landmarks, eye_indexes):
        points = []
        for index in eye_indexes:
            point = landmarks.landmark[index]
            points.append([point.x, point.y])
        
        points = np.array(points)
        vertical_dist1 = np.linalg.norm(points[1] - points[5])
        vertical_dist2 = np.linalg.norm(points[2] - points[4])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear

    def detect_mouth_open(self, landmarks):
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        
        distance = math.sqrt(
            (upper_lip.x - lower_lip.x)**2 + 
            (upper_lip.y - lower_lip.y)**2
        )
        return distance > 0.02

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.show_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                if self.show_contours:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                left_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.LEFT_EYE_INDEXES)
                right_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.RIGHT_EYE_INDEXES)
                mouth_open = self.detect_mouth_open(face_landmarks)
                blinking = left_ear < 0.2 and right_ear < 0.2
                
                metrics = {
                    "Left Eye EAR": f"{left_ear:.2f}",
                    "Right Eye EAR": f"{right_ear:.2f}",
                    "Mouth Status": "Open" if mouth_open else "Closed",
                    "Blink": "Detected" if blinking else "Not Detected"
                }
                
                if self.show_metrics:
                    self.draw_metrics_panel(frame, metrics)
                

        
        return frame

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter(f'face_detection_{timestamp}.avi', 
                            fourcc, 20.0, (width, height))

        cv2.namedWindow('Enhanced Face Detection')
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            processed_frame = self.process_frame(frame)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, f"REC {timestamp}", (10, height - 20),
                       self.FONT, 0.5, self.COLORS['warning'], 1)
            
            out.write(processed_frame)
            cv2.imshow('Enhanced Face Detection', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.show_mesh = not self.show_mesh
            elif key == ord('c'):
                self.show_contours = not self.show_contours
            elif key == ord('s'):
                self.show_metrics = not self.show_metrics
            elif key == ord('p'):

                screenshot_path = f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(screenshot_path, processed_frame)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RexzeaCasualFaceDetector()
    detector.start_webcam()

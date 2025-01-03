import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

class hai:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        self.NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

        self.current_mask = 'venetian' 
        self.masks = ['venetian', 'demon', 'cyber', 'dragon']
        
        self.mask_colors = {
            'gold': (0, 215, 255),
            'silver': (192, 192, 192),
            'dark_green': (0, 128, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'orange': (0, 165, 255), 
            'yellow': (0, 255, 255) 
        }




    # Venetian Mask
    def create_venetian_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)
    
        face_points = []
        for idx in self.FACE_OUTLINE:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            face_points.append([x, y])
        
        face_points = np.array(face_points, np.int32)
        
        cv2.fillPoly(mask_layer, [face_points], self.mask_colors['gold'])
        
        for idx in range(len(face_points)-1):
            pt1 = tuple(face_points[idx])
            pt2 = tuple(face_points[idx+1])
            cv2.line(mask_layer, pt1, pt2, self.mask_colors['silver'], 2)
        
        for eye_idx in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = []
            for idx in eye_idx:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, np.int32)
            cv2.fillPoly(mask_layer, [eye_points], self.mask_colors['black'])
            cv2.polylines(mask_layer, [eye_points], True, self.mask_colors['silver'], 2)
        
        center_x = int(landmarks.landmark[4].x * w)
        center_y = int(landmarks.landmark[4].y * h)
        
        for angle in range(0, 360, 45):
            end_x = int(center_x + 30 * math.cos(math.radians(angle)))
            end_y = int(center_y + 30 * math.sin(math.radians(angle)))
            cv2.line(mask_layer, (center_x, center_y), (end_x, end_y), 
                    self.mask_colors['silver'], 2)

        alpha = 0.7
        frame_with_mask = cv2.addWeighted(frame, 1, mask_layer, alpha, 0)
        return frame_with_mask





    # Demon Mask
    def create_demon_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)
        
        face_points = []
        for idx in self.FACE_OUTLINE:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            face_points.append([x, y])
        
        face_points = np.array(face_points, np.int32)
        cv2.fillPoly(mask_layer, [face_points], self.mask_colors['red'])
        
        top_head = landmarks.landmark[10]
        head_x = int(top_head.x * w)
        head_y = int(top_head.y * h)
        
        horn_left = np.array([
            [head_x-50, head_y],
            [head_x-30, head_y-80],
            [head_x-10, head_y-40]
        ], np.int32)
        cv2.fillPoly(mask_layer, [horn_left], self.mask_colors['black'])
        
        horn_right = np.array([
            [head_x+50, head_y],
            [head_x+30, head_y-80],
            [head_x+10, head_y-40]
        ], np.int32)
        cv2.fillPoly(mask_layer, [horn_right], self.mask_colors['black'])
        
        for eye_idx in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = []
            for idx in eye_idx:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, np.int32)
            cv2.fillPoly(mask_layer, [eye_points], (0, 165, 255))  
            
            eye_center = np.mean(eye_points, axis=0).astype(int)
            for radius in range(5, 20, 5):
                cv2.circle(mask_layer, tuple(eye_center), radius, (0, 165, 255), 2)

        for i in range(10):
            angle = np.random.randint(0, 360)
            radius = np.random.randint(50, 100)
            x = int(head_x + radius * math.cos(math.radians(angle)))
            y = int(head_y + radius * math.sin(math.radians(angle)))
            cv2.circle(mask_layer, (x, y), 10, (0, 0, 255), -1)

        alpha = 0.8
        frame_with_mask = cv2.addWeighted(frame, 1, mask_layer, alpha, 0)
        return frame_with_mask
    



    # Cyber Mask
    def create_cyber_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)
        
        for x in range(0, w, 20):
            cv2.line(mask_layer, (x, 0), (x, h), (0, 50, 0), 1)
        for y in range(0, h, 20):
            cv2.line(mask_layer, (0, y), (w, y), (0, 50, 0), 1)
        
        connections = self.mp_face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            x1 = int(start_point.x * w)
            y1 = int(start_point.y * h)
            x2 = int(end_point.x * w)
            y2 = int(end_point.y * h)
            
            cv2.line(mask_layer, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        eye_points = []
        for idx in self.LEFT_EYE + self.RIGHT_EYE:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points, np.int32)
        cv2.fillPoly(mask_layer, [eye_points], (255, 0, 0))
        
        text_y = int(landmarks.landmark[10].y * h) - 50
        cv2.putText(mask_layer, "CYBER.SYS", (int(w/2)-50, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        alpha = 0.7
        frame_with_mask = cv2.addWeighted(frame, 1, mask_layer, alpha, 0)
        
        bloom = cv2.GaussianBlur(mask_layer, (15, 15), 0)
        frame_with_mask = cv2.addWeighted(frame_with_mask, 1, bloom, 0.3, 0)
        
        return frame_with_mask
    




    # Dragon Mask
    def create_dragon_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)

        face_points = []
        for idx in self.FACE_OUTLINE:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            face_points.append([x, y])

        face_points = np.array(face_points, np.int32)
        cv2.fillPoly(mask_layer, [face_points], (0, 100, 0))  

        top_head = landmarks.landmark[10]
        head_x = int(top_head.x * w)
        head_y = int(top_head.y * h)

        horn_left = np.array([
            [head_x - 50, head_y],
            [head_x - 30, head_y - 80],
            [head_x - 10, head_y - 40]
        ], np.int32)
        cv2.fillPoly(mask_layer, [horn_left], (0, 128, 0)) 

        horn_right = np.array([
            [head_x + 50, head_y],
            [head_x + 30, head_y - 80],
            [head_x + 10, head_y - 40]
        ], np.int32)
        cv2.fillPoly(mask_layer, [horn_right], (0, 128, 0))

        for i in range(0, len(face_points), 5):  
            x, y = face_points[i]
            cv2.circle(mask_layer, (x, y), 10, (0, 255, 0), -1)  

        for i in range(0, len(face_points), 5):
            x, y = face_points[i]
            cv2.circle(mask_layer, (x, y), 15, (0, 200, 0), 2) 

        for eye_idx in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = []
            for idx in eye_idx:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, np.int32)
            cv2.fillPoly(mask_layer, [eye_points], (255, 0, 0))

            eye_center = np.mean(eye_points, axis=0).astype(int)
            for radius in range(5, 20, 5):
                cv2.circle(mask_layer, tuple(eye_center), radius, (255, 0, 0), 2)

        for i in range(10):
            angle = np.random.randint(0, 360)
            radius = np.random.randint(50, 100)
            x = int(head_x + radius * math.cos(math.radians(angle)))
            y = int(head_y + radius * math.sin(math.radians(angle)))
            cv2.circle(mask_layer, (x, y), 10, (0, 255, 255), -1)  

        for i in range(5):
            cv2.circle(mask_layer, (head_x, head_y), 50 + i * 10, (0, 255, 0), 1)  
        alpha = 0.8
        frame_with_mask = cv2.addWeighted(frame, 1, mask_layer, alpha, 0)
        return frame_with_mask
    

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.current_mask == 'venetian':
                    frame = self.create_venetian_mask(frame, face_landmarks)
                elif self.current_mask == 'demon':
                    frame = self.create_demon_mask(frame, face_landmarks)
                elif self.current_mask == 'cyber':
                    frame = self.create_cyber_mask(frame, face_landmarks)
                elif self.current_mask == 'dragon':
                    frame = self.create_dragon_mask(frame, face_landmarks)
        
        cv2.putText(frame, f"Mask: {self.current_mask} (Press 'n' to change)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Enhanced 3D Masks', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_index = self.masks.index(self.current_mask)
                next_index = (current_index + 1) % len(self.masks)
                self.current_mask = self.masks[next_index]
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'mask_{self.current_mask}_{timestamp}.jpg', processed_frame)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = hai()
    detector.start_webcam()




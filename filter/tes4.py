import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime
import time

class RexzeaFilterFaceDetector:
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
        self.masks = ['venetian', 'demon', 'cyber', 'angel', 'dragon']
        
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
            'yellow': (0, 255, 255),
            'dragon_red': (139, 0, 0),
            'dragon_bright': (178, 34, 34)
        }




# Venetian Mask
    def venetian_mask(self, frame, landmarks):
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
    def demon_mask(self, frame, landmarks):
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
    def cyber_mask(self, frame, landmarks):
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
    





    def angle_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)

        face_points = []
        for idx in self.FACE_OUTLINE:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            face_points.append([x, y])

        face_points = np.array(face_points, np.int32)
        cv2.fillPoly(mask_layer, [face_points], (240, 248, 255))

        top_head = landmarks.landmark[10]
        head_x = int(top_head.x * w)
        head_y = int(top_head.y * h)


        wing_left = np.array([
            [head_x-150, head_y-50],
            [head_x-300, head_y-200],
            [head_x-100, head_y-100],
            [head_x-50, head_y],
        ], np.int32)
        cv2.fillPoly(mask_layer, [wing_left], (255, 255, 255))

        
        wing_right = np.array([
            [head_x+150, head_y-50],
            [head_x+300, head_y-200],
            [head_x+100, head_y-100],
            [head_x+50, head_y]
        ], np.int32)
        cv2.fillPoly(mask_layer, [wing_right], (255, 255, 255))



        # halo
        halo_center = (head_x, head_y - 100)
        cv2.circle(mask_layer, halo_center, 50, (255, 255, 200), -1)
        cv2.circle(mask_layer, halo_center, 55, (200, 200, 255), 2)

        for eye_idx in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = []
            for idx in eye_idx:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, np.int32)
        
            eye_center = np.mean(eye_points, axis=0).astype(int)
            for radius in range(5, 20, 5):
                cv2.circle(mask_layer, tuple(eye_center), radius, (255, 255, 200), 2)
    
        glow = cv2.GaussianBlur(mask_layer, (21, 21), 0)
    
        frame_with_mask = cv2.addWeighted(frame, 0.7, mask_layer, 0.3, 0)
        frame_with_mask = cv2.addWeighted(frame_with_mask, 0.9, glow, 0.1, 0)
    
        cv2.putText(frame_with_mask, "ANGEL", 
                    (int(w/2)-50, head_y-200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 200), 2)
    
        return frame_with_mask


    


    # Dragon Mask
    def dragon_mask(self, frame, landmarks):
        h, w = frame.shape[:2]
        mask_layer = np.zeros_like(frame)
        fire_layer = np.zeros_like(frame)
        
        scale_pattern = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, 15):
            for x in range(0, w, 15):
                cv2.circle(scale_pattern, (x, y), 7, self.mask_colors['dragon_red'], -1)
                cv2.circle(scale_pattern, (x, y), 5, self.mask_colors['dragon_bright'], -1)
        
        face_contour = []
        for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
            for point_idx in connection:
                point = landmarks.landmark[point_idx]
                x = int(point.x * w)
                y = int(point.y * h)
                face_contour.append([x, y])
        
        face_contour = np.array(face_contour, np.int32)
        
        cv2.fillPoly(mask_layer, [face_contour], self.mask_colors['dragon_bright'])
        
        mask_area = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [face_contour], 255)
        scale_pattern_masked = cv2.bitwise_and(scale_pattern, scale_pattern, mask=mask_area)
        mask_layer = cv2.addWeighted(mask_layer, 0.7, scale_pattern_masked, 0.3, 0)
        
        for i in range(len(face_contour)):
            x, y = face_contour[i][0], face_contour[i][1]
            for j in range(20):
                offset_y = int(10 * np.sin(time.time() * 5 + i * 0.1))
                color_intensity = 255 - j * 10
                cv2.circle(fire_layer, 
                          (x, y - j * 2 + offset_y), 
                          2, 
                          (0, color_intensity, 255), 
                          -1)
        
        eye_centers = []
        for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
            eye_points = []
            for idx in eye_indices:
                point = landmarks.landmark[idx]
                x, y = int(point.x * w), int(point.y * h)
                eye_points.append([x, y])
            eye_points = np.array(eye_points, np.int32)
            
            M = cv2.moments(eye_points)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                eye_centers.append((cx, cy))
                
                for radius in range(15, 0, -1):
                    intensity = int(255 * (radius / 15))
                    cv2.circle(mask_layer, (cx, cy), radius, (0, intensity, intensity), -1)

        
        for eye_center in eye_centers:
            horn_top = (eye_center[0], eye_center[1] - 50)
            horn_points = np.array([
                eye_center,
                (eye_center[0] - 20, eye_center[1] - 20),
                horn_top,
                (eye_center[0] + 20, eye_center[1] - 20)
            ], np.int32)
            cv2.fillPoly(mask_layer, [horn_points], self.mask_colors['dragon_red'])
            cv2.line(mask_layer, eye_center, horn_top, self.mask_colors['dragon_bright'], 2)
        
        glow = cv2.GaussianBlur(mask_layer, (21, 21), 0)
        fire_glow = cv2.GaussianBlur(fire_layer, (15, 15), 0)
        
        frame_with_mask = cv2.addWeighted(frame, 0.6, mask_layer, 0.4, 0)
        frame_with_mask = cv2.addWeighted(frame_with_mask, 0.8, glow, 0.2, 0)
        frame_with_mask = cv2.addWeighted(frame_with_mask, 0.8, fire_layer, 0.6, 0)
        frame_with_mask = cv2.addWeighted(frame_with_mask, 0.8, fire_glow, 0.2, 0)
        
        text = "DRAGON"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = int(w/2 - text_size[0]/2)
        text_y = int(landmarks.landmark[10].y * h) - 50
        
        cv2.putText(frame_with_mask, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame_with_mask, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        return frame_with_mask

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.current_mask == 'venetian':
                    frame = self.venetian_mask(frame, face_landmarks)
                elif self.current_mask == 'demon':
                    frame = self.demon_mask(frame, face_landmarks)
                elif self.current_mask == 'cyber':
                    frame = self.cyber_mask(frame, face_landmarks)
                elif self.current_mask == 'angel':
                    frame = self.angle_mask(frame, face_landmarks)
                elif self.current_mask == 'dragon':
                    frame = self.dragon_mask(frame, face_landmarks)
        
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
    detector = RexzeaFilterFaceDetector()
    detector.start_webcam()

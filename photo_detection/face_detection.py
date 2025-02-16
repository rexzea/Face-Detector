import cv2
import numpy as np
import os
import sys
import logging
from datetime import datetime
from pathlib import Path # next updt
from typing import List, Dict, Tuple

class RexzeaFaceDetection:
    
    def __init__(self, min_face_size: int = 30):
        self.logger = self._setup_logger()
        self.min_face_size = (min_face_size, min_face_size)
        
        # init haar cascade detector
        self._init_detector()
        self._init_enhancement_params()
        
        self.logger.info("Face detector initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('RexzeaFaceDetection')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _init_detector(self):
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(haar_path)
        
        if self.face_detector.empty():
            raise ValueError("Error loading Haar cascade classifier")
        










    def _init_enhancement_params(self):
        self.enhancement_params = {
            'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)),
            'bilateral_params': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
        }



    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        # convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # contrast
        l_enhanced = self.enhancement_params['clahe'].apply(l)
        
        # merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(
            enhanced,
            **self.enhancement_params['bilateral_params']
        )
        
        return enhanced

    def detect_faces(self, image: np.ndarray) -> List[Dict]:

        """
        DETECT FACES IN THE IMAGE
        """

        # image
        enhanced_image = self.enhance_image(image)
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # convert to list of dictionaries with additional info
        return [
            {
                'bbox': (x, y, w, h),
                'confidence': self._estimate_confidence(gray[y:y+h, x:x+w])
            }
            for (x, y, w, h) in faces
        ] if len(faces) > 0 else []

    def _estimate_confidence(self, face_roi: np.ndarray) -> float:

        """
        ESTIMATE CONFIDENCE OF FACE DETECTION
        """


        # calculate basic metrics for confidence estimation
        clarity = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        contrast = face_roi.std()
        
        # normalize metrics
        clarity_score = min(clarity / 500.0, 1.0)
        contrast_score = min(contrast / 100.0, 1.0)
        
        # combined confidence score
        confidence = (clarity_score + contrast_score) / 2.0
        return min(max(confidence, 0.5), 1.0)

    def visualize_results(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # detection box
            self._draw_fancy_box(result, (x, y, w, h), confidence)
            
            # add face information
            self._add_face_info(result, face)
        
        # add detection summary
        self._add_detection_summary(result, len(faces))
        
        return result

    def _draw_fancy_box(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       confidence: float):
        x, y, w, h = bbox
        
        # get color based on confidwnce
        color = self._get_confidence_color(confidence)
        
        # main rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # corner highlights
        corner_length = min(w, h) // 4
        thickness = 2
        
        # corners
        for corner in [(0,0), (w,0), (0,h), (w,h)]:  # Top-left, top-right, bottom-left, bottom-right
            cx, cy = corner
            # horizontal lines
            cv2.line(image, 
                    (x + cx - (corner_length if cx else 0), y + cy),
                    (x + cx + (corner_length if not cx else 0), y + cy),
                    color, thickness)
            # vertical lines
            cv2.line(image,
                    (x + cx, y + cy - (corner_length if cy else 0)),
                    (x + cx, y + cy + (corner_length if not cy else 0)),
                    color, thickness)

    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        if confidence >= 0.8:
            return (0, 255, 0)  # green
        elif confidence >= 0.6:
            return (0, 255, 255)  # yellow
        else:
            return (0, 165, 255)  # 0range

    def _add_face_info(self, image: np.ndarray, face: Dict):
        x, y, w, h = face['bbox']
        confidence = face['confidence']
        
        # info text
        info_text = f"Conf: {confidence:.2f}"
        
        # add background for text
        (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - 20), (x + text_w, y), (0, 0, 0), -1)
        
        # add text
        cv2.putText(image, info_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _add_detection_summary(self, image: np.ndarray, num_faces: int):
        summary_text = f"Detected Faces: {num_faces}"
        cv2.putText(image, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def save_result(self, image: np.ndarray, output_path: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename, ext = os.path.splitext(output_path)
        final_path = f"{filename}_{timestamp}{ext}"
        
        # create directory if it doesnt exist
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
        
        # save image
        cv2.imwrite(final_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.logger.info(f"Result saved to {final_path}")
        
        return final_path
    







def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Face Detection System')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output', default='results/detected.jpg', help='Output path')
    parser.add_argument('--min-face-size', type=int, default=30, help='Minimum face size')
    
    args = parser.parse_args()

    try:
        # init detector
        detector = RexzeaFaceDetection(min_face_size=args.min_face_size)
        
        # load and process image
        print("Loading image...")
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Could not load image: {args.image_path}")
            
        print("Detecting faces...")
        faces = detector.detect_faces(image)
        
        print(f"Found {len(faces)} faces!")
        
        # visualize and save results
        result_image = detector.visualize_results(image, faces)
        saved_path = detector.save_result(result_image, args.output)
        
        print(f"Results saved to {saved_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
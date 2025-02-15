import cv2
import numpy as np
import os
import sys
import logging
from datetime import datetime
from pathlib import Path # next updt
from typing import Tuple, List, Optional # next updt (deff function)

class RexzeaFaceDetector:
    def __init__(self, confidence_threshold: float = 0.5): 



        """
        initialize the face detector with various detection models.
         
        Args:
            confidence_threshold (float): Detection confidence threshold (0-1)
        """
        self.logger = self._setup_logger()
        self.confidence_threshold = confidence_threshold
        


        # initialize different detection models
        self._init_haar_cascade()
        self._init_dnn_detector()
        
        self.logger.info("Face detector initialized successfully")







    def _setup_logger(self) -> logging.Logger:

        """Configure logging for the detector"""


        logger = logging.getLogger('RexzeaFaceDetector')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    





    def _init_haar_cascade(self):

        """Initialize Haar Cascade classifier"""

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.haar_cascade.empty():
            self.logger.error("Failed to load Haar cascade classifier")
            raise ValueError("Error loading Haar cascade classifier")
        


 


    def _init_dnn_detector(self):

        """Initialize DNN-based face detector"""

        # you would need to download these files separately
        model_path = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        config_path = "models/face_detector/deploy.prototxt"
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            self.dnn_detector = cv2.dnn.readNet(model_path, config_path)
            self.use_dnn = True
            self.logger.info("DNN detector loaded successfully")
        else:
            self.use_dnn = False
            self.logger.warning("DNN model files not found, falling back to Haar cascade")






    def load_image(self, image_path: str) -> np.ndarray:

        """
        Load and preprocess an image from the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns: 
            np.ndarray: Loaded image in BGR format
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """

        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Error loading image: {image_path}")
            
        return image
    





    def detect_faces(self, image: np.ndarray, method: str = 'haar') -> List[Tuple[int, int, int, int]]:

        """
        Detect faces in the image using specified method.
        
        Args:
            image (np.ndarray): Input image
            method (str): Detection method ('haar' or 'dnn')
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates (x, y, w, h)
        """

        if method == 'dnn' and self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
        





    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:

        """Detect faces using Haar cascade"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces if len(faces) > 0 else []






    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:

        """Detect faces using DNN model"""
        
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x, y, x2, y2 = box.astype(int)
                faces.append((x, y, x2 - x, y2 - y))
        
        return faces






    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                  draw_landmarks: bool = True) -> np.ndarray:
        
        """
        Draw detected faces on the image with optional facial landmarks.
        
        Args:
            image (np.ndarray): Input image
            faces (List[Tuple]): List of face coordinates
            draw_landmarks (bool): Whether to draw facial landmarks
            
        Returns:
            np.ndarray: Image with drawn faces
        """

        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            # draw face rectangle
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if draw_landmarks:
                # draw facial landmarks (simplified)
                face_roi = image[y:y+h, x:x+w]
                try:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    # Add basic landmark detection here if needed
                except Exception as e:
                    self.logger.warning(f"Could not draw landmarks: {str(e)}")
        
        # add detection info
        cv2.putText(result_image, f'Faces detected: {len(faces)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_image
    





    def save_image(self, image: np.ndarray, output_path: str, 
                  create_timestamp: bool = True) -> str:
        
        """
        Save the processed image with optional timestamp.
        
        Args:
            image (np.ndarray): Image to save
            output_path (str): Output file path
            create_timestamp (bool): Whether to add timestamp to filename
            
        Returns:
            str: Path where image was saved
        """

        if create_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename, ext = os.path.splitext(output_path)
            output_path = f"{filename}_{timestamp}{ext}"
        
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        cv2.imwrite(output_path, image)
        self.logger.info(f"Image saved successfully to {output_path}")
        return output_path


def main():
    """Main function to run face detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Face Detection Tool')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output', default='detected_faces.jpg', help='Output image path')
    parser.add_argument('--method', choices=['haar', 'dnn'], default='haar', help='Detection method')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--landmarks', action='store_true', help='Draw facial landmarks')
    
    args = parser.parse_args()

    try:
        # init detector
        detector = RexzeaFaceDetector(confidence_threshold=args.confidence)
        # process image
        print("Loading image...")
        image = detector.load_image(args.image_path)
        print(f"Detecting faces using {args.method} method...")
        faces = detector.detect_faces(image, method=args.method)
        print(f"Found {len(faces)} faces!")
        # draw results
        result_image = detector.draw_faces(image, faces, draw_landmarks=args.landmarks)
        # save results
        saved_path = detector.save_image(result_image, args.output)
        print(f"Results saved to {saved_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

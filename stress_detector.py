import tensorflow as tf
import cv2
import numpy as np
import os
import sys

class StressDetector:
    def __init__(self, model_path='stress_detection_model.h5'):
        """Initialize the stress detector with trained model"""
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at '{model_path}'")
            print("Please ensure 'stress_detection_model.h5' is in the same folder")
            sys.exit(1)
        
        print("🔄 Loading model...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Smoothing buffer
        self.stress_history = []
        self.window_size = 10
        
        print("🎥 Starting webcam...")
        print("Press 'q' to quit\n")
    
    def preprocess_face(self, face_roi):
        """Preprocess face for model input"""
        face = cv2.resize(face_roi, (48, 48))
        face = face.reshape(1, 48, 48, 1).astype('float32') / 255.0
        return face
    
    def predict_stress(self, face):
        """Predict stress level from face"""
        prob = self.model.predict(face, verbose=0)[0][0]
        
        # Temporal smoothing
        self.stress_history.append(prob)
        if len(self.stress_history) > self.window_size:
            self.stress_history.pop(0)
        
        avg_prob = np.mean(self.stress_history)
        is_stressed = avg_prob > 0.5
        
        return avg_prob, is_stressed
    
    def draw_results(self, frame, x, y, w, h, prob, is_stressed):
        """Draw detection results on frame"""
        # Color: Red=Stress, Green=Calm
        color = (0, 0, 255) if is_stressed else (0, 255, 0)
        
        # Label
        label = f"STRESS: {prob:.1%}" if is_stressed else f"CALM: {prob:.1%}"
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw probability bar
        bar_x, bar_y = x, y + h + 10
        bar_w, bar_h = w, 15
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50, 50, 50), -1)
        
        # Filled bar
        filled = int(bar_w * prob)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255, 255, 255), 1)
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✅ Webcam active!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face = self.preprocess_face(face_roi)
                
                # Predict
                prob, is_stressed = self.predict_stress(face)
                
                # Draw results
                self.draw_results(frame, x, y, w, h, prob, is_stressed)
            
            # Show face count
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show FPS info
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display
            cv2.imshow('Real-Time Stress Detection (79% Acc)', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Stopping...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")


def main():
    """Main entry point"""
    print("=" * 60)
    print("STRESS DETECTION SYSTEM")
    print("Model: CNN trained on FER2013 (79% accuracy)")
    print("=" * 60)
    
    # Check for model file
    model_path = 'stress_detection_model.h5'
    
    # Allow custom path via command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Initialize and run
    detector = StressDetector(model_path)
    detector.run()
    
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
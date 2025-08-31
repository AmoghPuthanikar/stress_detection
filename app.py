import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import os
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: linear-gradient(to right, green, red);
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stressed {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.2);
    }
    .not-stressed {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.2);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
    }
    .sidebar-content {
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Create cache directory if it doesn't exist
if not os.path.exists("cache"):
    os.makedirs("cache")

# Create history directory if it doesn't exist
if not os.path.exists("history"):
    os.makedirs("history")


# ===== MODEL MANAGEMENT =====
@st.cache_resource
def load_stress_model(model_path):
    """Load and cache the stress detection model"""
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ===== IMAGE PROCESSING FUNCTIONS =====
def preprocess_image(img, target_size=(48, 48)):
    """Preprocess image for model prediction"""
    try:
        # Convert to grayscale if the image has 3 channels
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Resize to target size
        resized = cv2.resize(gray, target_size)

        # Normalize pixel values
        normalized = resized.astype("float32") / 255.0

        # Reshape for model input (add channel dimension)
        return normalized.reshape(target_size[0], target_size[1], 1)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


def detect_faces(img):
    """Detect faces in an image and return the image with rectangles and face coordinates"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Create a copy of the image to draw on
        img_with_faces = img.copy()

        # Draw rectangles around faces
        for x, y, w, h in faces:
            cv2.rectangle(img_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_with_faces, faces
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        return img, []


def get_face_crops(img, faces):
    """Extract face regions from the image"""
    face_crops = []
    for x, y, w, h in faces:
        face_crops.append(img[y : y + h, x : x + w])
    return face_crops


# ===== VISUALIZATION FUNCTIONS =====
def create_stress_gauge(stress_level):
    """Create a matplotlib gauge chart for stress level visualization"""
    fig, ax = plt.subplots(figsize=(4, 0.3), subplot_kw={"aspect": "equal"})

    # Create a horizontal bar
    ax.barh(0, 100, height=1, color="lightgray")

    # Create the stress level indicator
    stress_color = plt.cm.RdYlGn_r(stress_level)
    ax.barh(0, stress_level * 100, height=1, color=stress_color)

    # Remove axes and spines
    ax.axis("off")

    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)

    return buf


def get_image_base64(img_buf):
    """Convert image buffer to base64 for HTML embedding"""
    return base64.b64encode(img_buf.getvalue()).decode()


# ===== HISTORY MANAGEMENT =====
def save_to_history(img, stress_level, timestamp):
    """Save detection results to history"""
    try:
        filename = f"history/detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save metadata
        with open(
            f"history/detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt", "w"
        ) as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Stress Level: {stress_level:.2f}\n")
            f.write(
                f"Status: {'Stressed' if stress_level >= 0.5 else 'Not Stressed'}\n"
            )

        return True
    except Exception as e:
        st.error(f"Error saving to history: {e}")
        return False


def load_history():
    """Load detection history"""
    history = []
    try:
        for file in os.listdir("history"):
            if file.endswith(".txt"):
                with open(os.path.join("history", file), "r") as f:
                    metadata = f.read()

                # Get corresponding image file
                img_file = file.replace(".txt", ".jpg")
                if os.path.exists(os.path.join("history", img_file)):
                    history.append(
                        {
                            "metadata": metadata,
                            "image": os.path.join("history", img_file),
                        }
                    )

        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["image"], reverse=True)
        return history
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []


# ===== WEBCAM HANDLING =====
class FaceCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None

    def transform(self, frame):
        self.last_frame = frame.to_ndarray(format="bgr24")
        output_img, _ = detect_faces(self.last_frame.copy())
        return output_img


# ===== MAIN APPLICATION =====
def main():
    # Load model
    model = load_stress_model("model/stress_model.h5")
    if model is None:
        st.error("Failed to load stress detection model. Please check the model path.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=80)
        st.markdown("## Settings")

        # Model settings
        st.markdown("### Model Configuration")
        confidence_threshold = st.slider(
            "Stress Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Adjust the threshold for classifying stress",
        )

        # Display options
        st.markdown("### Display Options")
        show_confidence = st.checkbox("Show Confidence Score", value=True)
        show_gauge = st.checkbox("Show Stress Gauge", value=True)

        # History options
        st.markdown("### History")
        save_results = st.checkbox("Save Results to History", value=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    st.markdown(
        "<h1 class='main-header'>🧠 Stress Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='sub-header'>Upload an image or use your webcam to detect stress levels</p>",
        unsafe_allow_html=True,
    )

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Detection", "History", "About"])

    # Detection Tab
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            # Input method selection
            option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

            image_ready = False
            img_to_process = None

            # Image upload option
            if option == "Upload Image":
                uploaded = st.file_uploader(
                    "Upload an image containing faces",
                    type=["jpg", "png", "jpeg"],
                    help="The image should clearly show the face(s)",
                )

                if uploaded is not None:
                    try:
                        img_pil = Image.open(uploaded).convert("RGB")
                        img = np.array(img_pil)
                        st.image(img, caption="Uploaded Image", use_column_width=True)
                        img_to_process = img
                        image_ready = True
                    except Exception as e:
                        st.error(f"Error processing uploaded image: {e}")

            # Webcam option
            elif option == "Use Webcam":
                st.info(
                    "Please allow camera access and click 'Start' to begin the webcam stream"
                )
                ctx = webrtc_streamer(
                    key="webcam",
                    video_transformer_factory=FaceCaptureTransformer,
                    media_stream_constraints={"video": True, "audio": False},
                )

                if ctx.video_transformer:
                    if st.button("Capture Frame", key="capture_btn"):
                        if ctx.video_transformer.last_frame is not None:
                            img_to_process = cv2.cvtColor(
                                ctx.video_transformer.last_frame, cv2.COLOR_BGR2RGB
                            )
                            st.image(
                                img_to_process,
                                caption="Captured Frame",
                                use_column_width=True,
                            )
                            image_ready = True
                        else:
                            st.warning(
                                "No frame captured. Please make sure the webcam is streaming."
                            )

        with col2:
            # Analysis section
            st.markdown("### Analysis Results")

            if image_ready and img_to_process is not None:
                if st.button("Analyze Image", key="analyze_btn"):
                    with st.spinner("Analyzing image..."):
                        # Detect faces
                        vis_img, faces = detect_faces(img_to_process.copy())

                        if len(faces) == 0:
                            st.warning(
                                "No faces detected. Please try another image with clearer faces."
                            )
                        else:
                            st.success(f"{len(faces)} face(s) detected")
                            st.image(
                                vis_img,
                                caption="Detected Face(s)",
                                use_column_width=True,
                            )

                            # Process each detected face
                            for i, (x, y, w, h) in enumerate(faces):
                                st.markdown(f"#### Face #{i + 1}")

                                # Extract and preprocess face
                                face = img_to_process[y : y + h, x : x + w]
                                preprocessed = preprocess_image(face)

                                if preprocessed is not None:
                                    # Make prediction
                                    pred = model.predict(
                                        np.expand_dims(preprocessed, 0)
                                    )[0][0]

                                    # Determine stress status based on threshold
                                    is_stressed = pred >= confidence_threshold
                                    status_class = (
                                        "stressed" if is_stressed else "not-stressed"
                                    )

                                    # Display results
                                    with st.container():
                                        st.markdown(
                                            f"<div class='result-card {status_class}'>",
                                            unsafe_allow_html=True,
                                        )

                                        # Show face crop
                                        st.image(face, width=150)

                                        # Show stress level
                                        if show_confidence:
                                            st.markdown(
                                                f"**Stress Probability:** {pred * 100:.2f}%"
                                            )

                                        # Show gauge visualization
                                        if show_gauge:
                                            gauge_buf = create_stress_gauge(pred)
                                            gauge_base64 = get_image_base64(gauge_buf)
                                            st.markdown(
                                                f"""
                                                <div style="text-align: center;">
                                                    <img src="data:image/png;base64,{gauge_base64}" style="width: 100%;">
                                                    <div style="display: flex; justify-content: space-between; width: 100%;">
                                                        <span>Not Stressed</span>
                                                        <span>Stressed</span>
                                                    </div>
                                                </div>
                                            """,
                                                unsafe_allow_html=True,
                                            )

                                        # Show prediction result
                                        if is_stressed:
                                            st.markdown(
                                                "<h3 style='color: red; text-align: center;'>Stressed 😫</h3>",
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            st.markdown(
                                                "<h3 style='color: green; text-align: center;'>Not Stressed 😌</h3>",
                                                unsafe_allow_html=True,
                                            )

                                        st.markdown("</div>", unsafe_allow_html=True)

                                    # Save to history if enabled
                                    if save_results:
                                        timestamp = datetime.now()
                                        if save_to_history(face, pred, timestamp):
                                            st.info("Result saved to history")
                                else:
                                    st.error(f"Failed to process face #{i + 1}")
            else:
                st.info(
                    "Please upload an image or capture a frame from webcam to begin analysis"
                )

    # History Tab
    with tab2:
        st.markdown("### Detection History")

        history = load_history()
        if not history:
            st.info("No detection history found. Analyze some images to build history.")
        else:
            st.success(f"Found {len(history)} previous detections")

            # Display history entries
            for i, entry in enumerate(
                history[:10]
            ):  # Show only the 10 most recent entries
                with st.expander(f"Detection #{i + 1}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        img_path = entry["image"]
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img, use_column_width=True)

                    with col2:
                        st.text(entry["metadata"])

            if len(history) > 10:
                st.info(f"{len(history) - 10} more entries not shown")

            if st.button("Clear History"):
                try:
                    for file in os.listdir("history"):
                        os.remove(os.path.join("history", file))
                    st.success("History cleared successfully")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing history: {e}")

    # About Tab
    with tab3:
        st.markdown("""
        ### About Stress Detection System
        
        This application uses a deep learning model to detect stress levels from facial expressions.
        
        #### How it works:
        1. Upload an image or capture from webcam
        2. The system detects faces in the image
        3. Each face is analyzed by the AI model
        4. The model predicts the probability of stress
        5. Results are displayed with visual indicators
        
        #### Model Information:
        - Architecture: Convolutional Neural Network (CNN)
        - Input: 48x48 grayscale face images
        - Output: Stress probability (0-1)
        
        #### Tips for best results:
        - Ensure good lighting conditions
        - Face should be clearly visible
        - Avoid extreme angles or occlusions
        - Multiple faces can be analyzed simultaneously
        
        #### Privacy Note:
        All processing happens locally on your device. Images are not sent to external servers.
        """)

    # Footer
    st.markdown(
        "<div class='footer'>Developed by AI Research Team | © 2023</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

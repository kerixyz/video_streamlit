from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

video = cv2.VideoCapture("data/bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")


# Function to analyze a single frame using a vision-language model
def analyze_frame(frame, model):
    
    PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return model(frame)['generated_text']

# Main Streamlit app function
def main():
    st.title("Twitch Video Analysis App")
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load video using OpenCV
        video = cv2.VideoCapture(temp_video_path)
        base64_frames = []
        
        # Process video frames
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        os.remove(temp_video_path)  # Clean up temporary file
        
        st.write(f"Video uploaded successfully! Total frames: {len(base64_frames)}")
        
        # Select a frame to analyze
        frame_index = st.slider("Select a frame to analyze", 0, len(base64_frames) - 1, step=50)
        
        if st.button("Analyze Frame"):
            # Display the selected frame
            frame_data = base64.b64decode(base64_frames[frame_index])
            frame_image = Image.open(io.BytesIO(frame_data))
            st.image(frame_image, caption=f"Frame {frame_index}")
            
            # Analyze the selected frame
            model = pipeline("image-to-text")  # Replace with your specific vision-language model
            description = analyze_frame(frame_image, model)
            
            st.write("Generated Description:")
            st.write(description)
   
if __name__ == "__main__":
    main()

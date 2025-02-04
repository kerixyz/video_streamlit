import streamlit as st
import streamlink
import cv2
import os
from PIL import Image
from transformers import pipeline

# Function to download a Twitch video using streamlink
def download_twitch_video(twitch_url, output_dir='temp_videos'):
    """
    Download a Twitch video using streamlink.
    
    Args:
        twitch_url (str): URL of the Twitch video
        output_dir (str): Directory to save the downloaded video
    
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get available streams
        streams = streamlink.streams(twitch_url)
        
        if not streams:
            raise ValueError("No streams found for the given URL")
        
        # Select the best quality stream
        best_stream = streams['best']
        
        # Generate a filename
        filename = os.path.join(output_dir, f"twitch_video_{hash(twitch_url)}.mp4")
        
        # Download the stream
        with open(filename, 'wb') as f:
            for chunk in best_stream.open():
                f.write(chunk)
        
        return filename
    
    except Exception as e:
        st.error(f"Error downloading Twitch video: {e}")
        return None

# Function to extract frames from a video at specified intervals
def extract_frames(video_path, interval=30):
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path (str): Path to the video file
        interval (int): Number of frames to skip between extractions
    
    Returns:
        list: List of extracted frames as PIL Images
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        frame_count += 1

    cap.release()
    return frames

# Function to analyze a single frame using a vision-language model
def analyze_frame(frame, model):
    """
    Analyze a single frame using a vision-language model.
    
    Args:
        frame (PIL.Image): The input image/frame
        model: The vision-language model pipeline
    
    Returns:
        str: The generated description or analysis of the frame
    """
    return model(frame)['generated_text']

# Main Streamlit app function
def main():
    st.title("Twitch Video Analysis App")
    
    # Input for Twitch video URL
    twitch_url = st.text_input("Enter Twitch Video URL:")
    
    if st.button("Analyze Video"):
        if not twitch_url:
            st.error("Please provide a Twitch video URL")
            return
        
        # Download the video
        st.write("Downloading video...")
        video_path = download_twitch_video(twitch_url)
        
        if video_path:
            st.success(f"Video downloaded successfully: {video_path}")
            
            # Extract frames
            st.write("Extracting frames...")
            frames = extract_frames(video_path, interval=30)
            st.write(f"Extracted {len(frames)} frames.")
            
            # Load vision-language model
            st.write("Loading AI model...")
            vl_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
            
            # Analyze frames
            st.write("Analyzing frames...")
            for i, frame in enumerate(frames):
                st.image(frame, caption=f"Frame {i + 1}", use_column_width=True)
                analysis = analyze_frame(frame, vl_model)
                st.write(f"Analysis for Frame {i + 1}: {analysis}")

if __name__ == "__main__":
    main()

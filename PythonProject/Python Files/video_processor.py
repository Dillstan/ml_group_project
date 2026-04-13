import cv2
import os

def extract_frames_from_video(video_path, output_folder):
    """
    Takes an .mp4 video and extracts exactly 1 frame per second.
    Saves the extracted frames as .jpg files in the output folder.
    """
    print(f" Opening video file: {video_path}")
    
    # 1. Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Error: Could not open video file.")
        return []

    # 2. Get video specs (Frames Per Second)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(f"🎞️ Video FPS detected: {fps}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0
    saved_image_paths = []

    print("Extracting 1 frame per second...")
    
    # 3. Loop through the video frame by frame
    while True:
        success, frame = cap.read()
        
        # If we reach the end of the video, break the loop
        if not success:
            break
            
        # 4. The 1-FPS Math: Only save the frame if it's a multiple of the FPS
        if frame_count % fps == 0:
            # Format the filename: e.g., "frame_001.jpg", "frame_002.jpg"
            frame_name = f"frame_{saved_count:03d}.jpg"
            save_path = os.path.join(output_folder, frame_name)
            
            # Save the image to the folder
            cv2.imwrite(save_path, frame)
            saved_image_paths.append(save_path)
            
            saved_count += 1
            
        frame_count += 1

    # Clean up
    cap.release()
    print(f"✅ Extraction complete! Saved {saved_count} frames to {output_folder}")
    
    return saved_image_paths

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Point these to a real video file and output folder
    TEST_VIDEO = "Creed_3_trailer.mp4" 
    OUTPUT_DIR = "creed3_extracted_frames"
    
    # Run the extractor
    extracted_files = extract_frames_from_video(TEST_VIDEO, OUTPUT_DIR)
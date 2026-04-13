import os
import glob
import time

# Import your existing search logic from your other file
# (Adjust 'process_image' to whatever your main execution function is named)
from multithread_search import embed_face, query_database, search_image

# Note: You may need to adapt the import based on how your multithread_search.py is structured

def process_video_directory(frames_folder):
    print(f"📂 Scanning folder: {frames_folder}")
    
    # Grab all the .jpg files in the folder and sort them numerically
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
    
    if not frame_paths:
        print("❌ No frames found! Check your folder path.")
        return

    print(f"🚀 Found {len(frame_paths)} frames. Starting pipeline...")
    
    start_time = time.time()

    # Loop through every frame and run your existing multithreaded logic
    for frame_path in frame_paths:
        print(f"\n--- Analyzing {os.path.basename(frame_path)} ---")
        
        # Here is where you call the main logic from multithread_search.py
        # Example: 
        # results = run_multithreaded_pipeline(frame_path)
        # print(results)
        search_image(frame_path)
       

    end_time = time.time()
    print(f"\n✅ Video Pipeline Complete! Processed {len(frame_paths)} frames in {end_time - start_time:.2f} seconds.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    FRAMES_DIR = "creed3_extracted_frames"
    process_video_directory(FRAMES_DIR)
# Import the specific functions from your new files
from step1_tmdb import convert_imdb_to_tmdb
from step3_tmdb import calculate_release_range
from dotenv import load_dotenv
import os 

if __name__ == "__main__":
    load_dotenv()  

    API_KEY = os.getenv("API_KEY")

    # MOCK DATA: Creed III Trio
    mock_imdb_ids = ["nm0430107", "nm1935086", "nm8244669"]
    mock_predicted_ages = [37, 38, 35] 
    
    print("--- Starting Algorithm Test ---")
    
    # Run Step 1 (Called from step1_tmdb.py)
    tmdb_ids = convert_imdb_to_tmdb(mock_imdb_ids, API_KEY)
    print(f"Step 1 Output (TMDb IDs): {tmdb_ids}")
    
    # --- DUMMY STEP 2 ---
    step_2_output_years = [2024, 2021, 2024] 
    print(f"Step 2 Output (Estimated Years): {step_2_output_years}")
    
    # Run Step 3 (Called from step3_release.py)
    year_range = calculate_release_range(step_2_output_years, error_margin=2)
    print(f"Step 3 Output (Target Search Range): {year_range[0]} to {year_range[1]}")
import math

def calculate_release_range(estimated_release_years, error_margin=3):
    """
    Takes the array of years calculated in Step 2.
    1-5 Actors: Range search (Min year to Max year).
    6+ Actors: Mean release date search.
    """
    num_actors = len(estimated_release_years)
    
    if num_actors == 0:
        return None

    if 1 <= num_actors <= 5:
        start_year = min(estimated_release_years) - error_margin
        end_year = max(estimated_release_years) + error_margin
        print(f" [1-5 Actors] Executing Range Search...")
        return (start_year, end_year)
        
    else:
        mean_year = sum(estimated_release_years) / num_actors
        start_year = math.floor(mean_year) - error_margin
        end_year = math.ceil(mean_year) + error_margin
        print(f" [6+ Actors] Executing Mean Release Search...")
        return (start_year, end_year)
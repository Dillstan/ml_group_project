#This was mainly created by GEMINI to help me create a helper that would restart my script if it failed overnight except in certain circumstances

import subprocess
import time
import os

WORKER_SCRIPT = "manual_create_db.py"
LOG_FILE = "./logs/db_reg_progress_intersection"
DEFAULT_START_INDEX = 0


def get_last_successful_index():
    """
    Reads the log file to find the last '100th' checkpoint saved.
    """
    if not os.path.exists(LOG_FILE):
        return DEFAULT_START_INDEX

    try:
        with open(LOG_FILE, 'r') as f:
            # Read all lines
            lines = f.readlines()

            # Read backwards to find the last valid entry
            for line in reversed(lines):
                parts = line.split('-')
                # We expect lines like: "52700 - 2023-10-27 10:00:00"
                if len(parts) >= 2:
                    try:
                        # Extract the number before the dash
                        last_index = int(parts[0].strip())
                        return last_index
                    except ValueError:
                        continue  # Skip init messages or headers
        return DEFAULT_START_INDEX
    except Exception as e:
        print(f"Watchdog Error reading log: {e}")
        return DEFAULT_START_INDEX


def main():
    print("--- WATCHDOG ACTIVATED ---")

    while True:
        start_idx = get_last_successful_index()
        print(f"\n[Watchdog] Launching worker at line {start_idx}...")

        try:
            process = subprocess.run(
                ["python3", WORKER_SCRIPT, str(start_idx)],
                check=False
            )
            exit_code = process.returncode

        except KeyboardInterrupt:
            print("\n[Watchdog] Stopping...")
            break

        # --- NEW LOGIC HERE ---
        if exit_code == 0:
            print("[Watchdog] Worker finished successfully!")
            break

        elif exit_code == 77:
            print("========================================")
            print("[Watchdog] STOPPED: DISK IS FULL.")
            print("Free up space and restart watchdog manually.")
            print("========================================")
            # playsound('alarm.mp3') # Optional: Wake you up
            break

        else:
            print(f"[Watchdog] Worker crashed with code {exit_code}.")
            print("[Watchdog] Restarting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    main()
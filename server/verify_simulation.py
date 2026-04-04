import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from app import run_full_episode
import pandas as pd

def test_full_run():
    print("Starting verification run...")
    # 20 clients, 10 rounds, medium difficulty, poisoners [3, 7, 12]
    result, df, fig = run_full_episode([3, 7, 12], 20, 10, "Medium")
    
    print("\n--- EPISODE SUMMARY ---")
    print(result)
    
    print("\n--- HISTORY HEAD ---")
    print(df.head(10))
    
    # Check if Round 10 exists
    if len(df) == 10:
        print("\nSUCCESS: All 10 rounds completed.")
    else:
        print(f"\nFAILURE: Only {len(df)} rounds completed.")

    # Check if Graders are non-zero
    if "0.000" not in result:
         print("SUCCESS: Graders returned non-zero values (likely).")
    else:
         print("WARNING: Some graders still returned 0.000. Checking accuracy...")
         
if __name__ == "__main__":
    test_full_run()

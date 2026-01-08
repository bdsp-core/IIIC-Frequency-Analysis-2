"""
Fix annotation CSV files by converting absolute paths to filenames only.

This script processes all annotation CSV files and extracts just the filename
from the full path, making the files portable across different systems.
"""

import pandas as pd
from pathlib import Path
import os

# Robust path handling
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == 'code' else script_dir

# Define paths
annotations_dir = repo_root / 'data' / 'annotations'

# Verify annotations directory exists
if not annotations_dir.exists():
    raise FileNotFoundError(
        f"Annotations directory not found: {annotations_dir}\n"
        f"Please ensure the data/annotations folder exists."
    )

# Find all annotation CSV files
annotation_files = sorted(list(annotations_dir.glob('*.csv')))

if not annotation_files:
    print(f"No CSV files found in {annotations_dir}")
    exit(1)

print("="*70)
print("Fixing Annotation File Paths")
print("="*70)
print(f"\nFound {len(annotation_files)} annotation files")

for csv_file in annotation_files:
    print(f"\nProcessing: {csv_file.name}")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if 'files' column exists
        if 'files' not in df.columns:
            print(f"  Warning: No 'files' column found. Skipping.")
            continue

        # Show sample of original paths
        print(f"  Original path example: {df['files'].iloc[0]}")

        # Extract just the filename from full path
        # Remove .png extension if present, then add .mat
        df['files'] = df['files'].apply(lambda x: Path(x).stem.replace('_score', '') + '.mat')

        # Show sample of fixed paths
        print(f"  Fixed filename example: {df['files'].iloc[0]}")

        # Save back to CSV
        df.to_csv(csv_file, index=False)
        print(f"  ✓ Updated {len(df)} rows")

    except Exception as e:
        print(f"  ✗ Error processing {csv_file.name}: {e}")

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"\nAll annotation files have been updated in: {annotations_dir}")
print("\nThe 'files' column now contains just the .mat filename,")
print("making the annotations portable across different systems.")

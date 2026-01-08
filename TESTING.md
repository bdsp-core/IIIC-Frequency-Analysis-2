# Testing Guide

This document describes how to test the repository after setting up.

## Prerequisites

1. **Environment setup** - Ensure you've created the conda environment:
```bash
conda env create -f code/environment.yml
conda activate foe
```

2. **Data downloaded** - Dataset must be downloaded from BDSP (see [DATASET_INFO.md](DATASET_INFO.md))

## Step 1: Fix Annotation Paths (One-Time Setup)

After downloading the dataset, run this script once to fix the annotation CSV files:

```bash
cd code
python fix_annotation_paths.py
```

**What this does:**
- Converts absolute paths in annotation CSVs to relative filenames
- Makes annotations portable across different systems
- Required before running analysis scripts

**Expected output:**
```
======================================================================
Fixing Annotation File Paths
======================================================================

Found 12 annotation files

Processing: GPDS_LB_2_2025.csv
  Original path example: /Users/someone/Desktop/gpd/file_score.png
  Fixed filename example: file.mat
  ✓ Updated 296 rows
...
```

## Step 2: Test Detector Functions

Test that all 6 detector variants work correctly:

```bash
cd code
python test_detectors.py
```

**What this tests:**
- ✓ File loading (MATLAB .mat files)
- ✓ RDA detectors: rda1a_fft, rda1b_fft, rda2_hht
- ✓ PD detectors: pd_detect, pd_detect_alternate (apd), pd_detect_alternate (zscore)
- ✓ Output format validation

**Expected output:**
```
======================================================================
Testing EEG Detector Functions
======================================================================

----------------------------------------------------------------------
Testing RDA Detectors
----------------------------------------------------------------------

Loading LRDA sample: abn514_20130114_210759_3843_seg_5.mat
Segment shape: (19, 10000)

  Testing rda1a_fft...
    ✓ rda1a_fft passed
      Event type: LRDA
      Frequency: 1.52 Hz
      Spatial extent: 0.67
      Spatial areas: ['LF', 'LT', 'LO']

  Testing rda1b_fft...
    ✓ rda1b_fft passed
...

Test Summary
----------------------------------------------------------------------

Total tests run: 6
Passed: 6
Failed: 0

✓ All tests passed! The detectors are working correctly.
```

## Step 3: Test Visualization

Test visualization script on sample files:

```bash
cd code
python visualize_output.py
```

**What this does:**
- Loads one LRDA and one LPD file
- Runs detection algorithms
- Displays results with matplotlib

**Note:** This opens matplotlib windows. Close each window to proceed.

## Step 4: Run Full Analysis (Optional)

⚠️ **Warning:** This processes ~1,000+ files and takes 2-4 hours

```bash
cd code
python extract_frequency_spatial_extent.py
```

**What this does:**
- Processes all EEG files for each event type (GPD, LPD, GRDA, LRDA)
- Runs all detector variants
- Saves results to `results/*.csv`

**Monitor progress:**
```
**********************************************************************
LRDA
**********************************************************************
Processing LRDA: 100%|████████████| 212/212 [05:23<00:00,  1.57s/it]
Results saved to: .../results/lrda_results.csv
```

**Expected outputs:**
- `results/gpd_results.csv` (~296 rows)
- `results/lpd_results.csv` (~271 rows)
- `results/grda_results.csv` (~287 rows)
- `results/lrda_results.csv` (~212 rows)

## Step 5: Run IRR Analysis (Optional)

Test the inter-rater reliability notebooks:

```bash
cd code
jupyter notebook
```

Then open and run:
- `irr_analysis_onagreement.ipynb` - Analysis on agreement subset
- `irr_analysis_fulldataset.ipynb` - Full dataset analysis

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fooof'`

**Solution:** Activate conda environment first
```bash
conda activate foe
```

### File Not Found Errors

**Problem:** `Data directory not found`

**Solution:** Ensure data is in correct location
```bash
# Expected structure:
IIIC-Frequency-Analysis-2/
├── code/
└── data/
    ├── dataset_eeg/
    │   ├── gpd/
    │   ├── lpd/
    │   ├── grda/
    └── lrda/
    └── annotations/
```

### Path Errors in Annotations

**Problem:** Annotations reference `/Users/someone/...` paths

**Solution:** Run fix_annotation_paths.py
```bash
cd code
python fix_annotation_paths.py
```

### Detector Crashes

**Problem:** Detector throws error on specific file

**Expected behavior:** Script should catch error and continue with next file
```python
try:
    mat = load_mat_file(str(file_path))
except Exception as e:
    print(f"Failed to load {filename}: {e}")
    continue  # Skip to next file
```

If script stops completely, please report issue with:
- File name that caused error
- Complete error traceback
- Python version and OS

## Quick Verification Checklist

Before running full analysis, verify:

- [ ] Conda environment activated (`conda activate foe`)
- [ ] Data directory exists and contains .mat files
- [ ] `fix_annotation_paths.py` has been run
- [ ] `test_detectors.py` passes all 6 tests
- [ ] `visualize_output.py` displays figures without errors

If all checks pass, you're ready to run the full analysis!

## Performance Benchmarks

Approximate run times on typical hardware:

| Script | Files Processed | Approx. Time |
|--------|----------------|--------------|
| `fix_annotation_paths.py` | 12 CSVs | < 1 second |
| `test_detectors.py` | 2 files | 30-60 seconds |
| `visualize_output.py` | 2 files | 30-60 seconds |
| `extract_frequency_spatial_extent.py` | ~1,068 files | 2-4 hours |

**Hardware tested:**
- MacBook Pro M1/M2: ~2 hours
- Linux workstation: ~2.5 hours
- Typical laptop: ~3-4 hours

## Reproducing Paper Results

To exactly reproduce the paper results:

1. Download the exact dataset version from BDSP
2. Run `fix_annotation_paths.py`
3. Run `extract_frequency_spatial_extent.py`
4. Run both IRR analysis notebooks
5. Compare results with paper figures/tables

The best-performing algorithms are:
- **RDA (LRDA/GRDA):** `rda1b_fft`
- **PD (LPD/GPD):** `pd_detect_alternate` with `pk_detect='apd'`

Results from these variants should match the paper's reported metrics.

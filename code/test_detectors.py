"""
Test script to verify all detector functions work correctly.

Tests each detector on a single sample file to ensure:
1. Files load correctly
2. Detectors run without errors
3. Output format is as expected
"""

import sys
from pathlib import Path
import numpy as np
import hdf5storage
import h5py

# Import detectors
import rda_detector as rda
import pd_detector as pddet
import pd_detector_alternate as pddeta

# Robust path handling
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == 'code' else script_dir
data_dir = repo_root / 'data' / 'dataset_eeg'

def load_mat_file(filepath):
    """Load MATLAB file, handling both v7.3 and earlier versions."""
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError as e:
        if 'HDF reader for matlab v7.3 files' in str(e):
            with h5py.File(filepath,'r') as f:
                return {key: f[key][()] for key in f.keys()}
        else:
            raise

def test_detector(detector_func, detector_name, segment, fs, *args, **kwargs):
    """Test a detector function and validate output."""
    print(f"\n  Testing {detector_name}...")
    try:
        result = detector_func(segment, fs, *args, **kwargs)

        # Check if result is a dictionary
        if not isinstance(result, dict) and not isinstance(result, tuple):
            print(f"    ✗ Expected dict output, got {type(result)}")
            return False

        # Handle tuple returns (some detectors return extra info)
        if isinstance(result, tuple):
            result = result[0]

        # Validate required keys
        required_keys = ['type_event', 'event_frequency', 'spatial_extent', 'spatial_areas']
        for key in required_keys:
            if key not in result:
                print(f"    ✗ Missing required key: {key}")
                return False

        # Print results
        print(f"    ✓ {detector_name} passed")
        print(f"      Event type: {result['type_event']}")
        print(f"      Frequency: {result['event_frequency']:.2f} Hz")
        print(f"      Spatial extent: {result['spatial_extent']:.2f}")
        print(f"      Spatial areas: {result['spatial_areas']}")

        return True

    except Exception as e:
        print(f"    ✗ {detector_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("Testing EEG Detector Functions")
    print("="*70)

    # Verify data directory
    if not data_dir.exists():
        print(f"\n✗ Data directory not found: {data_dir}")
        print("Please download the dataset first. See DATASET_INFO.md")
        sys.exit(1)

    fs = 200  # Sampling frequency
    passed = 0
    failed = 0

    # Test RDA detectors
    print("\n" + "-"*70)
    print("Testing RDA Detectors")
    print("-"*70)

    lrda_dir = data_dir / 'lrda'
    if lrda_dir.exists():
        lrda_files = sorted(list(lrda_dir.glob('*.mat')))
        if lrda_files:
            print(f"\nLoading LRDA sample: {lrda_files[0].name}")

            try:
                mat = load_mat_file(str(lrda_files[0]))
                try:
                    segment = mat['data']
                except KeyError:
                    segment = mat['data_50sec']

                print(f"Segment shape: {segment.shape}")

                # Test RDA detectors
                if test_detector(rda.rda1a_fft, "rda1a_fft", segment, fs):
                    passed += 1
                else:
                    failed += 1

                if test_detector(rda.rda1b_fft, "rda1b_fft", segment, fs, 0):
                    passed += 1
                else:
                    failed += 1

                if test_detector(rda.rda2_hht, "rda2_hht", segment, fs, 1):
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"✗ Failed to load LRDA file: {e}")
                failed += 3
        else:
            print(f"✗ No .mat files found in {lrda_dir}")
            failed += 3
    else:
        print(f"✗ LRDA directory not found: {lrda_dir}")
        failed += 3

    # Test PD detectors
    print("\n" + "-"*70)
    print("Testing PD Detectors")
    print("-"*70)

    lpd_dir = data_dir / 'lpd'
    if lpd_dir.exists():
        lpd_files = sorted(list(lpd_dir.glob('*.mat')))
        if lpd_files:
            print(f"\nLoading LPD sample: {lpd_files[0].name}")

            try:
                mat = load_mat_file(str(lpd_files[0]))
                try:
                    segment = mat['data']
                except KeyError:
                    segment = mat['data_50sec']

                print(f"Segment shape: {segment.shape}")

                # Test PD detectors
                if test_detector(pddet.pd_detect, "pd_detect", segment, fs):
                    passed += 1
                else:
                    failed += 1

                if test_detector(pddeta.pd_detect_alternate, "pd_detect_alternate (apd)",
                                segment, fs, pk_detect='apd'):
                    passed += 1
                else:
                    failed += 1

                if test_detector(pddeta.pd_detect_alternate, "pd_detect_alternate (zscore)",
                                segment, fs, pk_detect='zscore'):
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"✗ Failed to load LPD file: {e}")
                failed += 3
        else:
            print(f"✗ No .mat files found in {lpd_dir}")
            failed += 3
    else:
        print(f"✗ LPD directory not found: {lpd_dir}")
        failed += 3

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"\nTotal tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ All tests passed! The detectors are working correctly.")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

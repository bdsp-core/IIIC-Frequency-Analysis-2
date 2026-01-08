# Dataset Information

## Overview

This repository uses EEG data from the Brain Data Science Platform (BDSP). The data is **not included** in this repository due to size and privacy considerations.

## Dataset Specifications

### EEG Files
- **Total segments**: ~1,000+ segments
- **Format**: MATLAB `.mat` files
- **Duration**: 50 seconds per segment
- **Sampling rate**: 200 Hz
- **Channels**: 8 brain regions (bilateral frontal, temporal, central-parietal, occipital)
- **Variable name**: `data` or `data_50sec` (19-channel array × time samples)

### File Organization
```
data/dataset_eeg/
├── gpd/      # ~298 files - Generalized Periodic Discharges
├── lpd/      # ~271 files - Lateralized Periodic Discharges
├── grda/     # ~287 files - Generalized Rhythmic Delta Activity
└── lrda/     # ~212 files - Lateralized Rhythmic Delta Activity
```

### Annotation Files
```
data/annotations/
├── GPDS_LB_2_2025.csv   # Annotator LB - GPD
├── GPDS_PH_3_2025.csv   # Annotator PH - GPD
├── GPDS_SZ_3_2025.csv   # Annotator SZ - GPD
├── GRDA_LB_2_2025.csv   # Annotator LB - GRDA
├── GRDA_PH_3_2025.csv   # Annotator PH - GRDA
├── GRDA_SZ_3_2025.csv   # Annotator SZ - GRDA
├── LPDS_LB_2_2025.csv   # Annotator LB - LPD
├── LPDS_PH_3_3025.csv   # Annotator PH - LPD
├── LPDS_SZ_3_2025.csv   # Annotator SZ - LPD
├── LRDA_LB_2_2025.csv   # Annotator LB - LRDA
├── LRDA_PH_3_2025.csv   # Annotator PH - LRDA
└── LRDA_SZ_3_2025.csv   # Annotator SZ - LRDA
```

### Annotation Format
Each CSV file contains expert annotations with columns:
- `files`: Path to corresponding EEG file
- `frequency`: Expert-rated frequency (Hz)
- `spatial`: Spatial extent (proportion, 0-1)
- `spatial_area`: Space-separated list of affected brain regions

## Accessing the Data

### Step 1: Register for Access
1. Visit the Brain Data Science Platform: [https://bdsp.io](https://bdsp.io)
2. Create an account or sign in
3. Navigate to the dataset access request page
4. Fill out the data access request form, including:
   - Your research affiliation
   - Intended use of the data
   - IRB approval information (if applicable)

### Step 2: Data Use Agreement
- Review and sign the BDSP Data Use Agreement
- Agree to terms regarding:
  - Data confidentiality
  - No attempts to re-identify patients
  - Proper data security practices
  - Citation requirements

### Step 3: Download via AWS S3
Once approved (typically 1-2 business days), you will receive:
- AWS S3 bucket access credentials
- Access point URL
- Download instructions

**Example download using AWS CLI** (after receiving credentials):
```bash
# Install AWS CLI if needed
pip install awscli

# Configure credentials (provided by BDSP)
aws configure

# Download the dataset
aws s3 sync s3://[BUCKET-NAME]/IIIC-Frequency-Analysis/ ./data/ --region us-east-1
```

**Expected download size**: ~10-15 GB

### Step 4: Verify Data Structure
After downloading, verify your directory structure matches:
```bash
cd IIIC-Frequency-Analysis-2
ls -R data/
```

You should see:
- `data/dataset_eeg/gpd/` with ~298 `.mat` files
- `data/dataset_eeg/lpd/` with ~271 `.mat` files
- `data/dataset_eeg/grda/` with ~287 `.mat` files
- `data/dataset_eeg/lrda/` with ~212 `.mat` files
- `data/annotations/` with 12 CSV files

## Data Privacy and Ethics

This dataset contains de-identified EEG recordings collected under IRB approval. Users must:
- ✓ Maintain strict data confidentiality
- ✓ Store data securely with appropriate access controls
- ✓ Not attempt to re-identify subjects
- ✓ Use data only for approved research purposes
- ✓ Delete data when research is complete (per agreement)
- ✓ Cite the dataset appropriately in publications

## Troubleshooting

### Issue: Cannot access BDSP website
- Ensure you're accessing [https://bdsp.io](https://bdsp.io) (not http)
- Try a different browser
- Contact BDSP support: support@bdsp.io

### Issue: Data request pending for >3 days
- Check your email spam folder for approval notification
- Contact BDSP support with your request ID

### Issue: AWS download fails
- Verify AWS credentials are correctly configured
- Check internet connection stability
- Ensure sufficient disk space (~20 GB free)
- Try downloading in smaller batches by event type

### Issue: Cannot load `.mat` files
- Verify Python environment has `hdf5storage` and `h5py` installed
- Check MATLAB file version compatibility
- See `code/extract_frequency_spatial_extent.py` for loading example

## Alternative Access Methods

If AWS CLI download is not feasible, BDSP may provide alternative methods:
- Direct download links (for smaller subsets)
- BDSP compute environment with pre-loaded data
- Globus file transfer

Contact BDSP support to discuss alternatives.

## Contact

For data access questions:
- **BDSP Support**: support@bdsp.io
- **BDSP Website**: [https://bdsp.io](https://bdsp.io)

For code/analysis questions:
- Open an issue on this GitHub repository

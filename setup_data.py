#!/usr/bin/env python3
"""
AutoML Data Setup - Automatic Download and Extract
=================================================

This script automatically downloads and sets up the required datasets
for the AutoML text classification pipeline.

Usage:
    python setup_data.py [--data-path DATA_DIR]

Example:
    python setup_data.py --data-path data
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import argparse
import tempfile
import shutil
import pandas as pd


def download_file(url: str, filename: str, description: str = ""):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"File: {filename}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            bar_length = 40
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r   Progress: |{bar}| {percent:3d}% [{block_num * block_size:,}/{total_size:,} bytes]', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print()  # New line after progress bar
        print(f"Downloaded {description} successfully!")
        return True
    except Exception as e:
        print(f"\n Failed to download {description}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str, description: str = ""):
    """Extract a zip file."""
    print(f"Extracting {description}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {description} successfully!")
        return True
    except Exception as e:
        print(f"Failed to extract {description}: {e}")
        return False


def verify_data_structure(data_path: Path):
    """Verify that all required datasets are present."""
    required_datasets = ['amazon', 'ag_news', 'dbpedia', 'imdb', 'yelp']
    required_files = ['train.csv', 'test.csv']
    
    print("🔍 Verifying data structure...")
    
    missing = []
    for dataset in required_datasets:
        dataset_dir = data_path / dataset
        if not dataset_dir.exists():
            missing.append(f"Missing directory: {dataset_dir}")
            continue
            
        for file in required_files:
            file_path = dataset_dir / file
            if not file_path.exists():
                missing.append(f"Missing file: {file_path}")
    
    if missing:
        print("Data structure verification failed:")
        for item in missing:
            print(f"   • {item}")
        return False
    
    print("Data structure verification passed!")
    return True


def setup_automl_data(data_path: str = "data", force_download: bool = False):
    """
    Main function to setup AutoML datasets.
    
    Args:
        data_path: Directory where data should be stored
        force_download: Whether to re-download even if data exists
    """
    data_dir = Path(data_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("AutoML Data Setup Starting...")
    print(f"Data directory: {data_dir.absolute()}")
    
    # Check if data already exists (unless forced)
    if not force_download:
        required_datasets = ['amazon', 'ag_news', 'dbpedia', 'imdb', 'yelp']
        if all((data_dir / dataset).exists() for dataset in required_datasets):
            if verify_data_structure(data_dir):
                print("All datasets already present and valid!")
                return True
            else:
                print(" Data exists but structure is invalid. Re-downloading...")
    
    # URLs for the datasets
    phase1_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase1.zip"
    phase2_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-text/text-phase2.zip"
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download Phase 1 data (amazon, ag_news, dbpedia, imdb)
        phase1_zip = temp_path / "phase1.zip"
        if not download_file(phase1_url, str(phase1_zip), "Phase 1 data (4 datasets)"):
            return False
            
        # Download Phase 2 data (yelp)
        phase2_zip = temp_path / "phase2.zip"  
        if not download_file(phase2_url, str(phase2_zip), "Phase 2 data (yelp dataset)"):
            return False
        
        # Extract Phase 1
        if not extract_zip(str(phase1_zip), str(data_dir), "Phase 1 datasets"):
            return False
            
        # Extract Phase 2 to temporary location first
        temp_phase2_dir = temp_path / "phase2_temp"
        if not extract_zip(str(phase2_zip), str(temp_phase2_dir), "Phase 2 dataset"):
            return False
        
        # Move yelp folder from text-phase2 to correct location
        phase2_extracted = temp_phase2_dir / "text-phase2" / "yelp"
        if phase2_extracted.exists():
            import shutil
            target_yelp = data_dir / "yelp"
            if target_yelp.exists():
                shutil.rmtree(target_yelp)
            shutil.move(str(phase2_extracted), str(target_yelp))
            print("Moved yelp dataset to correct location")
        else:
            print(" Could not find yelp dataset in extracted Phase 2 data")
            return False
    
    # Verify everything is in place
    if verify_data_structure(data_dir):
        print("\n Data setup completed successfully!")
        print(f"\nYou can now run the AutoML pipeline:")
        print(f"   python run.py --data-path {data_path}")
        return True
    else:
        print("\n Data setup failed - verification unsuccessful")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download and setup AutoML datasets")
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data",
        help="Directory to store datasets (default: data)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if data exists"
    )
    
    args = parser.parse_args()
    
    success = setup_automl_data(args.data_path, args.force)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
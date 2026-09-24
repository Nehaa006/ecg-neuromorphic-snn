import os
import wfdb


def download_mitdb(download_dir: str = "data/raw"):
    """Downloads the PhysioNet MIT-BIH Arrhythmia Database directly to local directory."""
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading MIT-BIH dataset to {download_dir}...")
    
    wfdb.dl_database("mitdb", dl_dir=download_dir)
    print("Download complete!")


if __name__ == "__main__":
    download_mitdb()
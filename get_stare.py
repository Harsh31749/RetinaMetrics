import os
import ssl
import urllib.request
import tarfile
import gzip
import shutil

# Bypass university SSL certificate check
ssl._create_default_https_context = ssl._create_unverified_context

# Create clean folders
for folder in ["data/images", "data/labels", "data/clinical", "outputs"]:
    os.makedirs(folder, exist_ok=True)

FILES_TO_DOWNLOAD = [
    ("stare-images.tar", "https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar", "data/images"),
    ("labels-ah.tar", "https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar", "data/labels"),
]

EHR_URL = "https://cecas.clemson.edu/~ahoover/stare/diagnoses/all-mg-codes.txt"

def download_file_live(url, save_path):
    """Downloads a file in 64KB chunks with a live MB counter in the terminal."""
    with urllib.request.urlopen(url) as response, open(save_path, "wb") as out_file:
        downloaded_bytes = 0
        while True:
            chunk = response.read(65536)  # 64 KB chunk
            if not chunk:
                break
            out_file.write(chunk)
            downloaded_bytes += len(chunk)
            mb = downloaded_bytes / (1024 * 1024)
            print(f"\r   -> Downloaded: {mb:.2f} MB", end="", flush=True)
    print()

def unpack_tar_and_gz(tar_path, target_dir):
    """Extracts the .tar archive and converts all .ppm.gz files into .ppm images."""
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)

    count = 0
    for root, _, files in os.walk(target_dir):
        for fname in files:
            if fname.endswith(".gz"):
                gz_path = os.path.join(root, fname)
                ppm_path = os.path.join(target_dir, fname[:-3])
                with gzip.open(gz_path, "rb") as f_in, open(ppm_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                count += 1
    return count

if __name__ == "__main__":
    print("=== RetinaMetrics Step 1: Fetching STARE Dataset (~19 MB) ===")

    for filename, url, target_dir in FILES_TO_DOWNLOAD:
        print(f"\nDownloading {filename}...")
        download_file_live(url, filename)
        print("   -> Unpacking images...")
        total = unpack_tar_and_gz(filename, target_dir)
        print(f"   -> Saved {total} files in '{target_dir}/'")

    print("\nDownloading STARE Clinical Diagnosis Records (all-mg-codes.txt)...")
    download_file_live(EHR_URL, "data/clinical/all-mg-codes.txt")

    print("\n=== Step 1 Complete! Check your 'data/' folder in VS Code. ===")
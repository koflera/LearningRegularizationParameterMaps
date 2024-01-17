"""Download MOT17 data."""
import os
import time
from urllib.request import urlretrieve
import zipfile
import argparse
import shutil

def download(directory):
    
    destination = os.path.join(directory, "")
    os.makedirs(destination, exist_ok=True)

    download_path = os.path.join(destination, "download")
    file_name = "/data.zip"

    # Download data if zip-file does not already exist
    if not os.path.exists(download_path+file_name):
        print("downloading...")
        os.makedirs(download_path, exist_ok=True)
        data_path, msg = urlretrieve("https://motchallenge.net/data/MOT17Det.zip", download_path+file_name, report_hook)
        print(msg)
    else:
        data_path = download_path+file_name
        print("data already downloaded.")

    # Extract data
    print("unzipping data...")
    with zipfile.ZipFile(data_path, "r") as zip_file:
        zip_file.extractall(path=download_path)

    # Organize extracted data
    if not len(os.listdir(destination)) == 15:
        print("moving files to destination...")
        for subset in ["train/", "test/"]:
            subset_path = os.path.join(download_path, subset)
            for folder_name in os.listdir(os.path.join(download_path, subset)):
                shutil.move(os.path.join(subset_path, folder_name), os.path.join(destination, folder_name))

    # Clean up destination
    shutil.rmtree(os.path.join(download_path, "test"))
    shutil.rmtree(os.path.join(download_path, "train"))

    print("done!")


def report_hook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    print(f">> {percent} %, {speed} MB, KB/s, {round(duration)} seconds", end="\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download MOT17")
    parser.add_argument("path", help="output path")
    args = parser.parse_args()
    download(args.path)

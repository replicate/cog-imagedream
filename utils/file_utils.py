import os
import subprocess
import tarfile

def download_and_extract(url, dest):
    tmp_path = "tmp.tar"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        print(f"Downloading {url}...")
        output = subprocess.check_output(["pget", url, tmp_path])
    except subprocess.CalledProcessError as e:
        # If download fails, clean up and re-raise exception
        raise e
    tar = tarfile.open(tmp_path)
    tar.extractall(path=dest)
    tar.close()
    os.remove(tmp_path)

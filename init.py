import requests
from pathlib import Path
import tarfile
import shutil
import os

# Change working directory to the parent directory of the script
os.chdir(Path(__file__).parent)

def download_fastfetch():
    if not Path("./external/fastfetch").exists():
        print("Downloading fastfetch...")
        root = "fastfetch-linux-amd64"
        fastfetch = requests.get(f"https://github.com/fastfetch-cli/fastfetch/releases/download/2.41.0/{root}.tar.gz")

        tar_path = Path("./external/fastfetch.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(fastfetch.content)
        temp_extract_dir = Path("/tmp/fastfetch")
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(temp_extract_dir)

        # Move executable to external directory
        shutil.move(temp_extract_dir / root / 'usr' / 'bin' / 'fastfetch', "./external/fastfetch")
        os.chmod("./external/fastfetch", 0o755)

        # Clean up
        os.remove(Path("./external/fastfetch.tar.gz"))
    else:
        print("fastfetch already downloaded")

def call_fastfetch():
    output = os.popen("./external/fastfetch -l none").read()
    print(output[:-146])
    return output

download_fastfetch()
call_fastfetch()
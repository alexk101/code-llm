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
            tar.extractall(temp_extract_dir, filter='data')

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


def download_html_to_md():
    if not Path("./external/html2markdown").exists():
        target = "https://github.com/JohannesKaufmann/html-to-markdown/releases/download/v2.3.2/html-to-markdown_Linux_x86_64.tar.gz"
        response = requests.get(target)
        temp_dir = Path("/tmp")
        with open(temp_dir / "html-to-markdown_Linux_x86_64.tar.gz", "wb") as f:
            f.write(response.content)
        with tarfile.open(temp_dir / "html-to-markdown_Linux_x86_64.tar.gz", "r:gz") as tar:
            tar.extractall(temp_dir, filter='data')
        shutil.move(temp_dir / "html2markdown", "./external/html2markdown")
    else:
        print("html2markdown already downloaded")


def main():
    download_fastfetch()
    call_fastfetch()
    download_html_to_md()


if __name__ == "__main__":
    main()
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests

# Change working directory to the parent directory of the script
os.chdir(Path(__file__).parent)


def download_fastfetch():
    if not Path("./external/binaries/fastfetch").exists():
        # Get platform
        platform = os.uname().machine
        build = None
        if platform == "x86_64":
            build = "linux-amd64"
        elif platform == "arm64":
            build = "macos-universal"
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        print("Downloading fastfetch...")
        root = f"fastfetch-{build}"
        fastfetch = requests.get(
            f"https://github.com/fastfetch-cli/fastfetch/releases/download/2.41.0/{root}.tar.gz"
        )

        tar_path = Path("./external/fastfetch.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(fastfetch.content)
        temp_extract_dir = Path("/tmp/fastfetch")
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(temp_extract_dir, filter="data")

        # Move executable to external directory
        shutil.move(
            temp_extract_dir / root / "usr" / "bin" / "fastfetch",
            "./external/binaries/fastfetch",
        )
        os.chmod("./external/binaries/fastfetch", 0o755)

        # Clean up
        os.remove(Path("./external/fastfetch.tar.gz"))
    else:
        print("fastfetch already downloaded")


def call_fastfetch():
    output = subprocess.check_output(
        ["./external/binaries/fastfetch", "-l", "none"], text=True
    )
    print(output[:-146])
    return output


def download_html_to_md():
    if not Path("./external/binaries/html2markdown").exists():
        # Get platform
        platform = os.uname().machine
        build = None
        if platform == "x86_64":
            build = "Linux_x86_64"
        elif platform == "arm64":
            build = "Darwin_arm64"
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        print("Downloading html2markdown...")
        target = f"https://github.com/JohannesKaufmann/html-to-markdown/releases/download/v2.3.2/html-to-markdown_{build}.tar.gz"
        response = requests.get(target)
        temp_dir = Path("/tmp")
        with open(temp_dir / f"html-to-markdown_{build}.tar.gz", "wb") as f:
            f.write(response.content)
        with tarfile.open(temp_dir / f"html-to-markdown_{build}.tar.gz", "r:gz") as tar:
            tar.extractall(temp_dir, filter="data")
        shutil.move(temp_dir / "html2markdown", "./external/binaries/html2markdown")
    else:
        print("html2markdown already downloaded")


def main():
    binaries = Path("./external/binaries")
    binaries.mkdir(parents=True, exist_ok=True)
    download_fastfetch()
    call_fastfetch()
    download_html_to_md()


if __name__ == "__main__":
    main()

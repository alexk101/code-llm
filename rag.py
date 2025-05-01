from pathlib import Path
import requests
import tarfile
import subprocess
import shutil
from tqdm import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

lang_ref = {
    "fortran": [
        {
            "name": "Fortran stdlib",
            "docs": Path("external/stdlib/doc"),
            "resource": None,
            "kind": "markdown",
            "get": None
        },
    ],
    "python": [
        {
            "name": "Python 3.12 Documentation",
            "docs": Path("external/python-3.12-docs-text"),
            "resource": "https://docs.python.org/{python_version}/archives/{file_name}",
            "resource_args": {
                "python_version": "3.12",
                "file_name": "python-{python_version}-docs-text.tar.bz2"
            },
            "kind": "text",
            "get": None
        },
    ],
    "cpp": [
        {
            "name": "C++ Documentation",
            "docs": Path("external/cppreference-doc-20250209/reference/en.cppreference.com/w/cpp"),
            "resources": Path("external/cppreference-doc-20250209/reference/en.cppreference.com/w/cpp/resources"),
            "kind": "http",
        }
    ],
    "c": [
        {
            "name": "C Documentation",
            "docs": Path("external/cppreference-doc-20250209/reference/en.cppreference.com/w/c"),
            "resources": Path("external/cppreference-doc-20250209/reference/en.cppreference.com/w/c/resources"),
            "kind": "http",
        },
    ],
}

to_convert = [
    "cpp",
    "c"
]

def get_fortran_docs():
    pass

def get_resource(resource: str, resource_args: dict):
    response = requests.get(resource.format(**resource_args))
    temp_dir = Path("/tmp")
    output_dir = Path("external")
    file_name = resource_args["file_name"]

    if (output_dir / file_name).exists():
        print(f"{file_name} already exists")
        return

    with open(temp_dir / file_name, "wb") as f:
        f.write(response.content)
    with tarfile.open(temp_dir / file_name, "r:bz2") as tar:
        tar.extractall(output_dir, filter='data')


def get_cpp_docs():
    file_name = "cppreference-doc-20250209.tar.xz"
    target = f"https://github.com/PeterFeicht/cppreference-doc/releases/download/v20250209/{file_name}"
    temp_dir = Path("/tmp")
    output_dir = Path("external")

    if (output_dir / file_name[:-7]).exists():
        print(f"{file_name[:-7]} already exists")
        return

    response = requests.get(target)
    with open(temp_dir / file_name, "wb") as f:
        f.write(response.content)
    with tarfile.open(temp_dir / file_name, "r:xz") as tar:
        tar.extractall(output_dir, filter='data')


def convert_to_md():
    tool = Path("external/html2markdown")
    for lang in to_convert:
        print(f"Converting {lang} documentation to markdown")
        success = 0
        files = list(lang_ref[lang].rglob("*.html"))
        total_files = len(files)
        
        # Create output directories first
        Path(f"external/{lang}").mkdir(parents=True, exist_ok=True)
        
        for file in tqdm(files, desc=f"Converting {lang} files"):
            # Calculate relative path from the language root
            rel_path = file.relative_to(lang_ref[lang])
            # Create output path maintaining directory structure
            output_path = Path(f"external/{lang}") / rel_path.with_suffix('.md')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.exists():
                success += 1
                continue

            # Check if filename contains problematic characters
            problematic_chars = ['*', '()', '"']
            has_problematic_chars = any(char in file.name for char in problematic_chars)
            
            try:
                if has_problematic_chars:
                    # Create temporary files with safe names
                    temp_dir = Path("/tmp/html2md")
                    temp_dir.mkdir(exist_ok=True)
                    # Use a safe filename (hash) for temporary files
                    safe_name = f"temp_{abs(hash(str(file)))}"
                    temp_input = temp_dir / f"{safe_name}.html"
                    temp_output = temp_dir / f"{safe_name}.md"
                    
                    # Copy content to temporary file
                    shutil.copy2(file, temp_input)
                    
                    # Convert using temporary files
                    result = subprocess.run(
                        [str(tool), "--input", str(temp_input), "--output", str(temp_output)],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        # Read from temp and write to final location
                        with open(temp_output, 'r') as f:
                            content = f.read()
                            
                        # Replace .html with .md in all references
                        content = content.replace('.html', '.md')
                        
                        # Write to final path
                        with open(output_path, 'w') as f:
                            f.write(content)
                            
                        success += 1
                    else:
                        print(f"Failed to convert {file.name} to {output_path}")
                        print(f"Error: {result.stderr}")
                        
                    # Clean up temporary files
                    temp_input.unlink(missing_ok=True)
                    temp_output.unlink(missing_ok=True)
                else:
                    # For normal files, convert directly
                    result = subprocess.run(
                        [str(tool), "--input", str(file), "--output", str(output_path)],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        # Read the converted markdown
                        with open(output_path, 'r') as f:
                            content = f.read()
                        # Replace .html with .md in all references
                        content = content.replace('.html', '.md')
                        # Write back the updated content
                        with open(output_path, 'w') as f:
                            f.write(content)
                        success += 1
                    else:
                        print(f"Failed to convert {file.name} to {output_path}")
                        print(f"Error: {result.stderr}")
            except Exception as e:
                print(f"Exception when processing {file.name}: {str(e)}")
                
        print(f"converted {success}/{total_files} files")

def pdf_to_md(target: Path):
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(target)
    text, _, images = text_from_rendered(rendered)
    
    with open(target.with_suffix(".md"), "w") as f:
        f.write(text)

def main():
    get_python_docs()
    get_cpp_docs()
    convert_to_md()

    
if __name__ == "__main__":
    main()

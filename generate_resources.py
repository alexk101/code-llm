import os
import shutil
import subprocess
from pathlib import Path

import requests
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from tqdm import tqdm

from utils.preprocessing.validate_resources import validate_resources


def get_resource(lang, record):
    """Download and extract a resource."""
    # Create cache directories
    cache_dir = Path("cache")
    org_resources_dir = cache_dir / "org_resources"
    md_resources_dir = cache_dir / "md_resources"

    # Create subject-specific directories
    subject_org_dir = org_resources_dir / lang
    subject_md_dir = md_resources_dir / lang

    for dir_path in [
        cache_dir,
        org_resources_dir,
        md_resources_dir,
        subject_org_dir,
        subject_md_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create normalized name for directory (lowercase with underscores)
    dir_name = record.name.lower().replace(" ", "_")
    resource_dir = subject_org_dir / dir_name
    resource_dir.mkdir(parents=True, exist_ok=True)

    # Skip if resource should not be fetched
    if not record.get or not record.resource:
        return resource_dir

    # Format resource URL with provided arguments
    resource_url = record.resource.format(**record.resource_args)

    # Format filename if provided
    if record.file_name:
        file_name = record.file_name.format(**record.resource_args)
    else:
        file_name = resource_url.split("/")[-1]

    # Simple check to add extension only if file has no extension
    if record.kind and "." not in file_name:
        # Map of kinds to their expected extensions
        kind_to_extension = {
            "html": ".html",
            "pdf": ".pdf",
            "text": ".txt",
            "markdown": ".md",
        }

        # Get the expected extension for this kind
        expected_ext = kind_to_extension.get(record.kind.lower())

        if expected_ext:
            print(
                f"""Adding {expected_ext} extension
                to {file_name} based on kind: {record.kind}
                """
            )
            file_name = f"{file_name}{expected_ext}"

    local_file = resource_dir / file_name

    if local_file.exists():
        print(f"{file_name} already exists in {resource_dir}")
        # If target is specified, make sure we return the proper subdirectory path
        if record.target:
            # Format target path with resource args if needed
            formatted_target = record.target.format(
                filename=file_name, **record.resource_args
            )
            # Find the target path relative to the resource directory
            target_path = resource_dir / formatted_target
            if target_path.exists():
                return target_path
            else:
                print(
                    f"""Warning: Target path {target_path} does not exist,
                    using full resource directory"""
                )
        return resource_dir

    print(f"Downloading {resource_url} to {local_file}")
    # Download the resource
    response = requests.get(resource_url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Check if we need to modify the file name based
    # on content type (only for files without extension)
    if record.kind and "." not in file_name:
        content_type = response.headers.get("Content-Type", "").lower()

        # Map content types to expected file extensions
        content_type_to_ext = {
            "application/pdf": ".pdf",
            "text/html": ".html",
            "text/plain": ".txt",
            "application/zip": ".zip",
            "application/x-tar": ".tar",
            "application/x-gzip": ".tar.gz",
            "application/x-bzip2": ".tar.bz2",
            "application/x-xz": ".tar.xz",
        }

        for content_type_prefix, ext in content_type_to_ext.items():
            if content_type.startswith(content_type_prefix):
                new_file_name = f"{file_name}{ext}"
                print(
                    f"""Changing file name from {file_name} to {new_file_name}
                    based on Content-Type: {content_type}"""
                )
                file_name = new_file_name
                local_file = resource_dir / file_name
                break

    # Download with progress bar for large files
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(local_file, "wb") as f:
        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=file_name) as t:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)

    # Extract archives if necessary
    extensions = {".tar.bz2": "bz2", ".tar.gz": "gz", ".tar.xz": "xz", ".zip": "zip"}

    for ext, format_name in extensions.items():
        if file_name.endswith(ext):
            print(f"Extracting {file_name} to {resource_dir}")
            if ext == ".zip":
                import zipfile

                with zipfile.ZipFile(local_file, "r") as zip_ref:
                    zip_ref.extractall(resource_dir)
            else:
                import tarfile

                with tarfile.open(local_file, f"r:{format_name}") as tar:
                    tar.extractall(resource_dir, filter="data")
            break

    # If target is specified, check if it exists in the extracted archive
    if record.target:
        # Format target path with resource args if needed
        formatted_target = record.target.format(
            filename=file_name, **record.resource_args
        )
        # Find the target path relative to the resource directory
        target_path = resource_dir / formatted_target
        if target_path.exists():
            print(f"Using target path {target_path}")
            return target_path
        else:
            print(
                f"""Warning: Target path {target_path} does not exist,
                using full resource directory"""
            )

    return resource_dir


def convert_to_md(lang, record, source_dir):
    """Convert HTML files to markdown."""
    # Create normalized output directory under the subject
    cache_dir = Path("cache")
    md_resources_dir = cache_dir / "md_resources" / lang
    dir_name = record.name.lower().replace(" ", "_")
    output_dir = md_resources_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {lang} documentation ({record.name}) to markdown")

    # Find HTML files in the source directory
    html_files = list(source_dir.rglob("*.html"))
    total_files = len(html_files)

    if total_files == 0:
        print(f"No HTML files found in {source_dir}")
        return

    success = 0

    tool = Path("external/binaries/html2markdown")
    if not tool.exists():
        print(f"Error: HTML conversion tool not found at {tool}")
        return

    for file in tqdm(html_files, desc=f"Converting {record.name} files"):
        # Calculate relative path from the source root
        rel_path = file.relative_to(source_dir)
        # Create output path maintaining directory structure
        output_path = output_dir / rel_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            success += 1
            continue

        # Check if filename contains problematic characters
        problematic_chars = ["*", "()", '"']
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
                    [
                        str(tool),
                        "--input",
                        str(temp_input),
                        "--output",
                        str(temp_output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    # Read from temp and write to final location
                    with open(temp_output, "r") as f:
                        content = f.read()

                    # Replace .html with .md in all references
                    content = content.replace(".html", ".md")

                    # Write to final path
                    with open(output_path, "w") as f:
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
                    check=False,
                )

                if result.returncode == 0:
                    # Read the converted markdown
                    with open(output_path, "r") as f:
                        content = f.read()
                    # Replace .html with .md in all references
                    content = content.replace(".html", ".md")
                    # Write back the updated content
                    with open(output_path, "w") as f:
                        f.write(content)
                    success += 1
                else:
                    print(f"Failed to convert {file.name} to {output_path}")
                    print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Exception when processing {file.name}: {str(e)}")

    print(f"Converted {success}/{total_files} files for {record.name}")


def process_text_files(lang, record, source_dir):
    """Process text files to markdown."""
    # Create normalized output directory under the subject
    cache_dir = Path("cache")
    md_resources_dir = cache_dir / "md_resources" / lang
    dir_name = record.name.lower().replace(" ", "_")
    output_dir = md_resources_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = list(source_dir.rglob("*.txt"))
    if not text_files:
        print(f"No text files found in {source_dir}")
        return

    print(f"Processing {len(text_files)} text files for {record.name}")

    for text_file in tqdm(text_files, desc=f"Converting {record.name} text files"):
        rel_path = text_file.relative_to(source_dir)
        output_path = output_dir / rel_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        # Simple copy with extension change for text files
        with open(text_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Processed {len(text_files)} text files")


def execute_command(lang, record):
    """Execute a custom command to generate documentation."""
    if not record.cmd:
        print(f"No command specified for {record.name}")
        return None

    # Create normalized output directory under the subject
    cache_dir = Path("cache")
    org_resources_dir = cache_dir / "org_resources" / lang
    md_resources_dir = cache_dir / "md_resources" / lang
    dir_name = record.name.lower().replace(" ", "_")

    # Source directory is in org_resources
    source_dir = org_resources_dir / dir_name
    # Output directory is in md_resources
    output_dir = md_resources_dir / dir_name

    # Ensure directories exist
    org_resources_dir.mkdir(parents=True, exist_ok=True)
    md_resources_dir.mkdir(parents=True, exist_ok=True)

    # Create any parent directories for the source
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)

    print(f"Executing command for {record.name}: {record.cmd}")

    # Create a temporary environment dictionary with useful paths
    env_vars = {
        "SOURCE_DIR": str(source_dir),
        "OUTPUT_DIR": str(output_dir),
        "CACHE_DIR": str(cache_dir),
        "ORG_RESOURCES_DIR": str(org_resources_dir),
        "MD_RESOURCES_DIR": str(md_resources_dir),
        "RESOURCE_NAME": record.name,
        "RESOURCE_DIR_NAME": dir_name,
        "LANG": lang,
    }

    # Format the command with environment variables if needed
    formatted_cmd = record.cmd
    try:
        # Try to format the command with env vars
        if "{" in record.cmd and "}" in record.cmd:
            formatted_cmd = record.cmd.format(**env_vars)
    except KeyError as e:
        print(f"Warning: Failed to format command with environment variables: {e}")

    try:
        # Execute the command
        result = subprocess.run(
            formatted_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **env_vars},  # Merge with existing environment
        )

        print(f"Command executed successfully for {record.name}")
        if result.stdout:
            print(f"Command output: {result.stdout[:200]}...")

        # Handle special case for wget where it creates
        # a localhost:PORT directory structure
        if "wget" in formatted_cmd:
            # Look for localhost:PORT directory that wget typically creates
            localhost_dirs = list(source_dir.glob("localhost*"))
            if localhost_dirs:
                print(f"Found wget localhost directory: {localhost_dirs[0]}")
                # Use the localhost directory as the actual source for conversion
                source_dir = localhost_dirs[0]

        # If the command was successful and the kind is specified, perform conversion
        if record.kind and source_dir.exists():
            kind = record.kind.lower()

            # Create output directory
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            # Convert based on kind
            if kind == "html":
                convert_to_md(lang, record, source_dir)
            elif kind == "pdf":
                pdf_files = source_dir.rglob("*.pdf")
                for pdf_file in pdf_files:
                    pdf_to_md(lang, record, pdf_file, output_dir)
            elif kind == "text":
                process_text_files(lang, record, source_dir)
            elif kind == "markdown":
                # If already markdown, just copy files
                md_files = list(source_dir.rglob("*.md"))
                for md_file in md_files:
                    rel_path = md_file.relative_to(source_dir)
                    target_path = output_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(md_file, target_path)
                print(f"Copied {len(md_files)} markdown files for {record.name}")
            else:
                print(f"Warning: Unsupported resource kind: {kind}")

        # Return the source directory
        return source_dir
    except subprocess.CalledProcessError as e:
        print(f"Error executing command for {record.name}: {e}")
        print(f"Command stderr: {e.stderr}")
        return None


def pdf_to_md(lang, record, pdf_path, output_dir):
    """Convert PDF files to markdown."""
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        return

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)

    # Ensure output directory exists
    output_path = output_dir / pdf_path.with_suffix(".md").name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF {pdf_path} to {output_path}")
    with open(output_path, "w") as f:
        f.write(text)


def cleanup_empty_dirs(directory):
    """Remove empty directories recursively from bottom up."""
    if not directory.exists() or not directory.is_dir():
        return

    # First recursively clean up subdirectories
    for path in list(directory.iterdir()):
        if path.is_dir():
            cleanup_empty_dirs(path)

    # Check if the directory is now empty after cleaning its subdirectories
    if not any(directory.iterdir()):
        print(f"Removing empty directory: {directory}")
        directory.rmdir()


def main():
    """Main function to process resources."""
    # Validate resources file
    yaml_path = "resources.yml"
    print(f"Validating resources file: {yaml_path}")
    validated_resources = validate_resources(yaml_path)

    # Set up cache directories
    cache_dir = Path("cache")
    org_resources_dir = cache_dir / "org_resources"
    md_resources_dir = cache_dir / "md_resources"

    for dir_path in [cache_dir, org_resources_dir, md_resources_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process each language/subject with validated resources
    for lang, resources in validated_resources.items():
        # Create subject-specific directories
        subject_org_dir = org_resources_dir / lang
        subject_md_dir = md_resources_dir / lang
        subject_org_dir.mkdir(parents=True, exist_ok=True)
        subject_md_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing resources for {lang.upper()}")
        for record in resources:
            # Create normalized output directory name
            dir_name = record.name.lower().replace(" ", "_")
            output_dir = subject_md_dir / dir_name

            # Check if the resource already exists in the output directory
            if output_dir.exists():
                print(
                    f"Resource '{record.name}' already exists at {output_dir}, skipping"
                )
                continue

            source_dir = None

            # Handle cmd parameter (custom command to generate documentation)
            if record.cmd:
                execute_command(lang, record)
                continue
            # Handle source parameter (local resources)
            elif record.source:
                source_path = Path(record.source)
                if source_path.exists():
                    source_dir = source_path

                    # If kind is not specified and source is a directory,
                    # symlink it instead of converting
                    if record.kind is None and source_path.is_dir():
                        # Create parent directories if they don't exist
                        subject_md_dir.mkdir(parents=True, exist_ok=True)

                        # Create symlink
                        if not output_dir.exists():
                            print(
                                f"Creating symlink from {source_path} to {output_dir}"
                            )
                            output_dir.symlink_to(source_path, target_is_directory=True)
                        continue
                else:
                    print(f"Warning: Source path does not exist: {source_path}")
                    continue
            # Handle resource parameter (downloadable resources)
            elif record.resource:
                source_dir = get_resource(lang, record)
            else:
                output = (
                    f"Warning: Resource for {lang} ({record.name}) has no valid "
                    f"source, resource, or cmd"
                )
                print(output)
                continue

            # Skip further processing if we've symlinked the directory
            if record.source and record.kind is None and Path(record.source).is_dir():
                continue

            # Create output directory under the subject directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert resources based on kind
            kind = record.kind.lower() if record.kind else "markdown"

            if kind == "html":
                convert_to_md(lang, record, source_dir)
            elif kind == "pdf":
                # Find all PDF files in the source directory
                pdf_files = source_dir.rglob("*.pdf")
                for pdf_file in pdf_files:
                    pdf_to_md(lang, record, pdf_file, output_dir)
            elif kind == "text":
                process_text_files(lang, record, source_dir)
            elif kind == "http":
                # Special handling for HTTP/HTML content
                convert_to_md(lang, record, source_dir)
            elif kind == "markdown":
                # If already markdown, just copy files
                if source_dir and source_dir.exists():
                    md_files = list(source_dir.rglob("*.md"))
                    for md_file in md_files:
                        rel_path = md_file.relative_to(source_dir)
                        target_path = output_dir / rel_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(md_file, target_path)
                    print(f"Copied {len(md_files)} markdown files for {record.name}")
            else:
                print(f"Warning: Unsupported resource kind: {kind}")

    # Clean up empty directories in the cache after processing
    print("\nCleaning up empty directories in cache...")
    cleanup_empty_dirs(cache_dir)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()

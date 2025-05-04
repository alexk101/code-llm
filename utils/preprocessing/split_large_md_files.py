#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


def is_heading(line, max_level=3):
    """Check if a line is a markdown heading and return its level if so.

    Args:
        line: String containing a line of text
        max_level: Maximum heading level to consider (1-6)

    Returns:
        int: Heading level (1-max_level) or 0 if not a heading
    """
    # Match ATX headings: # Heading, ## Heading, etc.
    atx_match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if atx_match:
        level = len(atx_match.group(1))
        if level <= max_level:
            return level, atx_match.group(2).strip()

    # Match Setext headings: Heading\n======== or Heading\n--------
    if line.strip() and len(line.strip()) > 0:
        return 0, None

    # Match other common heading formats
    # Numbered sections like "1.2.3 Section Name"
    numbered_match = re.match(r"^(\d+\.(?:\d+\.)*)\s+(.+)$", line)
    if numbered_match:
        # Count the dots to determine level
        dots = numbered_match.group(1).count(".")
        level = dots + 1
        if level <= max_level:
            return level, numbered_match.group(2).strip()

    # Chapter/Section style headings
    section_match = re.match(
        r"^(Chapter|Section|Part|Module)\s+\d+[:\.]?\s*(.+)$", line.strip()
    )
    if section_match:
        # Treat these as level 1 or 2 based on the section type
        section_type = section_match.group(1).lower()
        level = 1 if section_type in ["chapter", "part"] else 2
        if level <= max_level:
            return level, section_match.group(2).strip()

    return 0, None


def check_setext_heading(line, next_line, max_level=3):
    """Check if a pair of lines forms a Setext heading."""
    if not next_line:
        return 0, None

    # Check for underlined headings: Heading\n=======
    if re.match(r"^=+$", next_line.strip()):
        if 1 <= max_level:
            return 1, line.strip()
    elif re.match(r"^-+$", next_line.strip()):
        if 2 <= max_level:
            return 2, line.strip()

    return 0, None


def sanitize_filename(title, max_length=40):
    """Convert a title to a safe filename."""
    # Remove invalid characters
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().lower()
    # Replace whitespace with hyphens
    safe_title = re.sub(r"[-\s]+", "-", safe_title)

    # Ensure the filename is not too long
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length]

    # Remove trailing hyphens
    safe_title = safe_title.rstrip("-")

    # Ensure we have something
    if not safe_title:
        safe_title = "section"

    return safe_title


class HierarchicalSection:
    """Represents a section in a hierarchical document structure."""

    def __init__(self, level, title, content="", parent=None):
        self.level = level
        self.title = title
        self.content = content
        self.parent = parent
        self.children = []
        self.index = 0  # For ordering siblings

    def add_child(self, child):
        """Add a child section."""
        child.parent = self
        child.index = len(self.children)
        self.children.append(child)

    def get_path_components(self):
        """Get path components for this section in the hierarchy."""
        if self.parent is None:
            return []

        parent_components = self.parent.get_path_components()
        # Add this component
        if self.level > 0:  # Don't include the root
            # Format: 001-section-name
            component = f"{self.index + 1:03d}-{sanitize_filename(self.title)}"
            return parent_components + [component]
        else:
            return parent_components

    def __repr__(self):
        return f"""<Section level={self.level} title='{self.title}'
        children={len(self.children)}>"""


def build_section_hierarchy(content, max_heading_level=3, min_size=500, max_size=50000):
    """Build a hierarchical structure of sections based on heading levels.

    Args:
        content: String containing markdown content
        max_heading_level: Maximum heading level to consider (1-6)
        min_size: Minimum section size in characters
        max_size: Maximum section size in characters

    Returns:
        root: Root section containing the hierarchical document structure
    """
    lines = content.split("\n")

    # Create a root section
    root = HierarchicalSection(0, "Root")

    # Create an initial section for content before any headings
    intro = HierarchicalSection(1, "Introduction", "")
    root.add_child(intro)

    # Keep track of the current section at each level
    current_sections = {0: root, 1: intro}
    current_section = intro

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        # Check for setext headings (underlined)
        setext_level, setext_title = check_setext_heading(
            line, next_line, max_heading_level
        )
        if setext_level > 0:
            # We found a setext heading, create a new section
            section = HierarchicalSection(
                setext_level, setext_title, f"{'#' * setext_level} {setext_title}\n\n"
            )

            # Add as child to the appropriate parent
            parent_level = setext_level - 1
            while parent_level > 0 and parent_level not in current_sections:
                parent_level -= 1
            current_sections[parent_level].add_child(section)

            # Update current section references
            current_sections[setext_level] = section
            current_section = section

            # Clean higher levels
            for level in list(current_sections.keys()):
                if level > setext_level:
                    del current_sections[level]

            # Skip the underline
            i += 2
            continue

        # Check for ATX and other headings
        level, title = is_heading(line, max_heading_level)

        if level > 0:
            # We found a heading, create a new section
            section = HierarchicalSection(level, title, f"{line}\n\n")

            # Add as child to the appropriate parent
            parent_level = level - 1
            while parent_level > 0 and parent_level not in current_sections:
                parent_level -= 1
            current_sections[parent_level].add_child(section)

            # Update current section references
            current_sections[level] = section
            current_section = section

            # Clean higher levels
            for level_key in list(current_sections.keys()):
                if level_key > level:
                    del current_sections[level_key]
        else:
            # Add line to current section
            current_section.content += f"{line}\n"

            # If current section is getting too large, force a split
            if len(current_section.content) >= max_size:
                # For root level, create a new part
                if current_section.level <= 1:
                    part_num = len(
                        [
                            c
                            for c in current_section.parent.children
                            if c.title.startswith("Part ")
                        ]
                    )
                    new_part = HierarchicalSection(
                        current_section.level,
                        f"Part {part_num + 1}",
                        f"{'#' * current_section.level} Part {part_num + 1}\n\n",
                    )
                    current_section.parent.add_child(new_part)
                    current_section = new_part
                    current_sections[current_section.level] = new_part

        i += 1

    # Prune empty sections or those with minimal content
    def prune_section(section):
        # Keep all non-empty higher level sections
        if section.level <= 1 and section.content.strip():
            return True

        # Keep sections with sufficient content
        content_size = len(section.content)
        if content_size >= min_size:
            return True

        # If section has children, keep it
        if section.children:
            return True

        # Otherwise prune
        return False

    def filter_tree(section):
        # Filter children
        section.children = [child for child in section.children if prune_section(child)]

        # Update indices
        for i, child in enumerate(section.children):
            child.index = i
            filter_tree(child)

    # Filter the tree starting from root
    filter_tree(root)

    # If we have no proper sections, create artificial parts based on size
    if (
        all(len(child.children) == 0 for child in root.children)
        and len(content) > max_size
    ):
        # If we just have a single large intro, split it into parts
        intro = root.children[0] if root.children else None
        if intro and len(intro.content) > max_size:
            # Clear existing content
            root.children = []

            # Create chunked parts
            chunk_size = max_size // 2  # Target a reasonable chunk size
            chunks = [
                content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
            ]

            for i, chunk in enumerate(chunks):
                section_title = f"Part {i + 1}"
                section = HierarchicalSection(
                    1,  # Level 1
                    section_title,
                    f"# {section_title}\n\n{chunk}",
                )
                root.add_child(section)

    return root


def write_section_files(
    section, base_dir, level_dirs=True, index_entries=None, path_prefix=""
):
    """Write section files to the appropriate directories.

    Args:
        section: The section to write
        base_dir: The base directory for all files
        level_dirs: Whether to create level-based directories
            (True) or flat structure (False)
        index_entries: List to collect index entries
        path_prefix: Prefix for paths in index

    Returns:
        filename: The filename of the written section
    """
    if index_entries is None:
        index_entries = []

    # Skip the root node
    if section.level == 0:
        # Process all children
        for child in section.children:
            write_section_files(child, base_dir, level_dirs, index_entries, path_prefix)
        return None

    # Determine the path structure
    if level_dirs and section.level > 1:
        # Get path components from parent hierarchy
        path_components = section.get_path_components()

        # The last component is this section's name - we'll use it for the filename
        if path_components:
            rel_dir = (
                os.path.join(*path_components[:-1]) if len(path_components) > 1 else ""
            )
            filename = f"{path_components[-1]}.md"
        else:
            rel_dir = ""
            filename = f"{section.index + 1:03d}-{sanitize_filename(section.title)}.md"

        # Create the full path
        target_dir = base_dir / rel_dir if rel_dir else base_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / filename
    else:
        # Flat structure - just use index number
        filename = f"{section.index + 1:03d}-{sanitize_filename(section.title)}.md"
        file_path = base_dir / filename

    # Write section content to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(section.content)

    # Create relative path for index
    if level_dirs:
        rel_path = os.path.join(path_prefix, file_path.relative_to(base_dir))
    else:
        rel_path = os.path.join(path_prefix, filename)

    # Add to index
    level_prefix = "  " * (section.level - 1)
    index_entries.append(
        (section.level, f"{level_prefix}- [{section.title}]({rel_path})")
    )

    # Process children if any
    for child in section.children:
        write_section_files(child, base_dir, level_dirs, index_entries, path_prefix)

    return filename


def process_large_md_file(
    md_file,
    output_dir,
    max_heading_level=3,
    min_section_size=500,
    max_section_size=50000,
    level_dirs=True,
):
    """Process a large markdown file into a hierarchical directory structure.

    Args:
        md_file: Path to the markdown file
        output_dir: Directory to output the split files
        max_heading_level: Maximum heading level to split on
        min_section_size: Minimum section size in characters
        max_section_size: Maximum section size in characters
        level_dirs: Whether to create directories based on heading levels

    Returns:
        bool: Whether processing was successful
    """
    try:
        # Read the file
        with open(md_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Skip if file is small enough
        if len(content) <= max_section_size:
            return False

        # Build section hierarchy
        root = build_section_hierarchy(
            content,
            max_heading_level=max_heading_level,
            min_size=min_section_size,
            max_size=max_section_size,
        )

        # Count total sections (not including root)
        total_sections = 0

        def count_sections(section):
            if section.level > 0:  # Don't count root
                return 1 + sum(count_sections(child) for child in section.children)
            else:
                return sum(count_sections(child) for child in section.children)

        total_sections = count_sections(root)

        if total_sections <= 1:
            return False

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write section files and collect index entries
        index_entries = []
        write_section_files(root, output_dir, level_dirs, index_entries)

        # Create index file
        index_path = output_dir / "index.md"
        index_content = f"# {md_file.stem}\n\n## Table of Contents\n\n"

        # Sort entries by level first, then by order of appearance
        # This ensures the TOC is properly nested
        index_entries.sort(key=lambda x: x[0])
        for _, entry in index_entries:
            index_content += f"{entry}\n"

        # Write index file
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        return True
    except Exception as e:
        print(f"Error processing {md_file}: {e}")
        import traceback

        traceback.print_exc()
        return False


def process_md_resources(
    resources_dir,
    max_heading_level=3,
    min_section_size=500,
    max_section_size=50000,
    level_dirs=True,
    dry_run=False,
):
    """Process all markdown resources in the resources directory.

    Args:
        resources_dir: Path to resources directory
        max_heading_level: Maximum heading level to use for splitting (1-6)
        min_section_size: Minimum section size in characters
        max_section_size: Maximum section size in characters
        level_dirs: Whether to create directories based on heading hierarchy
        dry_run: Whether to perform a dry run (no changes)

    Returns:
        int: Number of processed files
    """
    resources_dir = Path(resources_dir)

    if not resources_dir.exists():
        print(f"Resources directory not found: {resources_dir}")
        return 0

    # Create old_md_resources directory (parallel to md_resources)
    old_resources_base = resources_dir.parent / "old_md_resources"
    print(f"Creating directory for original files: {old_resources_base}")
    old_resources_base.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories (language/subject directories)
    subject_dirs = [d for d in resources_dir.iterdir() if d.is_dir()]

    total_processed = 0
    total_candidates = 0

    # Process each subject directory
    for subject_dir in subject_dirs:
        print(f"\nProcessing subject: {subject_dir.name}")

        # Create corresponding subject directory in old_resources
        old_subject_dir = old_resources_base / subject_dir.name
        old_subject_dir.mkdir(parents=True, exist_ok=True)

        # Get all resource directories for this subject
        resource_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

        # Dictionary to track resources with only one markdown file
        single_file_resources = defaultdict(list)

        # Find resources with only one markdown file
        for resource_dir in resource_dirs:
            md_files = list(resource_dir.glob("*.md"))

            # Skip if resource already has subdirectories with markdown files
            has_subdir_with_md = any(
                d.is_dir() and list(d.glob("*.md")) for d in resource_dir.iterdir()
            )

            if len(md_files) == 1 and not has_subdir_with_md:
                md_file = md_files[0]
                # Check file size
                file_size = md_file.stat().st_size
                if file_size > max_section_size:
                    single_file_resources[resource_dir.name].append(
                        (md_file, file_size)
                    )
                    total_candidates += 1

        if not single_file_resources:
            print(f"No large single-file resources found in {subject_dir.name}")
            continue

        print(
            f"""Found {sum(len(files) for files in single_file_resources.values())}
            large single-file resources"""
        )

        # Process each resource
        for resource_name, files in single_file_resources.items():
            for md_file, file_size in files:
                print(f"Processing {md_file.name} ({file_size / 1024:.1f} KB)")

                # Create output directory with the same name as the file
                output_dir = md_file.parent / md_file.stem

                # Create corresponding resource directory in old_resources
                old_resource_dir = old_subject_dir / resource_name
                old_resource_dir.mkdir(parents=True, exist_ok=True)

                # Path to save the original file
                old_file_path = old_resource_dir / md_file.name

                kind = "hierarchical" if level_dirs else "flat"
                if dry_run:
                    print(
                        f"""Would split {md_file} into {output_dir} using
                        {kind} structure (dry run)"""
                    )
                    print(f"  Would save original file to {old_file_path} (dry run)")
                    total_processed += 1
                    continue

                # First, copy the original file to old_resources
                try:
                    print(f"  Saving original file to {old_file_path}")
                    shutil.copy2(md_file, old_file_path)
                except Exception as e:
                    print(f"  Error copying original file: {e}")
                    continue

                # Process the file
                success = process_large_md_file(
                    md_file,
                    output_dir,
                    max_heading_level=max_heading_level,
                    min_section_size=min_section_size,
                    max_section_size=max_section_size,
                    level_dirs=level_dirs,
                )

                if success:
                    print(
                        f"""Split {md_file} into a {kind}
                        structure in {output_dir}"""
                    )

                    # Create a redirect in the original file
                    with open(md_file, "w", encoding="utf-8") as f:
                        f.write(
                            f"""# {md_file.stem}
                            \n\nThis content has been split into multiple files.
                            \n\n[Go to index]({md_file.stem}/index.md)\n"""
                        )

                    total_processed += 1
                else:
                    print(f"  Could not split {md_file} (no suitable sections found)")

                    # If we couldn't split the file, restore the original
                    # from our backup
                    try:
                        shutil.copy2(old_file_path, md_file)
                        print("  Restored original file from backup")
                    except Exception as e:
                        print(f"  Error restoring original file: {e}")

    print(f"\nProcessed {total_processed}/{total_candidates} large markdown files")
    print(f"Original files preserved in {old_resources_base}")
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="Split large markdown files into smaller sections based on headings"
    )
    parser.add_argument(
        "--resources-dir",
        default="cache/md_resources",
        help="Path to the markdown resources directory",
    )
    parser.add_argument(
        "--max-heading-level",
        type=int,
        default=3,
        help="Maximum heading level to use for splitting (1-6)",
    )
    parser.add_argument(
        "--min-section-size",
        type=int,
        default=500,
        help="Minimum section size in characters",
    )
    parser.add_argument(
        "--max-section-size",
        type=int,
        default=50000,
        help="Maximum section size in characters",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat directory structure instead of hierarchical",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run (no changes made)"
    )

    args = parser.parse_args()

    process_md_resources(
        args.resources_dir,
        max_heading_level=args.max_heading_level,
        min_section_size=args.min_section_size,
        max_section_size=args.max_section_size,
        level_dirs=not args.flat,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

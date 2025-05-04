# Code-LLM Tools Documentation

## Core Tools

### generate_resources.py

This script downloads and processes programming language documentation from various sources as specified in `resources.yml`. It supports multiple document formats and includes functionality to convert them to markdown for consistent processing.

**Key Functions:**
- `get_resource(lang, record)`: Downloads and extracts resources from URLs or archives
- `convert_to_md(lang, record, source_dir)`: Converts HTML documentation to markdown
- `process_text_files(lang, record, source_dir)`: Processes plain text documentation
- `pdf_to_md(lang, record, pdf_path, output_dir)`: Converts PDF documentation to markdown
- `execute_command(lang, record)`: Executes custom commands to generate documentation
- `cleanup_empty_dirs(directory)`: Removes empty directories from the cache

**Usage:**
```
python generate_resources.py
```

### generate_tools.py

A utility for generating tools and processing language information. It retrieves programming language information from external sources and processes them for use in the project.

**Key Functions:**
- `get_language_info()`: Retrieves programming language popularity information from TIOBE index

**Usage:**
```
python generate_tools.py
```

### init.py

Initialization script that sets up the environment for the project. It downloads necessary binaries and tools.

**Key Functions:**
- `download_fastfetch()`: Downloads the fastfetch CLI tool based on platform
- `call_fastfetch()`: Calls fastfetch to display system information
- `download_html_to_md()`: Downloads the html2markdown converter tool
- `main()`: Main initialization function that sets up all tools

**Usage:**
```
python init.py
```

## Visualization Tools

### viz/graph_viz.py

Generates graph visualizations of markdown document relationships, showing how different programming language documentation is interconnected.

**Key Features:**
- Creates a network graph of markdown files and their links
- Generates node embeddings using various methods (Node2Vec, GGVec, ProNE, etc.)
- Visualizes embeddings using UMAP dimensionality reduction
- Calculates graph statistics and metrics

**Key Classes:**
- `MarkdownGraphVisualizer`: Main class for building and visualizing markdown document graphs

**Usage:**
```
python viz/graph_viz.py [--resources-dir DIR] [--output-dir DIR] [--dimensions N]
                        [--embedding-methods METHOD1,METHOD2] [--min-degree N]
                        [--workers N] [--no-save]
```

### viz/embed_viz.py

Visualizes code embeddings from the Rosetta Code dataset using CodeBERT and dimensionality reduction techniques.

**Key Features:**
- Generates embeddings for code samples using CodeBERT
- Reduces dimensions using UMAP or PCA
- Creates 2D and 3D visualizations of code embeddings by language or task
- Supports both supervised and unsupervised dimensionality reduction

**Key Classes:**
- `CodeEmbeddingVisualizer`: Handles embedding generation and visualization

**Usage:**
```
python viz/embed_viz.py
```

## Utility Tools

### utils/data.py

Handles dataset loading and preprocessing, particularly for the Rosetta Code dataset.

**Key Components:**
- Dataset management with Polars
- Language normalization and relabeling
- Blacklisted language filtering

**Usage:**
```python
from utils.data import get_dataset

# Load the Rosetta Code dataset
df = get_dataset()
```

### utils/validate_resources.py

Validates the `resources.yml` file to ensure proper configuration of resources.

**Key Components:**
- `ResourceConfig` dataclass: Defines the structure of resource configurations
- `validate_resources(yaml_path)`: Validates a resources YAML file 
- `check_output_structure(resources_dict)`: Validates the output directory structure

**Usage:**
```python
from utils.validate_resources import validate_resources

# Validate resources configuration
resources = validate_resources("resources.yml")
```

### utils/split_large_md_files.py

Splits large markdown files into smaller, more manageable chunks based on headings.

**Key Functions:**
- `extract_sections(content, max_heading_level, min_size, max_size)`: Splits content into sections
- `process_large_md_file(md_file, output_dir, max_heading_level, min_section_size, max_section_size)`: Processes a single markdown file
- `process_md_resources(resources_dir, max_heading_level, min_section_size, max_section_size, dry_run)`: Processes all markdown resources

**Usage:**
```
python utils/split_large_md_files.py [--resources-dir DIR] [--max-heading-level N] [--min-section-size N] [--max-section-size N] [--dry-run]
```

## Resource Configuration

### resources.yml

This YAML file defines the documentation resources to be downloaded and processed by `generate_resources.py`. Each resource is organized by programming language.

**Configuration Structure:**
- Top-level keys represent programming languages or subjects
- Each subject contains a list of resources
- Each resource has the following possible attributes:
  - `name`: Identifier for the resource
  - `resource`: URL to download the resource (optional)
  - `resource_args`: Variables to format into the resource URL (optional)
  - `kind`: Type of resource (html, pdf, text, markdown)
  - `source`: Local path to a resource (alternative to `resource`)
  - `get`: Boolean flag to indicate if the resource should be downloaded
  - `target`: Specific subdirectory within extracted archives to use
  - `cmd`: Custom command to generate documentation

**Example:**
```yaml
python:
  - name: "Python 3.12 Documentation"
    resource: "https://docs.python.org/{python_version}/archives/python-{python_version}-docs-text.tar.bz2"
    resource_args:
      python_version: "3.12"
    kind: "text"
    get: true
```

## Cache Structure

The project uses a cache directory to store downloaded and processed resources:

- `cache/org_resources/`: Original downloaded resources organized by language
- `cache/md_resources/`: Markdown-converted resources organized by language
- `cache/embeddings/`: Cached embeddings and dimensionality reduction results

## Development Setup

Requirements:
- Python 3.12+
- Dependencies specified in `pyproject.toml`

External dependencies (automatically downloaded by init.py):
- fastfetch: System information display
- html2markdown: HTML to Markdown converter

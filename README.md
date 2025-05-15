# Code Transliteration and Auto Optimizer

## Datasets

- [Rosetta Code](https://huggingface.co/datasets/christopher/rosetta-code)

For our experiments, we utilize the Rosetta Code dataset, which provides implementations of the same algorithms across multiple programming languages. We've converted this resource into a structured database to facilitate algorithmic translation and optimization experiments. Our focus is primarily on the TIOBE top 20 programming languages, which represent the most widely used languages in the industry.

## Getting Started

This section provides step-by-step instructions to set up and run the project from start to finish.

### Prerequisites

- Python 3.12 or higher
- Git
- Linux
- A local LLM server (default: running on http://127.0.0.1:1234/v1/chat/completions)
- Compilers/interpreters for languages you want to test (as defined in `language_tools.yaml`)
- [uv](https://github.com/astral-sh/uv)

### Local LLM

You can initialize any llm local, as long as it has an http endpoint for chat completions. Ones which will work out of the box

1. [Llamacpp](https://github.com/ggml-org/llama.cpp)
2. [LMstudio](https://lmstudio.ai/) (user friendly graphical llamacpp)

The model with which this was tested was Gemma-3 27b.

### Installation

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/your-username/code-llm.git
   cd code-llm
   git submodule update --init --recursive
   ```

2. Create and activate a virtual environment:
   ```bash
   uv python install
   uv sync
   source .venv/bin/activate
   ```

### Generating Resources

1. Generate documentation resources for GraphRAG:
   ```bash
   python generate_resources.py
   ```
   This will fetch and process documentation for supported programming languages and store them in the `cache/` directory.

### Initializing GraphRAG Database

1. Initialize and build the GraphRAG knowledge graph:
   ```bash
   python -c "from utils.rag import GraphRAG; rag = GraphRAG(); rag.index()"
   ```
   
   This command:
   - Processes the documentation in the `cache/` directory
   - Builds a knowledge graph from the documentation
   - Generates embeddings for the nodes
   - Stores the graph and embeddings in the Milvus database

2. You can verify the initialization by running a test query:
   ```bash
   python -c "from utils.rag import GraphRAG; rag = GraphRAG(); print(rag.query('How to create an array in Python'))"
   ```

### Visualization Tools

The project includes visualization tools in the `viz` module to help you understand the data and results better.

#### Code Embedding Visualization

The `embed_viz.py` script visualizes code embeddings across different programming languages and tasks:

```bash
# Generate embeddings and visualize in 2D and 3D
python -m viz.embed_viz

# Customize the visualization
python -m viz.embed_viz --method umap --n-languages 10 --supervised
```

This will generate visualizations showing how different programming languages and tasks cluster in the embedding space, using techniques like UMAP and PCA. Visualizations are stored in the `plots/embeddings` directory.

#### Knowledge Graph Visualization

The `graph_viz.py` script visualizes the knowledge graph used by GraphRAG:

```bash
# Generate and visualize the knowledge graph
python -m viz.graph_viz

# Customize with different reduction methods
python -m viz.graph_viz --reduction-method tsne
```

This creates visualizations of the documentation knowledge graph, showing relationships between different programming languages and concepts. Visualizations are stored in the `plots/graph_emb` directory with different dimensionality reduction techniques (UMAP, t-SNE, PCA).

### Running Experiments

1. Run a basic experiment:
   ```bash
   python run_experiment.py --name my_first_experiment
   ```

2. For a more customized experiment:
   ```bash
   python run_experiment.py --name custom_exp \
     --source-language Python \
     --target-languages C Java JavaScript \
     --num-problems 5 \
     --use-graphrag
   ```

3. View results in the `experiment_results/` directory.

### Visualizing Results

To generate visualizations of experiment results:
```bash
python cli.py --visualize experiment_results/your_experiment_name
```

### End-to-End Workflow

A complete workflow consists of these steps:

1. **Setup**: Install dependencies and generate resources
   ```bash
   # After installation steps above
   python generate_resources.py
   ```

2. **Run experiment**: Translate code from one language to others
   ```bash
   python run_experiment.py --name full_workflow \
     --source-language Python \
     --target-languages C Java JavaScript Rust \
     --num-problems 10 \
     --use-graphrag
   ```

3. **Evaluate results**: Check the experiment output
   ```bash
   # Results are in experiment_results/full_workflow/
   # You can view metrics.json for performance summary
   ```

4. **Visualize**: Generate plots from results
   ```bash
   python cli.py --visualize experiment_results/full_workflow
   ```

5. **Test specific translations**: Use the CLI to test individual implementations
   ```bash
   # Test a translated implementation
   python cli.py experiment_results/full_workflow/translations_graphrag/fibonacci/c/implementation.c \
     --language c \
     --test experiment_results/full_workflow/test_cases/fibonacci/test_cases.json
   ```

## Proposed External dependencies

- [fast-fetch](https://github.com/fastfetch-cli/fastfetch)

## General

**Overall Idea**

### Phase 1: Initialization

1. Select a research paper (code or no code).
2. Generate many strict test cases for correctness + performance evaluation framework.
3. Generate high level pseudocode for generic reference.

- What hardware did this code originally run on?
    - Process architecture
    - Accelerator
    - Operating System
    - Environment Dependencies
- What hardware do we want to run this on?
- Are we in any way hardware bound? (Memory, required hardware, etc)


### Phase 2: Code Translation

5. Choose a programming language for the final implementation.
6. Translate codebase to language using `llm-function` to define compilation/running functions.
7. Use defined functions to run initial implementation in new language, using test cases from step 3 to ensure correctness.

### Phase 3: Optimization through Critique

8. Reference high level pseudo code and consider performance improvements.
9. Using a fixed recursion depth (number of critique passes), have one agent propose and optimization, and the other critique it.
10. Ensure correctness after each pass

### Cross language testing

Every program must be able to accept command line arguments which will serve as the entry point for testing.

## Resource Specification

Resources as specified in yaml. The top level is the subject. By default this is the programming language. However, this is theoretically extensible to any subject. For example, if you have a group of resources related to material science, quantum physics, etc. There can be multiple resources per subject. All fetched resources are stored in the `cache/org_resources` directory, each in a directory according to the `name` parameter. The final, markdown resources are contained in `cache/md_resources`.

The yaml has the following structure

- `name`: The name of resource. This will be used to identify the resource.
- `resource` (Optional): The url to the resource. If the resource is managed through a git submodule, this should not be set and the `source` parameter should be used.
- `resource_args` (Optional): Variadic parameter that can include any number of parameters which are parsed as `**kwargs` into the `resource` parameter. Useful for simplifying the fetching of resources with common structured resources, such as version numbers or commit ids.
- `kind`: The kind of resources. Manually specified, since this indicates the actual filetype in the resource, which can be an archive, which when extracted, contains resources of this type. Determines what type of conversion is used. If this is not specified, it is assumed the the resources are already markdown.
- `source` (Optional): The local path to a resource. Mutually exclusive with the `resource` parameter. If this is a directory and `kind` is specified, it is converted into the cache. If this is a directory and `kind` is not specified, it is symlinked.
- `get` (Optional): Used with the `resource` parameter to indicate if a resources should be fetched. Ignored when used with the `source` parameter.
- `target` (Optional): Specifies a subpath within an extracted archive to use as the actual source. This is useful when a single archive contains documentation for multiple languages or subjects. Can include format specifiers like `{filename}` which will be replaced with the downloaded filename.
- `cmd` (Optional): Command to generate the documentation

## Tool Specification

To enable compilation, execution, and testing of code across multiple programming languages, we use a language tools configuration system. This allows for flexible, language-specific handling of code through a standard interface.

### Language Tools Configuration

Languages are configured in the `language_tools.yaml` file with the following structure:

```yaml
language_name:
  extension: file_extension
  compile: compile_command  # Optional for interpreted languages
  run: run_command
```

The configuration supports placeholder variables:
- `{source}`: Path to the source code file
- `{output}`: Path to the compiled output (without extension)

For example, the C language configuration:

```yaml
c:
  extension: c
  compile: gcc -o {output} {source}
  run: {output}
```

For interpreted languages like Python:

```yaml
python:
  extension: py
  run: python {source}
```

### Command Line Interface

The CLI supports compiling, running, and testing code in different languages:

```bash
# Run code in a specific language
python cli.py examples/fibonacci.py --language python --run

# Compile-only for compiled languages
python cli.py examples/fibonacci.c --language c --compile

# Run with arguments
python cli.py examples/fibonacci.py --language python --run --args 10

# Run tests using a JSON test file
python cli.py examples/fibonacci.py --language python --test examples/test_cases.json
```

Test cases are defined in JSON format:

```json
[
  {
    "input": ["arg1", "arg2"],
    "expected": "expected output"
  }
]
```

### Programmatic API

You can also use the tool programmatically:

```python
from utils.tools import LanguageTools

# Initialize with language configuration
tools = LanguageTools("language_tools.yaml")

# Compile code
success, message = tools.compile(code, "c")

# Run code with arguments
success, output = tools.run(code, "python", args=["10"])

# Run tests
results = tools.test(code, "python", test_cases)
```

## Experiment Module

The experiment module provides a framework for conducting code translation experiments using the Rosetta Code dataset and language tools. It automates the workflow of:

1. Collecting examples from Rosetta Code
2. Generating test cases for each problem
3. Creating pseudocode from source language implementations
4. Translating pseudocode to target languages
5. Evaluating the correctness of translations
6. Generating reports with metrics

The module can optionally enhance translations with GraphRAG, which provides relevant documentation and code examples from the target language to improve translation quality.

### Language Support

The experiment uses languages from the TIOBE index (retrieved via `get_language_info()`) that are also verified to be supported by our language tools configuration through the `verify_language_tools()` validation function. This ensures that all languages used in the experiment are properly configured for compilation and execution.

### Local LLM Integration

The experiment framework uses a local HTTP server (default: http://127.0.0.1:1234/v1/chat/completions) for all LLM operations, making it compatible with locally hosted models. This eliminates the need for external API services and enables fully self-contained experimentation.

### Running Experiments

You can run experiments using the `run_experiment.py` script:

```bash
# Run a basic experiment with default settings
python run_experiment.py --name my_experiment

# Customize the experiment
python run_experiment.py --name custom_exp --source-language Python --target-languages C Java JavaScript --num-problems 10

# Use GraphRAG for enhanced translation with documentation context
python run_experiment.py --name graphrag_exp --use-graphrag --target-languages C++ Rust Go

# Use a different LLM API endpoint
python run_experiment.py --name local_llm --llm-api http://localhost:8000/v1/chat/completions
```

Available options:

- `--name`: Experiment name (default: timestamp-based name)
- `--output-dir`: Directory for experiment results (default: `experiment_results`)
- `--num-problems`: Number of problems to include (default: 5)
- `--min-implementations`: Minimum implementations required for a problem (default: 5)
- `--source-language`: Source language for translation (default: Python)
- `--target-languages`: Target languages to translate to (default: all supported languages)
- `--skip-pseudocode`: Skip the pseudocode generation step
- `--use-graphrag`: Enable GraphRAG to enhance translation with documentation context
- `--top-n-languages`: Number of top languages to include (default: 20)
- `--llm-api`: URL for the LLM API server (default: http://127.0.0.1:1234/v1/chat/completions)

### Experiment Results

The experiment generates comprehensive results including:
- Problem descriptions and implementations
- Generated test cases for each problem
- Pseudocode generated from source language
- Translations to target languages
- Compilation success rates for each language
- Test case pass rates for each language
- A detailed report with visualizations

Results are stored in the specified output directory with the experiment name.

### Programmatic Usage

You can also use the experiment module programmatically:

```python
from experiment import Experiment

# Initialize experiment
exp = Experiment(
    experiment_name="my_experiment",
    output_dir="experiment_results",
    llm_api_url="http://127.0.0.1:1234/v1/chat/completions"  # Use local LLM server
)

# Run complete experiment
metrics = exp.run_full_experiment(
    num_problems=5,
    source_language="Python",
    target_languages=["C", "Java", "JavaScript"],
    use_pseudocode=True,
    use_graphrag=True  # Enable GraphRAG enhancement
)

# Or run individual phases
problems = exp.generate_test_set(num_problems=5)
test_cases = exp.generate_test_cases(problems)
pseudocode = exp.generate_pseudocode(problems, source_language="Python")
translations = exp.translate_to_languages(
    problems, 
    target_languages=["C", "Java"], 
    pseudocode=pseudocode,
    use_graphrag=True  # Enable GraphRAG enhancement
)
results = exp.evaluate_translations(problems, translations, test_cases)
report = exp.generate_report()
```

## GraphRAG Implementation

We've implemented a Graph RAG (Retrieval Augmented Generation) system that combines knowledge graphs with vector embeddings to enhance the context retrieval process for LLMs.

### Features

- Builds knowledge graphs from documentation files
- Stores graph nodes and relationships in a vector database (Milvus)
- Implements hybrid retrieval that combines vector similarity with graph relationships
- Compatible with local LLMs via LLama.cpp
- Supports context-aware embeddings for improved semantic understanding

### Implementation Structure

- `rag.py` - Main GraphRAG class that orchestrates the retrieval and generation process
- `config.py` - Configuration settings and management
- `vectordb.py` - Milvus vector database operations
- `graph_processor.py` - Knowledge graph construction and querying
- `embedding.py` - Vector embedding generation and management
- `retriever.py` - Hybrid retrieval logic combining vector search with graph relationships

### Usage

```python
from graphrag import GraphRAG

# Initialize GraphRAG with default configuration
rag = GraphRAG()

# Index the knowledge graph and store in vector database
rag.index()

# Query with hybrid graph-vector retrieval 
results = rag.query("How do I implement a binary search in C?")

# Generate a response using local LLM with context
response = rag.generate("What's the difference between merge sort and quicksort?")
```

### How It Works

1. **Indexing**:
   - Parses markdown documentation from multiple languages
   - Builds a knowledge graph from file links and directory structures
   - Generates context-aware embeddings for each node
   - Stores graph data in Milvus collections

2. **Retrieval**:
   - Performs initial vector similarity search
   - Retrieves relevant relationships from the graph
   - Reranks results based on graph connectedness
   - Returns a combined set of the most relevant context

3. **Generation**:
   - Formats the retrieved context for the LLM
   - Sends prompt to local LLM (or external API)
   - Returns the generated response

This implementation allows for more accurate context retrieval by combining the strengths of vector search (semantic similarity) with graph-based knowledge representation (relationships and structure).

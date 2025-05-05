# Code Transliteration and Auto Optimizer

## Datasets

- [Rosetta Code](https://huggingface.co/datasets/christopher/rosetta-code)

For our experiments, we utilize the Rosetta Code dataset, which provides implementations of the same algorithms across multiple programming languages. We've converted this resource into a structured database to facilitate algorithmic translation and optimization experiments. Our focus is primarily on the TIOBE top 20 programming languages, which represent the most widely used languages in the industry.

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

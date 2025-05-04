# Code Transliteration and Auto Optimizer

## Dependencies

- Graphviz


## Datasets

- [Rosetta Code](https://huggingface.co/datasets/christopher/rosetta-code)

## Proposed External dependencies

- [aichat](https://github.com/sigoden/aichat)
- [llm-functions](https://github.com/sigoden/llm-functions)
- [mcp-cli](https://github.com/chrishayuk/mcp-cli)
- [fast-fetch](https://github.com/fastfetch-cli/fastfetch)

## General

**Overall Idea**

### Phase 1: Initialization

1. Select a research paper (code or no code).
2. Use code from paper/generate code with [paper2code](https://arxiv.org/abs/2504.17192).
3. Generate many strict test cases for correctness + performance evaluation framework.
4. Generate high level pseudocode for generic reference.

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

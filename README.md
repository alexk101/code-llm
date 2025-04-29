# Code Transliteration and Auto Optimizer

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
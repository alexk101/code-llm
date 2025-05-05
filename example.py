#!/usr/bin/env python3
"""
GraphRAG example usage script.

This script demonstrates how to use the GraphRAG system.
"""

import argparse
import os

# Import GraphRAG
from utils.rag.rag import GraphRAG


def main():
    """Run the GraphRAG example."""
    parser = argparse.ArgumentParser(description="GraphRAG example")
    parser.add_argument(
        "--config", default="graphrag_config.json", help="Path to config file"
    )
    parser.add_argument("--query", type=str, help="Query to execute")
    parser.add_argument("--index", action="store_true", help="Force rebuild index")
    parser.add_argument(
        "--language", type=str, default=None, help="Filter results by language"
    )
    parser.add_argument(
        "--llm-api",
        type=str,
        default="http://127.0.0.1:1234/v1/chat/completions",
        help="LLM API endpoint (default: http://127.0.0.1:1234/v1/chat/completions)",
    )
    parser.add_argument(
        "--graph-backend",
        type=str,
        default="networkx",
        choices=["csr", "networkx"],
        help="Graph backend to use (default: networkx)",
    )
    args = parser.parse_args()

    # Create and initialize GraphRAG
    print("Initializing GraphRAG...")
    rag = GraphRAG(args.config, graph_backend=args.graph_backend)

    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        print(f"Creating default config at {args.config}")
        cfg = rag.config
        cfg.save()

    # Update LLM API URL if specified
    if args.llm_api:
        rag.config.config["llm"]["provider"] = "http"
        rag.config.config["llm"]["api_url"] = args.llm_api
        rag.config.save()
        print(f"Updated LLM API URL to: {args.llm_api}")

    # Rebuild the index if requested
    if args.index:
        print("Building index...")
        rag.index(force_rebuild=True)

    # If no query provided, run the interactive mode
    if not args.query:
        interactive_mode(rag, args.language)
    else:
        # Execute a single query
        execute_query(rag, args.query, args.language)


def execute_query(rag, query, language=None):
    """Execute a single query and display results.

    Args:
        rag: GraphRAG instance
        query: Query string
        language: Optional language filter
    """
    print(f"\nQuery: {query}")

    # Apply language filter if provided
    filters = {"subject": language} if language else None

    # First, get retrieval results only to show what's being used
    print("\nRetrieving context...")
    retrieval_results = rag.query(
        query, top_k=5, rerank=True, filters=filters, verbose=True
    )

    # Display retrieved context
    print("\nRetrieved Context:")
    print("-----------------")
    for i, passage in enumerate(retrieval_results.get("passages", [])):
        print(
            f"{i + 1}. {passage['title']}({passage['subject']}, {passage['node_type']})"
        )
        if len(passage.get("text", "")) > 200:
            print(f"   {passage['text'][:200]}...")
        else:
            print(f"   {passage['text']}")

    # Generate response
    print("\nGenerating response...")
    response = rag.generate(
        query,
        top_k=5,
        rerank=True,
        filters=filters,
        verbose=False,
    )

    # Print the response
    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)


def interactive_mode(rag, language=None):
    """Run an interactive query session.

    Args:
        rag: GraphRAG instance
        language: Optional language filter
    """
    print("\nGraphRAG Interactive Mode")
    print("Enter your queries, or 'exit' to quit")
    print("=" * 80)

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break

        if not query:
            continue

        execute_query(rag, query, language)


if __name__ == "__main__":
    main()

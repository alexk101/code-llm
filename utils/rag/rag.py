"""
Graph RAG (Retrieval Augmented Generation) main module.

This module provides the core functionality for running Graph RAG queries using
knowledge graphs and vector databases together.
"""

from typing import Dict

from .config import GraphRAGConfig
from .embedding import EmbeddingManager
from .graph_processor import GraphProcessor
from .retriever import HybridRetriever
from .vectordb import MilvusManager


class GraphRAG:
    """Main GraphRAG class that orchestrates the retrieval and generation process."""

    def __init__(
        self, config_path: str = "graphrag_config.json", graph_backend: str = "networkx"
    ):
        """Initialize the GraphRAG system with configuration.

        Args:
            config_path: Path to the GraphRAG configuration file
            graph_backend: Graph backend to use ('csr' or 'networkx')
        """
        self.config = GraphRAGConfig(config_path)
        self.embedding_manager = EmbeddingManager(self.config)
        self.graph_processor = GraphProcessor(self.config)
        # Set the graph backend
        if hasattr(self.graph_processor.markdown_graph, "backend"):
            self.graph_processor.markdown_graph.backend = graph_backend.lower()
        self.vector_db = MilvusManager(self.config)
        self.retriever = HybridRetriever(
            self.vector_db, self.graph_processor, self.embedding_manager, self.config
        )

    def index(self, force_rebuild: bool = False):
        """Index all resources into the vector database and build the knowledge graph.

        Args:
            force_rebuild: If True, rebuild the indices even if they exist
        """
        # Process the graph
        self.graph_processor.build_graph(force_rebuild=force_rebuild)

        # Generate embeddings for nodes
        node_embeddings = self.embedding_manager.embed_graph_nodes(
            self.graph_processor.get_graph_nodes()
        )

        # Initialize or update vector collections
        self.vector_db.initialize_collections()

        # Index nodes and relationships in vector database
        self.vector_db.index_passages(node_embeddings)
        self.vector_db.index_relationships(
            self.graph_processor.get_graph_relationships()
        )

        print(f"Indexed {len(node_embeddings)} nodes with their relationships")

    def query(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        filters: Dict = None,
        verbose: bool = False,
    ) -> Dict:
        """Run a query through the GraphRAG system.

        Args:
            query: The user query
            top_k: Number of results to return
            rerank: Whether to rerank results using relationship data
            filters: Optional filters to apply to
                vector search (e.g., {"subject": "python"})
            verbose: If True, print detailed retrieval information

        Returns:
            Dict containing the results and context information
        """
        # Encode the query
        query_embedding = self.embedding_manager.embed_query(query)

        # Retrieve relevant information using hybrid approach
        results = self.retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
            verbose=verbose,
        )

        return results

    def generate(
        self,
        query: str,
        llm_provider: str = None,
        top_k: int = 5,
        rerank: bool = True,
        filters: Dict = None,
        verbose: bool = False,
    ) -> str:
        """Generate a response using the LLM with context from GraphRAG retrieval.

        Args:
            query: The user query
            llm_provider: Optional override for LLM provider
            top_k: Number of results to include in context
            rerank: Whether to rerank results
            filters: Optional filters to apply to vector search
            verbose: If True, print detailed information

        Returns:
            Generated response from the LLM
        """
        # Get retrieval results
        results = self.query(
            query, top_k=top_k, rerank=rerank, filters=filters, verbose=verbose
        )

        # Use configured or specified LLM provider
        provider = llm_provider or self.config.config["llm"]["provider"]

        # Format the context for the LLM
        context = self._format_context_for_llm(results)

        # Generate response using the retriever's LLM integration
        response = self.retriever.generate_response(query, context, provider)

        return response

    def _format_context_for_llm(self, results: Dict) -> str:
        """Format the retrieved results into a context string for the LLM.

        Args:
            results: The retrieval results

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get context length limits from config or use defaults
        llm_config = self.config.config.get("llm", {})
        context_limits = llm_config.get("context_limits", {})

        # Set a maximum length for each passage text to avoid context overflow
        max_text_length = context_limits.get(
            "max_passage_length", 1000
        )  # Characters per passage
        max_total_length = context_limits.get(
            "max_total_length", 8000
        )  # Total characters for context
        max_title_length = context_limits.get(
            "max_title_length", 100
        )  # Maximum title length

        total_length = 0

        for i, item in enumerate(results.get("passages", [])):
            # Get the title and limit its length
            title = item.get("title", "Untitled")
            if len(title) > max_title_length:
                title = title[: max_title_length - 3] + "..."

            # Format the header
            header = f"[{i + 1}] {title}"
            context_parts.append(header)

            # Get and truncate the text if needed
            text = item.get("text", "")
            if len(text) > max_text_length:
                text = text[: max_text_length - 3] + "..."

            context_parts.append(text)
            context_parts.append("")  # Empty line for separation

            # Track total length
            total_length += len(header) + len(text) + 1  # +1 for the separator

            # Stop adding more if we've reached the max total length
            if total_length >= max_total_length:
                context_parts.append(
                    "... (additional context omitted due to length constraints)"
                )
                break

        return "\n".join(context_parts)


def main():
    """Command-line entry point for GraphRAG."""
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG: Graph-based RAG system")
    parser.add_argument(
        "--config", default="graphrag_config.json", help="Path to config file"
    )
    parser.add_argument("--index", action="store_true", help="Index content")
    parser.add_argument("--force", action="store_true", help="Force reindexing")
    parser.add_argument("--query", type=str, help="Query to run")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    graph_rag = GraphRAG(args.config)

    if args.index:
        graph_rag.index(force_rebuild=args.force)

    if args.query:
        results = graph_rag.generate(args.query, top_k=args.top_k, verbose=True)
        print("\nGenerated Response:")
        print(results)


if __name__ == "__main__":
    main()

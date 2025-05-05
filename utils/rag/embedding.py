"""
Embedding module for generating and managing vector embeddings.

This module handles the creation of embeddings for queries, nodes,
and relationships using local or remote embedding models.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .config import GraphRAGConfig


class EmbeddingManager:
    """Manages embeddings for GraphRAG."""

    def __init__(self, config: GraphRAGConfig):
        """Initialize the embedding manager.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.embedding_config = config.embedding
        self.model_type = self.embedding_config["model"]
        self.model_name = self.embedding_config["model_name"]
        self.batch_size = self.embedding_config["batch_size"]
        self.context_aware = self.embedding_config["context_aware"]

        # Set cache path
        cache_file = config.graph["embeddings_cache"]
        self.embeddings_cache_path = Path(cache_file)

        # Initialize embeddings cache
        self.embeddings_cache = {}
        self._load_embeddings_cache()

        # Load the embedding model
        self.model = self._load_embedding_model()

    def _load_embedding_model(self):
        """Load the appropriate embedding model based on configuration.

        Returns:
            Embedding model instance
        """
        model_type = self.model_type.lower()

        if model_type == "local":
            try:
                from sentence_transformers import SentenceTransformer

                print(f"Loading local embedding model: {self.model_name}")
                return SentenceTransformer(self.model_name)
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers package is required for local embeddings. "
                    "Please install it in your environment using: "
                    "pip install sentence-transformers"
                ) from err

        elif model_type == "huggingface":
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                print(f"Loading HuggingFace model: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)

                # Return a wrapper class that mimics the encode method
                class HFEmbedder:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.model.to(self.device)

                    def mean_pooling(self, model_output, attention_mask):
                        token_embeddings = model_output[0]
                        input_mask_expanded = (
                            attention_mask.unsqueeze(-1)
                            .expand(token_embeddings.size())
                            .float()
                        )
                        return torch.sum(
                            token_embeddings * input_mask_expanded, 1
                        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                    def encode(self, texts, **kwargs):
                        if isinstance(texts, str):
                            texts = [texts]

                        # Tokenize and prepare for the model
                        encoded_input = self.tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(self.device)

                        # Compute embeddings
                        with torch.no_grad():
                            model_output = self.model(**encoded_input)

                        # Mean pooling
                        embeddings = self.mean_pooling(
                            model_output, encoded_input["attention_mask"]
                        )
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                        # Convert to numpy and return
                        np_embeddings = embeddings.cpu().numpy()
                        if len(np_embeddings) == 1:
                            return np_embeddings[0]
                        return np_embeddings

                return HFEmbedder(model, tokenizer)

            except ImportError as err:
                raise ImportError(
                    "Missing dependencies. "
                    "Please install them in your environment using: "
                    "pip install transformers torch"
                ) from err

        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")

    def _load_embeddings_cache(self):
        """Load embeddings from cache file if it exists."""
        if self.embeddings_cache_path.exists():
            try:
                print(f"Loading embeddings cache from {self.embeddings_cache_path}")
                cache_data = np.load(self.embeddings_cache_path, allow_pickle=True)

                # Extract and rebuild the cache dictionary
                node_ids = cache_data["node_ids"]
                embeddings = cache_data["embeddings"]

                if len(node_ids) == len(embeddings):
                    for i, node_id in enumerate(node_ids):
                        self.embeddings_cache[str(node_id)] = embeddings[i]

                    print(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
                else:
                    print("Warning: Embeddings cache appears corrupted")
            except Exception as e:
                print(f"Error loading embeddings cache: {e}")
                self.embeddings_cache = {}

    def _save_embeddings_cache(self):
        """Save embeddings to cache file."""
        try:
            print(f"Saving {len(self.embeddings_cache)} embeddings to cache")

            # Convert to arrays for saving
            node_ids = np.array(list(self.embeddings_cache.keys()), dtype=object)
            embeddings = np.array(list(self.embeddings_cache.values()), dtype=object)

            # Save to npz file
            np.savez_compressed(
                self.embeddings_cache_path, node_ids=node_ids, embeddings=embeddings
            )

            print(f"Embeddings cache saved to {self.embeddings_cache_path}")
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a query.

        Args:
            query: The query text

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(query)
            return embedding.tolist() if hasattr(embedding, "tolist") else embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            # Return a zero vector with the correct dimension as fallback
            return [0.0] * self.config.vector_db["dimension"]

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist() if hasattr(embedding, "tolist") else embedding
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            # Return a zero vector with the correct dimension as fallback
            return [0.0] * self.config.vector_db["dimension"]

    def embed_graph_nodes(self, nodes: Dict[str, Dict]) -> Dict[str, Dict]:
        """Generate embeddings for all graph nodes.

        Args:
            nodes: Dictionary of node ID to node data

        Returns:
            Dictionary of node ID to node data with embeddings
        """
        print(f"Generating embeddings for {len(nodes)} nodes")

        # Track nodes that need embedding
        nodes_to_embed = {}
        for node_id, node_data in nodes.items():
            if node_id in self.embeddings_cache:
                # Use cached embedding
                node_data["embedding"] = self.embeddings_cache[node_id]
            else:
                # Add to the list to be embedded
                nodes_to_embed[node_id] = node_data

        print(
            f"Using {len(self.embeddings_cache)} cached embeddings, ",
            f"generating {len(nodes_to_embed)} new embeddings",
        )

        if not nodes_to_embed:
            print("All node embeddings were cached, no new embeddings to generate")
            return nodes

        # Process in batches
        batch_size = self.batch_size
        node_ids = list(nodes_to_embed.keys())
        total_batches = (len(node_ids) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(node_ids), batch_size),
            desc="Generating embeddings",
            total=total_batches,
        ):
            batch_node_ids = node_ids[i : i + batch_size]

            # Generate embeddings with different strategies based on node type
            batch_texts = []
            for node_id in batch_node_ids:
                node_data = nodes_to_embed[node_id]

                if self.context_aware:
                    # Context-aware: combine title, subject, and
                    # text with type information
                    node_type = node_data.get("type", "unknown")
                    subject = node_data.get("subject", "unknown")
                    title = node_data.get("title", "")
                    text = node_data.get("text", "")

                    # Use a template based on node type
                    if node_type == "file":
                        context_text = (
                            f"Title: {title}\nSubject: {subject}\nType: {node_type}\n\n"
                            f"{text}"
                        )
                    elif node_type == "module":
                        context_text = (
                            f"Module: {title}\nSubject: {subject}\nType: {node_type}"
                        )
                    elif node_type == "subject":
                        context_text = f"Subject: {subject}\nType: {node_type}"
                    else:
                        context_text = f"{title} ({node_type}, {subject}): {text}"

                    batch_texts.append(context_text)
                else:
                    # Simple approach: just use the text or title
                    text = node_data.get("text", "")
                    if not text:
                        text = node_data.get("title", "")
                    batch_texts.append(text)

            try:
                # Generate embeddings for the batch
                batch_embeddings = self.model.encode(batch_texts)

                # Convert to list format if needed
                if len(batch_node_ids) == 1 and not isinstance(batch_embeddings, list):
                    batch_embeddings = [batch_embeddings]

                # Apply embeddings to nodes and update cache
                for j, node_id in enumerate(batch_node_ids):
                    embedding = batch_embeddings[j]
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()

                    nodes[node_id]["embedding"] = embedding
                    self.embeddings_cache[node_id] = embedding

            except Exception as e:
                print(f"Error generating batch embeddings: {e}")

                # Generate fallback embeddings individually
                print("Falling back to individual embedding generation")
                for node_id in batch_node_ids:
                    try:
                        text = batch_texts[batch_node_ids.index(node_id)]
                        embedding = self.model.encode(text)
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()

                        nodes[node_id]["embedding"] = embedding
                        self.embeddings_cache[node_id] = embedding
                    except Exception as inner_e:
                        print(
                            f"Error generating embedding for node {node_id}: {inner_e}"
                        )
                        # Use a zero vector as fallback
                        nodes[node_id]["embedding"] = [0.0] * self.config.vector_db[
                            "dimension"
                        ]

        # Save the cache
        self._save_embeddings_cache()

        return nodes

    def embed_relationship(
        self, relationship: Dict, source_node: Dict, target_node: Dict
    ) -> List[float]:
        """Generate an embedding for a relationship.

        Args:
            relationship: Relationship dictionary
            source_node: Source node dictionary
            target_node: Target node dictionary

        Returns:
            Embedding vector
        """
        try:
            # Combine information about the relationship and its nodes
            relation_type = relationship.get("relation_type", "linked_to")
            source_title = source_node.get("title", "")
            target_title = target_node.get("title", "")
            description = relationship.get("description", "")

            # Create a text representation of the relationship
            if description:
                text = description
            else:
                text = f"{source_title} {relation_type} {target_title}"

            # Generate the embedding
            embedding = self.model.encode(text)
            return embedding.tolist() if hasattr(embedding, "tolist") else embedding

        except Exception as e:
            print(f"Error generating relationship embedding: {e}")
            # Return a zero vector with the correct dimension as fallback
            return [0.0] * self.config.vector_db["dimension"]

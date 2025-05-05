"""
Vector database module for Milvus integration.

This module handles all operations related to the Milvus vector database,
including collection creation, data insertion, and vector search.
"""

from typing import Dict, List

import numpy as np
from pymilvus import DataType, MilvusClient

from .config import GraphRAGConfig


class MilvusManager:
    """Manages Milvus vector database operations for GraphRAG."""

    def __init__(self, config: GraphRAGConfig):
        """Initialize the Milvus connection and collections.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.vector_config = config.vector_db

        # Initialize Milvus client
        self.client = MilvusClient(
            uri=self.vector_config["uri"], token=self.vector_config.get("token")
        )

        # Collection names
        self.passage_collection = self.vector_config["passage_collection"]
        self.relation_collection = self.vector_config["relation_collection"]

        # Collection configs
        self.dimension = self.vector_config["dimension"]
        self.metric_type = self.vector_config["metric_type"]

        # Get index configs from config or use defaults
        self.index_config = self.vector_config.get("index_config", {})
        self.index_type = self.index_config.get("index_type", "IVF_FLAT")
        self.index_params = self.index_config.get("params", {"nlist": 1024})
        self.search_params = self.index_config.get("search_params", {"ef": 64})

    def initialize_collections(self):
        """Initialize or validate the Milvus collections."""
        # Sample a vector to determine the actual dimension first

        # Check if collections exist
        has_passage_collection = self._has_collection(self.passage_collection)
        has_relation_collection = self._has_collection(self.relation_collection)

        # Create collections if they don't exist
        if not has_passage_collection:
            self._create_passage_collection()

        if not has_relation_collection:
            self._create_relationship_collection()

    def _has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if the collection exists
        """
        return collection_name in self.client.list_collections()

    def _create_passage_collection(self):
        """Create the passages collection in Milvus."""
        print(f"Creating collection: {self.passage_collection}")

        # Create schema first
        schema = self.client.create_schema()

        # Add fields to schema
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        schema.add_field(
            field_name="node_id", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="subject", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="node_type", datatype=DataType.VARCHAR, max_length=50
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension
        )

        # Create collection using schema
        self.client.create_collection(
            collection_name=self.passage_collection, schema=schema
        )

        # Prepare index parameters for embedding field
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=self.index_params,
        )

        # Create index on embedding field
        self.client.create_index(
            collection_name=self.passage_collection, index_params=index_params
        )

        print(f"Collection {self.passage_collection} created and indexed")

    def _create_relationship_collection(self):
        """Create the relationships collection in Milvus."""
        print(f"Creating collection: {self.relation_collection}")

        # Create schema first
        schema = self.client.create_schema()

        # Add fields to schema
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        schema.add_field(
            field_name="source_id", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="target_id", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="relation_type", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(field_name="weight", datatype=DataType.FLOAT)
        schema.add_field(
            field_name="description", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension
        )

        # Create collection using schema
        self.client.create_collection(
            collection_name=self.relation_collection, schema=schema
        )

        # Prepare index parameters for embedding field
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=self.index_params,
        )

        # Create index on embedding field
        self.client.create_index(
            collection_name=self.relation_collection, index_params=index_params
        )

        print(f"Collection {self.relation_collection} created and indexed")

    def index_passages(self, node_embeddings: Dict[str, Dict]):
        """Index node data as passages in Milvus.

        Args:
            node_embeddings: Dictionary of node ID to node data with embeddings
        """
        if not node_embeddings:
            print("No node embeddings provided for indexing")
            return

        print(
            f"Indexing {len(node_embeddings)} passages into {self.passage_collection}"
        )

        # Prepare data for batch insertion
        entities = []
        batch_size = min(1000, len(node_embeddings))  # Limit batch size

        # Check sample embedding to confirm dimension before creating collections
        sample_keys = list(node_embeddings.keys())

        if sample_keys:
            sample_embedding = node_embeddings[sample_keys[0]].get("embedding", [])
            if (
                isinstance(sample_embedding, np.ndarray)
                and len(sample_embedding.shape) > 0
            ):
                actual_dim = sample_embedding.shape[0]
                if actual_dim != self.dimension:
                    print(
                        f"WARNING: Updating dimension from {self.dimension}",
                        f"to {actual_dim}",
                    )
                    self.dimension = actual_dim

                    # We need to drop and recreate collections
                    # with the correct dimension
                    if self._has_collection(self.passage_collection):
                        print(
                            f"Dropping collection {self.passage_collection}",
                            "to recreate with correct dimension",
                        )
                        self.client.drop_collection(self.passage_collection)
                        self._create_passage_collection()

                    if self._has_collection(self.relation_collection):
                        print(
                            f"Dropping collection {self.relation_collection}",
                            "to recreate with correct dimension",
                        )
                        self.client.drop_collection(self.relation_collection)
                        self._create_relationship_collection()

        # If dimensions weren't detected from sample
        # or collections don't exist, create them
        if not self._has_collection(self.passage_collection):
            self._create_passage_collection()

        if not self._has_collection(self.relation_collection):
            self._create_relationship_collection()

        for node_id, node_data in node_embeddings.items():
            # Ensure embedding is a numpy float32 array
            embedding = node_data.get("embedding", [])

            # Check if embedding exists and handle type conversion safely
            if embedding is None or (
                isinstance(embedding, list) and len(embedding) == 0
            ):
                # Skip if no embedding is available
                continue

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            elif embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)

            # Verify dimension
            if embedding.shape[0] != self.dimension:
                print(
                    f"WARNING: Skipping vector with dimension {embedding.shape[0]},",
                    f"expected {self.dimension}",
                )
                continue

            # Extract fields
            entity = {
                "id": node_id,
                "node_id": node_id,
                "title": node_data.get("title", ""),
                "text": node_data.get("text", ""),
                "subject": node_data.get("subject", "unknown"),
                "node_type": node_data.get("type", "unknown"),
                "embedding": embedding,
            }
            entities.append(entity)

            # Insert in batches for better performance
            if len(entities) >= batch_size:
                self._insert_passage_batch(entities)
                entities = []

        # Insert any remaining entities
        if entities:
            self._insert_passage_batch(entities)

        print(f"Indexed {len(node_embeddings)} passages successfully")

    def _insert_passage_batch(self, entities: List[Dict]):
        """Insert a batch of passage entities.

        Args:
            entities: List of passage dictionaries to insert
        """
        try:
            _ = self.client.insert(
                collection_name=self.passage_collection, data=entities
            )
            # Wait for the insert to complete
            self.client.flush(self.passage_collection)
        except Exception as e:
            print(f"Error inserting batch: {e}")

    def index_relationships(self, relationships: List[Dict]):
        """Index relationship data in Milvus.

        Args:
            relationships: List of relationship dictionaries
        """
        if not relationships:
            print("No relationships provided for indexing")
            return

        output = (
            f"Indexing {len(relationships)} ",
            f"relationships into {self.relation_collection}",
        )
        print(output)

        # Prepare data for batch insertion
        entities = []
        batch_size = min(1000, len(relationships))  # Limit batch size

        for i, rel in enumerate(relationships):
            # Generate a unique ID for the relationship
            rel_id = f"rel_{i}_{rel['source_id']}_{rel['target_id']}"

            # Ensure embedding is a numpy float32 array
            embedding = rel.get("embedding", [])

            # Check if embedding exists and handle type conversion safely
            if embedding is None or (
                isinstance(embedding, list) and len(embedding) == 0
            ):
                # Skip if no embedding is available
                continue

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            elif embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)

            entity = {
                "id": rel_id,
                "source_id": rel["source_id"],
                "target_id": rel["target_id"],
                "relation_type": rel.get("relation_type", "linked_to"),
                "weight": rel.get("weight", 1.0),
                "description": rel.get("description", ""),
                "embedding": embedding,
            }
            entities.append(entity)

            # Insert in batches for better performance
            if len(entities) >= batch_size:
                self._insert_relationship_batch(entities)
                entities = []

        # Insert any remaining entities
        if entities:
            self._insert_relationship_batch(entities)

        print(f"Indexed {len(relationships)} relationships successfully")

    def _insert_relationship_batch(self, entities: List[Dict]):
        """Insert a batch of relationship entities.

        Args:
            entities: List of relationship dictionaries to insert
        """
        try:
            _ = self.client.insert(
                collection_name=self.relation_collection, data=entities
            )
            # Wait for the insert to complete
            self.client.flush(self.relation_collection)
        except Exception as e:
            print(f"Error inserting relationship batch: {e}")

    def search_passages(
        self, query_embedding: List[float], top_k: int = 10, filters: Dict = None
    ) -> List[Dict]:
        """Search for passages by vector similarity.

        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of passage results
        """
        # Ensure query_embedding is a NumPy array of correct dimensions
        import numpy as np

        # Convert to numpy array if it's not already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Ensure it's float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Check dimensions
        if query_embedding.shape[0] != self.dimension:
            print(
                f"WARNING: Query vector dimension {query_embedding.shape[0]}",
                f"doesn't match collection dimension {self.dimension}",
            )
            # Resize the vector if needed (this is a simplistic approach)
            if query_embedding.shape[0] < self.dimension:
                # Pad with zeros
                padded = np.zeros(self.dimension, dtype=np.float32)
                padded[: query_embedding.shape[0]] = query_embedding
                query_embedding = padded
            else:
                # Truncate
                query_embedding = query_embedding[: self.dimension]

        search_params = {"metric_type": self.metric_type, "params": self.search_params}

        # Build filter expression if provided
        expr = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Handle IN operator
                    values_str = ", ".join([f'"{v}"' for v in value])
                    conditions.append(f"{key} in [{values_str}]")
                else:
                    # Handle equality
                    conditions.append(f'{key} == "{value}"')

            if conditions:
                expr = " && ".join(conditions)

        try:
            # Only pass expr if it's not None to avoid the multiple values error
            search_args = {
                "collection_name": self.passage_collection,
                "data": [query_embedding.tolist()],
                "anns_field": "embedding",
                "params": search_params,
                "limit": top_k,
                "output_fields": ["node_id", "title", "text", "subject", "node_type"],
            }

            if expr is not None:
                search_args["expr"] = expr

            results = self.client.search(**search_args)

            # Format the results
            passages = []
            if results and len(results) > 0:
                for hit in results[0]:
                    # Safely extract fields with defaults if missing
                    entity = hit.get("entity", {})
                    passage = {
                        "id": entity.get("node_id", ""),
                        "title": entity.get("title", ""),
                        "text": entity.get("text", ""),
                        "subject": entity.get("subject", "unknown"),
                        "node_type": entity.get("node_type", "unknown"),
                        # Handle both possible score field names
                        "score": hit.get("score", hit.get("distance", 0.0)),
                    }
                    passages.append(passage)

            return passages
        except Exception as e:
            print(f"Error searching passages: {e}")
            return []

    def search_relationships(
        self, query_embedding: List[float], top_k: int = 20
    ) -> List[Dict]:
        """Search for relationships by vector similarity.

        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return

        Returns:
            List of relationship results
        """
        # Ensure query_embedding is a NumPy array of correct dimensions
        import numpy as np

        # Convert to numpy array if it's not already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Ensure it's float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Check dimensions
        if query_embedding.shape[0] != self.dimension:
            print(
                f"WARNING: Query vector dimension {query_embedding.shape[0]}",
                f"doesn't match collection dimension {self.dimension}",
            )
            # Resize the vector if needed (this is a simplistic approach)
            if query_embedding.shape[0] < self.dimension:
                # Pad with zeros
                padded = np.zeros(self.dimension, dtype=np.float32)
                padded[: query_embedding.shape[0]] = query_embedding
                query_embedding = padded
            else:
                # Truncate
                query_embedding = query_embedding[: self.dimension]

        search_params = {"metric_type": self.metric_type, "params": self.search_params}

        try:
            results = self.client.search(
                collection_name=self.relation_collection,
                data=[query_embedding.tolist()],
                anns_field="embedding",
                params=search_params,
                limit=top_k,
                output_fields=[
                    "source_id",
                    "target_id",
                    "relation_type",
                    "weight",
                    "description",
                ],
            )

            # Format the results
            relationships = []
            if results and len(results) > 0:
                for hit in results[0]:
                    # Safely extract fields with defaults if missing
                    entity = hit.get("entity", {})
                    relationship = {
                        "source_id": entity.get("source_id", ""),
                        "target_id": entity.get("target_id", ""),
                        "relation_type": entity.get("relation_type", ""),
                        "weight": entity.get("weight", 1.0),
                        "description": entity.get("description", ""),
                        # Handle both possible score field names
                        "score": hit.get("score", hit.get("distance", 0.0)),
                    }
                    relationships.append(relationship)

            return relationships
        except Exception as e:
            print(f"Error searching relationships: {e}")
            return []

    def get_relationships_for_node(self, node_id: str) -> List[Dict]:
        """Get all relationships involving a specific node.

        Args:
            node_id: The node ID to find relationships for

        Returns:
            List of relationship dictionaries
        """
        try:
            # Search for relationships where the node is either source or target
            expr = f'source_id == "{node_id}" || target_id == "{node_id}"'

            results = self.client.query(
                collection_name=self.relation_collection,
                filter=expr,
                output_fields=[
                    "source_id",
                    "target_id",
                    "relation_type",
                    "weight",
                    "description",
                ],
            )

            return results
        except Exception as e:
            print(f"Error getting relationships for node {node_id}: {e}")
            return []

    def drop_collections(self):
        """Drop all collections (useful for rebuilding the database)."""
        try:
            if self._has_collection(self.passage_collection):
                self.client.drop_collection(self.passage_collection)
                print(f"Dropped collection {self.passage_collection}")

            if self._has_collection(self.relation_collection):
                self.client.drop_collection(self.relation_collection)
                print(f"Dropped collection {self.relation_collection}")
        except Exception as e:
            print(f"Error dropping collections: {e}")

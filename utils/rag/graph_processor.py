"""
Graph processor module for knowledge graph operations.

This module handles knowledge graph construction, processing, and querying,
building on the existing MarkdownGraph implementation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from utils.graphs.markdown_graph import MarkdownGraph

from .config import GraphRAGConfig


class GraphProcessor:
    """Handles knowledge graph operations for GraphRAG."""

    def __init__(self, config: GraphRAGConfig):
        """Initialize the graph processor.

        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.graph_config = config.graph

        # Set up paths
        self.resources_dir = Path(self.graph_config["resources_dir"])
        self.graph_cache = self.graph_config["graph_cache"]

        # Initialize the graph with NetworkX backend if available, fallback to CSR
        self.markdown_graph = MarkdownGraph(
            resources_dir=str(self.resources_dir), backend="networkx"
        )
        self.graph_built = False

        # Node and relationship caches
        self.nodes = {}
        self.relationships = []

    def build_graph(self, force_rebuild: bool = False):
        """Build or load the knowledge graph.

        Args:
            force_rebuild: If True, rebuild the graph even if a cache exists
        """
        if force_rebuild and os.path.exists(self.graph_cache):
            print(f"Removing existing graph cache: {self.graph_cache}")
            os.remove(self.graph_cache)

        print("Building knowledge graph from markdown resources...")
        # Use the existing MarkdownGraph to build the graph
        self.markdown_graph.build_graph(graph_cache=self.graph_cache)

        if (
            self.markdown_graph.graph is None
            and self.markdown_graph.networkx_graph is None
        ):
            raise RuntimeError("Failed to build graph - check for errors in the log")

        # Process nodes and relationships
        self._extract_nodes_and_relationships()
        self.graph_built = True

        # Print basic stats
        stats = self.markdown_graph.calculate_graph_stats()
        print(f"Graph built with {stats.get('num_nodes', 0)} nodes")
        print(f"Node types: {', '.join(stats.get('node_types', {}).keys())}")

    def _extract_nodes_and_relationships(self):
        """Extract nodes and relationships from the graph.

        This processes the internal graph structure and creates dictionaries
        for nodes and relationships with additional metadata.
        """
        if (
            self.markdown_graph.graph is None
            and self.markdown_graph.networkx_graph is None
        ):
            raise RuntimeError("Graph not built yet")

        print("Extracting nodes and relationships from graph...")

        # Process nodes
        for node_id, node_str in self.markdown_graph.node_id_to_str.items():
            # Get node metadata
            metadata = self.markdown_graph.node_id_to_metadata.get(node_id, {})
            node_type = metadata.get("type", "unknown")
            subject = metadata.get("subject", "unknown")

            # Construct a title from the node string
            if node_type == "file":
                # For files, use the filename without path or extension
                title = Path(node_str).stem
            elif node_type == "module":
                # For modules (directories), use the last directory name
                title = Path(node_str).name
            elif node_type == "subject":
                # For subject nodes, use the subject name
                title = node_str.split(":", 1)[1] if ":" in node_str else node_str
            else:
                title = node_str

            # Get node content if it's a file
            content = ""
            if node_type == "file":
                file_path = self.resources_dir / node_str
                if file_path.exists():
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="replace"
                        ) as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading content for {node_str}: {e}")

            # Store the node
            self.nodes[str(node_id)] = {
                "id": str(node_id),
                "node_str": node_str,
                "title": title,
                "text": content,
                "type": node_type,
                "subject": subject,
            }

        # Process relationships (edges) using the new get_edges method
        print("Extracting edges from graph...")
        edges = self.markdown_graph.get_edges()
        print(f"Found {len(edges)} edges in the graph")

        # Process each edge into a relationship
        for source, target in edges:
            # Convert indices to node IDs
            source_id = str(source)
            target_id = str(target)

            # Extract source and target node info
            source_node = self.nodes.get(source_id, {})
            target_node = self.nodes.get(target_id, {})

            # Skip self-loops
            if source_id == target_id:
                continue

            # Determine relationship type based on node types
            source_type = source_node.get("type", "unknown")
            target_type = target_node.get("type", "unknown")

            if source_type == "file" and target_type == "file":
                relation_type = "references"
            elif source_type == "file" and target_type == "module":
                relation_type = "belongs_to"
            elif source_type == "file" and target_type == "subject":
                relation_type = "categorized_as"
            elif source_type == "module" and target_type == "module":
                relation_type = "contains"
            else:
                relation_type = "linked_to"

            # Create a description
            source_name = source_node.get("title", source_id)
            target_name = target_node.get("title", target_id)
            description = f"{source_name} {relation_type} {target_name}"

            # Add the relationship
            self.relationships.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "weight": 1.0,  # Default weight
                    "description": description,
                }
            )

        print(
            f"Extracted {len(self.nodes)} nodes and ",
            f"{len(self.relationships)} relationships",
        )

    def get_graph_nodes(self) -> Dict[str, Dict]:
        """Get all nodes in the graph with their metadata.

        Returns:
            Dictionary mapping node IDs to node data
        """
        if not self.graph_built:
            raise RuntimeError("Graph not built yet. Call build_graph() first.")
        return self.nodes

    def get_graph_relationships(self) -> List[Dict]:
        """Get all relationships in the graph.

        Returns:
            List of relationship dictionaries
        """
        if not self.graph_built:
            raise RuntimeError("Graph not built yet. Call build_graph() first.")
        return self.relationships

    def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Get a node by its ID.

        Args:
            node_id: The node ID to look up

        Returns:
            Node dictionary or None if not found
        """
        return self.nodes.get(node_id)

    def get_node_by_path(self, path: str) -> Optional[Dict]:
        """Get a node by its path/node_str.

        Args:
            path: The node string path to look up

        Returns:
            Node dictionary or None if not found
        """
        for node in self.nodes.values():
            if node["node_str"] == path:
                return node
        return None

    def get_related_nodes(
        self, node_id: str, relation_types: List[str] = None
    ) -> List[Dict]:
        """Get nodes directly related to the given node.

        Args:
            node_id: The source node ID
            relation_types: Optional list of relation types to filter by

        Returns:
            List of related node dictionaries
        """
        related_nodes = []

        for rel in self.relationships:
            # Check if this relationship involves the node
            if rel["source_id"] == node_id:
                # Node is the source
                target_id = rel["target_id"]
                if relation_types is None or rel["relation_type"] in relation_types:
                    target_node = self.get_node_by_id(target_id)
                    if target_node:
                        related_nodes.append(
                            {
                                **target_node,
                                "relation": rel["relation_type"],
                                "direction": "outgoing",
                            }
                        )
            elif rel["target_id"] == node_id:
                # Node is the target
                source_id = rel["source_id"]
                if relation_types is None or rel["relation_type"] in relation_types:
                    source_node = self.get_node_by_id(source_id)
                    if source_node:
                        related_nodes.append(
                            {
                                **source_node,
                                "relation": rel["relation_type"],
                                "direction": "incoming",
                            }
                        )

        return related_nodes

    def get_nodes_by_subject(self, subject: str) -> List[Dict]:
        """Get all nodes for a specific subject.

        Args:
            subject: The subject to filter by

        Returns:
            List of node dictionaries for the subject
        """
        return [node for node in self.nodes.values() if node["subject"] == subject]

    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type.

        Args:
            node_type: The node type to filter by

        Returns:
            List of node dictionaries of the specified type
        """
        return [node for node in self.nodes.values() if node["type"] == node_type]

    def get_shortest_path(self, start_node_id: str, end_node_id: str) -> List[Dict]:
        """Find the shortest path between two nodes.

        Args:
            start_node_id: The starting node ID
            end_node_id: The ending node ID

        Returns:
            List of nodes along the shortest path, or empty list if no path exists
        """
        # Build an adjacency list for BFS
        adjacency = {}
        for rel in self.relationships:
            source_id = rel["source_id"]
            target_id = rel["target_id"]

            if source_id not in adjacency:
                adjacency[source_id] = []
            if target_id not in adjacency:
                adjacency[target_id] = []

            # Add bidirectional edges for the BFS
            adjacency[source_id].append(target_id)
            adjacency[target_id].append(source_id)

        # Run BFS
        visited = {start_node_id}
        queue = [(start_node_id, [start_node_id])]

        while queue:
            current, path = queue.pop(0)

            if current == end_node_id:
                # Found the path
                return [self.get_node_by_id(node_id) for node_id in path]

            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        return []

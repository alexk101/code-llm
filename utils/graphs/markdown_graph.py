"""
Markdown graph module for constructing and manipulating graphs from markdown files.
"""

import multiprocessing
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import csrgraph as cg
import networkx as nx
import numpy as np
from nodevectors import GGVec, Glove, GraRep, Node2Vec, ProNE, SKLearnEmbedder
from scipy import sparse
from tqdm import tqdm

# Determine the optimal number of CPU cores for processing
CPU_COUNT = multiprocessing.cpu_count()


class MarkdownGraph:
    """Class to create and manipulate a graph of markdown files and their links."""

    def __init__(self, resources_dir="cache/md_resources", backend="csr"):
        """Initialize the graph with the path to the markdown resources.

        Args:
            resources_dir: Path to the markdown resources directory
            backend: Graph backend to use, 'csr' or 'networkx' (default: 'csr')
        """
        self.resources_dir = Path(resources_dir)
        self.graph = None
        self.networkx_graph = None  # For storing the NetworkX graph
        self.backend = backend.lower()

        # Validate backend choice
        if self.backend not in ["csr", "networkx"]:
            print(f"Warning: Unknown backend '{backend}'. Falling back to 'csr'.")
            self.backend = "csr"

        self.http_node_str = (
            "external-http"  # Special node name for all external HTTP links
        )
        self.md_files = []
        self.subjects = {}  # Map of directory names to normalized subjects
        self.modules = set()  # Set of module directory names (strings)

        # Mappings for csrgraph (integer IDs)
        self.node_str_to_id = {}
        self.node_id_to_str = {}
        self.node_id_to_metadata = {}
        self.next_node_id = 0

        # Regular expressions for extracting links
        self.md_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self.html_link_pattern = re.compile(
            r'<a\s+(?:[^>]*?)href="([^"]*)"[^>]*>(.*?)</a>', re.IGNORECASE
        )

    def _get_or_create_node_id(self, node_str, metadata=None):
        """Get the integer ID for a node string, creating it if necessary."""
        if node_str not in self.node_str_to_id:
            node_id = self.next_node_id
            self.node_str_to_id[node_str] = node_id
            self.node_id_to_str[node_id] = node_str
            if metadata:
                self.node_id_to_metadata[node_id] = metadata
            self.next_node_id += 1
            return node_id
        else:
            # Update metadata if node exists and new metadata provided
            node_id = self.node_str_to_id[node_str]
            if metadata and node_id not in self.node_id_to_metadata:
                self.node_id_to_metadata[node_id] = metadata
            return node_id

    def load_subjects(self):
        """Load subject information directly from cache directory structure."""
        if not self.resources_dir.exists():
            print(f"Warning: Resources directory not found: {self.resources_dir}")
            return

        # Get all immediate subdirectories of the resources directory
        for subdir in self.resources_dir.iterdir():
            if subdir.is_dir():
                subject_name = subdir.name
                self.subjects[subject_name] = subject_name.lower()

        print(f"Found {len(self.subjects)} subjects: {', '.join(self.subjects.keys())}")

    def find_markdown_files(self):
        """Find all markdown files in the resources directory."""
        if not self.resources_dir.exists():
            raise FileNotFoundError(
                f"Resources directory not found: {self.resources_dir}"
            )

        # Recursively find all markdown files
        self.md_files = list(self.resources_dir.glob("**/*.md"))
        print(f"Found {len(self.md_files)} markdown files")

        # Find all subdirectories (modules)
        all_dirs = set()
        for md_file in self.md_files:
            # Add all parent directories up to resources_dir
            current = md_file.parent
            while current != self.resources_dir and current != current.parent:
                all_dirs.add(current)
                current = current.parent

        # Convert to relative paths (strings)
        self.modules = {str(d.relative_to(self.resources_dir)) for d in all_dirs}
        print(f"Found {len(self.modules)} module directories")

    def extract_links_from_md(self, md_file):
        """Extract links from a markdown file."""
        try:
            with open(md_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            links = []

            # Find markdown links: [text](link)
            md_links = self.md_link_pattern.findall(content)
            for _, link in md_links:
                links.append(link)

            # Find HTML links: <a href="link">text</a>
            html_links = self.html_link_pattern.findall(content)
            for link, _ in html_links:
                links.append(link)

            return links
        except Exception as e:
            print(f"Error reading file {md_file}: {e}")
            return []

    def is_external_link(self, link):
        """Check if a link is external (http/https)."""
        parsed = urlparse(link)
        return parsed.scheme in ("http", "https")

    def normalize_internal_link(self, link, source_file):
        """Normalize an internal link to a Path object relative to resources_dir."""
        # Remove URL fragments
        link = link.split("#")[0]

        # If empty after removing fragment, it's a self-reference
        if not link:
            return source_file

        # Safety check: Skip extremely long links
        if len(link) > 250:
            return None  # Indicate link should be skipped

        # Handle relative links
        source_dir = source_file.parent
        try:
            if link.startswith("./"):
                link = link[2:]
                target = source_dir / link
            elif link.startswith("../"):
                target_path = source_dir
                while link.startswith("../"):
                    target_path = target_path.parent
                    link = link[3:]
                target = target_path / link
            else:
                target = source_dir / link
        except Exception:
            # Handle potential path errors during resolution
            return None

        # Ensure the link has .md extension if it looks like a file without one
        if not target.suffix and not target.is_dir():
            # Check if adding .md makes it an existing file
            potential_target_md = target.with_suffix(".md")
            if potential_target_md.exists():
                target = potential_target_md
            elif (
                not target.exists()
            ):  # If neither target nor target.md exists, maybe skip?
                return None

        # Convert to relative path from resources_dir
        try:
            rel_target_path = target.relative_to(self.resources_dir)
            return str(rel_target_path)  # Return string path
        except ValueError:
            # Link points outside the resources dir, treat as invalid internal link
            return None
        except Exception:
            # Catch other potential errors during path handling
            return None

    def get_subject_for_path(self, path_str):
        """Determine the subject for a given file or directory path string.

        Directory structure:
        md_resources/[subject]/[resource]/...
        Where [subject] is the programming language or topic.
        """
        # For the external HTTP node, return 'external'
        if path_str == self.http_node_str:
            return "external"

        # Split the path into components
        path_components = path_str.replace("\\", "/").split("/")

        # The subject is always the second level directory (index 0)
        # Example: python/frameworks/flask/routing.md → subject is "python"
        if path_components and len(path_components) > 0:
            return path_components[0].lower()
        else:
            return "unknown"

    def _get_node_metadata(self, node_str):
        """Helper to determine node type and subject."""
        node_type = "unknown"

        # Check if this is a subject node
        if node_str.startswith("subject:"):
            subject_name = node_str.split(":", 1)[1]
            return {"type": "subject", "subject": subject_name}
        # External links node
        elif node_str == self.http_node_str:
            node_type = "external"
        # Module nodes
        elif node_str in self.modules:
            node_type = "module"
        # File nodes
        elif node_str.endswith(".md"):
            node_type = "file"

        subject = self.get_subject_for_path(node_str)
        return {"type": node_type, "subject": subject}

    def _process_file(self, md_file):
        """Process a single markdown file and extract its links.
        This function is designed to be used with multiprocessing.

        Args:
            md_file: Path to the markdown file to process

        Returns:
            List of (source_str, target_str) tuples representing edges
        """
        rel_source_str = str(md_file.relative_to(self.resources_dir))
        links = self.extract_links_from_md(md_file)
        edges = []

        # Create implicit connections

        # 1. Get subject from the path (first directory level)
        path_parts = rel_source_str.split("/")
        if len(path_parts) > 0:
            subject = path_parts[0]
            # Create subject node if needed (we'll use a special prefix to identify it)
            subject_node = f"subject:{subject}"
            # Add edge from file to subject
            edges.append((rel_source_str, subject_node))

        # 2. Create edges to parent modules (directories)
        current_path = Path(rel_source_str).parent
        while str(current_path) != ".":
            module_str = str(current_path)
            if module_str in self.modules:
                # Add edge from file to its parent module
                edges.append((rel_source_str, module_str))
            current_path = current_path.parent

        # Process explicit links from the file
        for link in links:
            if self.is_external_link(link):
                # External link
                edges.append((rel_source_str, self.http_node_str))
            else:
                # Process internal link
                target_str = self.normalize_internal_link(link, md_file)

                # Skip self-references or invalid links
                if target_str is None or target_str == rel_source_str:
                    continue

                # Check if target file exists (relative path string)
                target_path = self.resources_dir / target_str
                if target_path.exists():
                    edges.append((rel_source_str, target_str))
                # else:
                # print(f"Skipping edge to non-existent target: {target_str}")

        return rel_source_str, edges

    def build_graph(self, workers=None, graph_cache="graph_cache.npz", backend=None):
        """Build the graph from markdown files and their links by constructing
        a scipy sparse matrix.

        If a cached graph file exists, it will be loaded instead of rebuilding
        the graph.

        Args:
            workers: Number of worker processes for extraction
                (None = use all available cores)
            graph_cache: Path to save/load the graph cache
            backend: Override the graph backend ('csr' or 'networkx')
        """
        # Set backend if provided
        if backend:
            self.backend = backend.lower()

        # Check if cached graph exists
        if os.path.exists(graph_cache):
            print(f"Loading graph from cache: {graph_cache}")
            try:
                # Load the cached graph
                cached_data = np.load(graph_cache, allow_pickle=True)

                # Extract components
                sparse_mat_data = cached_data["mat_data"]
                sparse_mat_indices = cached_data["mat_indices"]
                sparse_mat_indptr = cached_data["mat_indptr"]
                sparse_mat_shape = tuple(cached_data["mat_shape"])
                node_names = cached_data["node_names"]

                # Reconstruct sparse matrix
                sparse_mat = sparse.csr_matrix(
                    (sparse_mat_data, sparse_mat_indices, sparse_mat_indptr),
                    shape=sparse_mat_shape,
                )

                # Create CSRGraph from sparse matrix
                self.graph = cg.csrgraph(sparse_mat, nodenames=node_names)

                # If using NetworkX backend, also create a NetworkX graph
                if self.backend == "networkx":
                    self.networkx_graph = nx.DiGraph()
                    # Add nodes with metadata
                    for i, node_name in enumerate(node_names):
                        node_metadata = {}
                        if (
                            "node_types" in cached_data
                            and "node_subjects" in cached_data
                        ):
                            node_metadata = {
                                "type": cached_data["node_types"][i],
                                "subject": cached_data["node_subjects"][i],
                            }
                        self.networkx_graph.add_node(i, name=node_name, **node_metadata)

                    # Add edges from sparse matrix
                    for i in range(sparse_mat.shape[0]):
                        for j in sparse_mat.indices[
                            sparse_mat.indptr[i] : sparse_mat.indptr[i + 1]
                        ]:
                            weight = sparse_mat.data[
                                sparse_mat.indptr[i] : sparse_mat.indptr[i + 1]
                            ][
                                sparse_mat.indices[
                                    sparse_mat.indptr[i] : sparse_mat.indptr[i + 1]
                                ]
                                == j
                            ][0]
                            self.networkx_graph.add_edge(i, j, weight=weight)

                # Reconstruct metadata
                self.node_id_to_str = dict(enumerate(node_names))
                self.node_str_to_id = {name: idx for idx, name in enumerate(node_names)}

                # Load node metadata if available
                if "node_types" in cached_data and "node_subjects" in cached_data:
                    node_types = cached_data["node_types"]
                    node_subjects = cached_data["node_subjects"]

                    self.node_id_to_metadata = {}
                    for node_id, node_type, node_subject in zip(
                        range(len(node_names)), node_types, node_subjects, strict=False
                    ):
                        self.node_id_to_metadata[node_id] = {
                            "type": node_type,
                            "subject": node_subject,
                        }
                else:
                    # If metadata not available, recreate it
                    print("Metadata not found in cache, recreating...")
                    self.node_id_to_metadata = {}
                    for node_id, node_str in self.node_id_to_str.items():
                        self.node_id_to_metadata[node_id] = self._get_node_metadata(
                            node_str
                        )

                print(
                    "Successfully loaded graph with",
                    f"{self.graph.nnodes} nodes from cache",
                )
                return
            except Exception as e:
                print(f"Error loading cached graph: {e}")
                print("Building graph from scratch instead...")

        # If we get here, we're building the graph from scratch
        self.load_subjects()
        self.find_markdown_files()

        # Use all available cores if workers not specified
        if workers is None:
            workers = CPU_COUNT

        print(f"Building graph using {workers} CPU cores...")

        # --- Collect string edges using sequential processing ---
        all_edges_str = []  # List of (source_str, target_str)
        processed_files = set()

        # --- Sequential Processing ---
        print(f"Processing {len(self.md_files)} files sequentially...")
        for md_file in tqdm(self.md_files, desc="Extracting links (sequential)"):
            try:
                rel_source_str, edges = self._process_file(md_file)
                processed_files.add(rel_source_str)
                all_edges_str.extend(edges)
            except Exception as e:
                print(f"Error processing {md_file}: {e}")

        # --- Create graph ---
        print(
            f"Collected {len(all_edges_str)} edges from {len(processed_files)} files."
        )
        if not all_edges_str:
            print("Warning: No edges found. Creating an empty graph.")
            self.graph = None
            self.networkx_graph = None
            print("Graph building complete: 0 nodes, 0 edges")
            return

        print("Creating node mapping...")
        # Create a mapping of node names to integers
        all_nodes = set()
        for src, dst in all_edges_str:
            all_nodes.add(src)
            all_nodes.add(dst)

        # Sort for deterministic ordering
        node_names = sorted(list(all_nodes))
        node_to_id = {node: i for i, node in enumerate(node_names)}

        print(f"Found {len(node_names)} unique nodes")

        # Create edge list as (row, col, data) format for sparse matrix
        print("Creating sparse matrix...")
        rows, cols, data = [], [], []
        for src, dst in all_edges_str:
            rows.append(node_to_id[src])
            cols.append(node_to_id[dst])
            data.append(1.0)  # Default weight

        # Create sparse CSR matrix
        nnodes = len(node_names)
        sparse_mat = sparse.csr_matrix((data, (rows, cols)), shape=(nnodes, nnodes))

        # Store node metadata
        print("Creating node metadata...")
        self.node_id_to_str = dict(enumerate(node_names))
        self.node_str_to_id = {name: idx for idx, name in enumerate(node_names)}
        self.node_id_to_metadata = {}

        for node_id, node_str in self.node_id_to_str.items():
            self.node_id_to_metadata[node_id] = self._get_node_metadata(node_str)

        # Save the graph to cache - always save in CSR format for compatibility
        print(f"Saving graph to cache: {graph_cache}")
        self._save_graph_cache(graph_cache, sparse_mat, node_names)

        # Create the graph based on the selected backend
        if self.backend == "networkx":
            print("Creating NetworkX graph...")
            try:
                # Create a NetworkX directed graph
                self.networkx_graph = nx.DiGraph()

                # Add nodes with metadata
                for node_id, node_str in self.node_id_to_str.items():
                    metadata = self.node_id_to_metadata.get(node_id, {})
                    self.networkx_graph.add_node(node_id, name=node_str, **metadata)

                # Add edges
                for src, dst in zip(rows, cols, strict=False):
                    self.networkx_graph.add_edge(src, dst, weight=1.0)

                print(
                    "NetworkX graph created with",
                    f"{self.networkx_graph.number_of_nodes()} nodes",
                    f"and {self.networkx_graph.number_of_edges()} edges",
                )

                # Always create CSR graph as well for compatibility with other code
                print("Creating CSRGraph from sparse matrix for compatibility...")
                self.graph = cg.csrgraph(sparse_mat, nodenames=node_names)

            except Exception as e:
                print(f"Error creating NetworkX graph: {e}")
                import traceback

                traceback.print_exc()

                # Fall back to CSR graph
                print("Falling back to CSR graph...")
                self.backend = "csr"

        # Create CSR graph if needed
        if self.backend == "csr" or self.graph is None:
            print("Creating CSRGraph from sparse matrix...")
            try:
                # Create CSRGraph directly from the sparse matrix
                self.graph = cg.csrgraph(sparse_mat, nodenames=node_names)
                print(f"CSR graph building complete: {self.graph.nnodes} nodes")
            except Exception as e:
                print(f"Error creating CSRGraph: {e}")
                import traceback

                traceback.print_exc()
                self.graph = None  # Indicate failure

    def _save_graph_cache(self, cache_path, sparse_mat, node_names):
        """Save the graph to a cache file for faster loading in the future."""
        try:
            # Extract components from sparse matrix
            mat_data = sparse_mat.data
            mat_indices = sparse_mat.indices
            mat_indptr = sparse_mat.indptr
            mat_shape = np.array(sparse_mat.shape)

            # Extract node metadata for saving
            node_types = [m.get("type", "") for m in self.node_id_to_metadata.values()]
            node_subjects = [
                m.get("subject", "") for m in self.node_id_to_metadata.values()
            ]

            # Save to npz file
            np.savez_compressed(
                cache_path,
                mat_data=mat_data,
                mat_indices=mat_indices,
                mat_indptr=mat_indptr,
                mat_shape=mat_shape,
                node_names=np.array(node_names, dtype=object),
                node_types=np.array(node_types, dtype=object),
                node_subjects=np.array(node_subjects, dtype=object),
            )
            print(f"Graph successfully cached to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save graph to cache: {e}")
            import traceback

            traceback.print_exc()

    def calculate_graph_stats(self):
        """Calculate and return statistics about the graph."""
        if self.graph is None:
            return {}

        stats = {
            "num_nodes": self.graph.nnodes,
            # CSRGraph doesn't have a direct nedges attribute
        }

        # Count node types and subjects from metadata
        type_counts = defaultdict(int)
        subject_counts = defaultdict(int)
        for _, metadata in self.node_id_to_metadata.items():
            type_counts[metadata.get("type", "unknown")] += 1
            subject_counts[metadata.get("subject", "unknown")] += 1

        stats["node_types"] = dict(type_counts)
        stats["subjects"] = dict(subject_counts)
        stats["num_files"] = type_counts.get("file", 0)
        stats["num_modules"] = type_counts.get("module", 0)

        return stats

    def save_graph_info(self, output_file="graph_info.npz"):
        """Save the graph metadata in NumPy format."""
        if self.graph is None:
            print("Error: Graph not built.")
            return

        try:
            # Convert node_id_to_metadata to separate lists for saving
            node_ids = list(self.node_id_to_metadata.keys())
            node_types = [m.get("type", "") for m in self.node_id_to_metadata.values()]
            node_subjects = [
                m.get("subject", "") for m in self.node_id_to_metadata.values()
            ]
            node_strs = [self.node_id_to_str.get(nid, "") for nid in node_ids]

            # Save the metadata
            np.savez_compressed(
                output_file,
                num_nodes=self.graph.nnodes,
                node_ids=np.array(node_ids),
                node_strs=np.array(node_strs),
                node_types=np.array(node_types),
                node_subjects=np.array(node_subjects),
            )
            print(f"Graph metadata saved to {output_file}")
        except Exception as e:
            print(f"Warning: Could not save graph metadata due to: {e}")
            import traceback

            traceback.print_exc()

    def print_stats(self, stats=None):
        """Print statistics about the graph."""
        if stats is None:
            stats = self.calculate_graph_stats()

        print("\n=== Graph Statistics ===")
        print(f"Nodes: {stats.get('num_nodes', 'N/A')}")

        print("\nNode Types:")
        for ntype, count in sorted(stats.get("node_types", {}).items()):
            print(f"  - {ntype.capitalize()}: {count} nodes")

        print("\nSubjects:")
        for subject, count in sorted(
            stats.get("subjects", {}).items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  - {subject.capitalize()}: {count} nodes")

    def generate_embeddings(
        self, method="ggvec", dimensions=128, workers=None, **kwargs
    ):
        """Generate embeddings for the graph using one of the nodevectors methods.

        Args:
            method: Which embedding method to use. Options:
                   'node2vec' - Node2Vec algorithm
                   'ggvec' - GGVec algorithm (fast, good for visualization)
                   'prone' - ProNE algorithm (fastest)
                   'grarep' - GraRep algorithm
                   'glove' - GloVe algorithm
            dimensions: Dimensionality of the embeddings
            workers: Number of parallel workers (None = use all available)
            **kwargs: Additional arguments specific to the embedding method

        Returns:
            Dictionary of node_id: embedding vector pairs
        """
        if self.graph is None:
            print("Error: Graph not built yet.")
            return None

        # Use all available cores if workers not specified
        if workers is None:
            workers = CPU_COUNT

        print(f"Generating graph embeddings using {method} with {workers} CPU cores...")
        print(f"Embedding parameters: dims={dimensions}")

        try:
            # Select embedding method
            if method.lower() == "node2vec":
                # Node2Vec parameters
                walklen = kwargs.get("walklen", 10)
                epochs = kwargs.get("epochs", 20)
                return_weight = kwargs.get("return_weight", 1.0)
                neighbor_weight = kwargs.get("neighbor_weight", 1.0)

                print(
                    f"""Node2Vec parameters: walklen={walklen}, epochs={epochs},
                    return_weight={return_weight}, neighbor_weight={neighbor_weight}"""
                )

                model = Node2Vec(
                    n_components=dimensions,
                    walklen=walklen,
                    epochs=epochs,
                    return_weight=return_weight,
                    neighbor_weight=neighbor_weight,
                    threads=workers,
                    keep_walks=False,
                    verbose=True,
                )

            elif method.lower() == "ggvec":
                # GGVec parameters
                order = kwargs.get("order", 1)
                learning_rate = kwargs.get("learning_rate", 0.1)
                negative_ratio = kwargs.get("negative_ratio", 0.1)
                tol = kwargs.get("tol", 0.1)

                print(
                    f"""GGVec parameters: order={order}, lr={learning_rate},
                    negative_ratio={negative_ratio}, tol={tol}"""
                )

                model = GGVec(
                    n_components=dimensions,
                    order=order,
                    learning_rate=learning_rate,
                    negative_ratio=negative_ratio,
                    tol=tol,
                    threads=workers,
                    verbose=True,
                )

            elif method.lower() == "prone":
                # ProNE parameters
                step = kwargs.get("step", 5)
                mu = kwargs.get("mu", 0.1)
                theta = kwargs.get("theta", 0.5)

                print(f"ProNE parameters: step={step}, mu={mu}, theta={theta}")

                model = ProNE(
                    n_components=dimensions, step=step, mu=mu, theta=theta, verbose=True
                )

            elif method.lower() == "grarep":
                # GraRep parameters
                order = kwargs.get("order", 1)
                # Create a specific embedder for GraRep
                from sklearn.decomposition import TruncatedSVD

                embedder = TruncatedSVD(n_components=dimensions)

                print(f"GraRep parameters: order={order}")

                model = GraRep(
                    n_components=dimensions,
                    order=order,
                    embedder=embedder,  # Explicitly pass the embedder with n_components
                    verbose=True,
                )

            elif method.lower() == "glove":
                # GloVe parameters
                learning_rate = kwargs.get("learning_rate", 0.05)
                tol = kwargs.get("tol", 0.0001)
                max_epoch = kwargs.get("max_epoch", 100)
                max_count = kwargs.get("max_count", 100)

                print(
                    f"""GloVe parameters: lr={learning_rate}, tol={tol},
                    max_epoch={max_epoch}, max_count={max_count}"""
                )

                model = Glove(
                    n_components=dimensions,
                    learning_rate=learning_rate,
                    tol=tol,
                    max_epoch=max_epoch,
                    max_count=max_count,
                    verbose=True,
                )

            elif method.lower() == "umap":
                # Use UMAP with SKLearnEmbedder
                n_neighbors = kwargs.get("n_neighbors", 15)
                min_dist = kwargs.get("min_dist", 0.1)

                print(
                    f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}"
                )

                # Create the UMAP model first
                import umap as umap_module

                # The issue is here - the SKLearnEmbedder tries to call the UMAP object
                # Instead, pass the model class and its parameters separately
                model = SKLearnEmbedder(
                    embedder=umap_module.UMAP,  # Pass the class, not an instance
                    n_components=dimensions,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42,
                    verbose=True,
                )

            else:
                raise ValueError(
                    f"""Unknown embedding method: {method}.
                    Choose from 'node2vec', 'ggvec', 'prone', 'grarep',
                    'glove', or 'umap'."""
                )

            # Fit the model
            embeddings_matrix = model.fit_transform(self.graph)

            # Convert matrix to a dictionary {node_id: embedding}
            embeddings = {
                node_id: embeddings_matrix[node_id]
                for node_id in range(self.graph.nnodes)
            }

            print(f"Generated embeddings for {len(embeddings)} nodes using {method}")
            return embeddings

        except Exception as e:
            print(f"Error generating graph embeddings with {method}: {e}")
            import traceback

            traceback.print_exc()  # Print detailed traceback
            return None

    def get_edges(self):
        """Get all edges in the graph.

        Returns:
            List of (source_id, target_id) tuples representing edges
        """
        if self.graph is None and self.networkx_graph is None:
            print("No graph available. Call build_graph() first.")
            return []

        # Prefer NetworkX if available
        if self.backend == "networkx" and self.networkx_graph is not None:
            return list(self.networkx_graph.edges())

        # Fall back to CSR graph
        if self.graph is not None:
            edges = []

            # Try different methods to extract edges based on available attributes
            if hasattr(self.graph, "mat"):
                # Get edges from the matrix
                csr_matrix = self.graph.mat
                source_indices, target_indices = csr_matrix.nonzero()

                for src, dst in zip(source_indices, target_indices, strict=False):
                    edges.append((src, dst))

            elif hasattr(self.graph, "src") and hasattr(self.graph, "dst"):
                # Get edges from src/dst arrays
                for src, dst in zip(self.graph.src, self.graph.dst, strict=False):
                    edges.append((src, dst))

            return edges

        return []

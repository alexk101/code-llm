import argparse
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import the MarkdownGraph class
from utils.graphs import MarkdownGraph

# Determine the optimal number of CPU cores for processing
CPU_COUNT = multiprocessing.cpu_count()


class MarkdownGraphVisualizer:
    """
    Class to create and visualize a graph
    of markdown files and their links using node2vec.
    """

    def __init__(self, resources_dir="cache/md_resources"):
        """Initialize the visualizer with the path to the markdown resources."""
        self.resources_dir = Path(resources_dir)
        self.graph_builder = MarkdownGraph(resources_dir)
        self.output_base_dir = Path("plots/graph_emb")

    def get_node_colors(self):
        """Generate node colors based on subject using stored metadata."""
        # Use seaborn's tab10 color palette
        default_colors = sns.color_palette("tab10").as_hex()

        # Special colors
        special_colors = {
            "external": "#FF5733",  # Red for external
            "unknown": "#CCCCCC",  # Gray for unknown
        }

        # Extract all unique subjects from the graph metadata
        all_subjects = set()
        for metadata in self.graph_builder.node_id_to_metadata.values():
            subject = metadata.get("subject", "unknown")
            if subject not in ["external", "unknown"]:
                all_subjects.add(subject)

        # Create a color map for all subjects
        subject_colors = dict(special_colors)  # Start with special colors

        # Assign colors to subjects in sorted order for consistency
        color_idx = 0
        for subject in sorted(all_subjects):
            if subject not in subject_colors:
                subject_colors[subject] = default_colors[
                    color_idx % len(default_colors)
                ]
                color_idx += 1

        # Generate colors for each node ID based on their subject
        node_id_colors = {}
        for node_id, metadata in self.graph_builder.node_id_to_metadata.items():
            subject = metadata.get("subject", "unknown")
            node_id_colors[node_id] = subject_colors.get(
                subject, subject_colors["unknown"]
            )

        return node_id_colors, subject_colors  # Returns map from ID to color

    def reduce_dimensions(
        self, embedding_matrix, method="umap", n_components=2, random_state=42, **kwargs
    ):
        """Reduce high-dimensional embeddings to 2D for visualization.

        Args:
            embedding_matrix: Matrix of node embeddings
            method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
            n_components: Number of components in the reduced space (typically 2)
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters specific to each method

        Returns:
            array: Reduced embeddings in 2D space
        """
        print(f"Reducing dimensions with {method.upper()}...")

        try:
            if method.lower() == "umap":
                # UMAP parameters
                n_neighbors = kwargs.get("n_neighbors", 15)
                min_dist = kwargs.get("min_dist", 0.1)

                print(
                    f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}"
                )

                import umap

                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                    verbose=kwargs.get("verbose", True),
                )

            elif method.lower() == "tsne":
                # t-SNE parameters
                perplexity = kwargs.get("perplexity", 30.0)
                early_exaggeration = kwargs.get("early_exaggeration", 12.0)
                learning_rate = kwargs.get("learning_rate", 200.0)

                print(
                    f"""t-SNE parameters: perplexity={perplexity},
                    early_exaggeration={early_exaggeration},
                    learning_rate={learning_rate}"""
                )

                from sklearn.manifold import TSNE

                reducer = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    verbose=kwargs.get("verbose", True),
                )

            elif method.lower() == "pca":
                # PCA parameters
                print(f"Using PCA with {n_components} components")

                from sklearn.decomposition import PCA

                reducer = PCA(n_components=n_components, random_state=random_state)

            else:
                raise ValueError(
                    f"""Unknown dimensionality reduction method: {method}.
                    Choose from 'umap', 'tsne', or 'pca'."""
                )

            # Apply the dimensionality reduction
            reduced_embeddings = reducer.fit_transform(embedding_matrix)
            print(f"{method.upper()} reduction complete")
            return reduced_embeddings

        except Exception as e:
            print(f"Error reducing dimensions with {method}: {e}")
            import traceback

            traceback.print_exc()

            # If the primary method fails, fall back to PCA as it's the most reliable
            if method.lower() != "pca":
                print("Falling back to PCA...")
                try:
                    from sklearn.decomposition import PCA

                    reducer = PCA(n_components=n_components)
                    return reducer.fit_transform(embedding_matrix)
                except Exception as e:
                    print(f"PCA fallback also failed: {e}")

            return None

    def visualize_node2vec(
        self,
        embeddings=None,
        output_file="node2vec_viz.png",
        figsize=(24, 20),
        random_seed=42,
        reduction_method="umap",
        title=None,
        **kwargs,
    ):
        """Visualize the graph using node embeddings.

        Args:
            embeddings: Pre-computed node embeddings (node_id: vector dict)
            output_file: Path to save the visualization
            figsize: Size of the figure
            random_seed: Random seed for reproducibility
            reduction_method: Method for reduction ('umap', 'tsne', or 'pca')
            title: Optional title override for the plot
            **kwargs: Additional parameters for the dimensionality reduction method
        """
        if self.graph_builder.graph is None:
            print("Error: Graph not built.")
            return

        # Generate embeddings if not provided
        if embeddings is None:
            print("Embeddings not provided, attempting to generate...")
            embeddings = self.graph_builder.generate_embeddings()  # Use defaults
            if embeddings is None:
                print("Failed to generate embeddings. Aborting visualization.")
                return

        # Use all available nodes from embeddings
        nodes_to_visualize_ids = list(embeddings.keys())

        print(
            f"Visualizing all {len(nodes_to_visualize_ids)} nodes with graph embeddings"
        )

        # Create embedding matrix only for selected nodes
        embedding_matrix = np.array(
            [embeddings[node_id] for node_id in nodes_to_visualize_ids]
        )

        # Reduce dimensionality to 2D for visualization
        embedding_2d = self.reduce_dimensions(
            embedding_matrix=embedding_matrix,
            method=reduction_method,
            n_components=2,
            random_state=random_seed,
            **kwargs,
        )

        if embedding_2d is None:
            print("Dimensionality reduction failed. Cannot create visualization.")
            return

        # Get node colors and types for visualization
        node_colors_viz = []
        node_sizes_viz = []
        node_types_viz = []
        node_subjects_viz = []

        colors_dict_id, subject_colors = self.get_node_colors()

        # Prepare node attributes for the nodes being visualized
        print("Preparing node attributes for visualization...")
        for node_id in nodes_to_visualize_ids:
            metadata = self.graph_builder.node_id_to_metadata.get(node_id, {})
            subject = metadata.get("subject", "unknown")
            node_type = metadata.get("type", "unknown")

            # Get node color based on subject
            node_colors_viz.append(colors_dict_id.get(node_id, "#CCCCCC"))

            # Node size based on type
            if node_type == "subject":
                node_sizes_viz.append(500)  # Make subject nodes much larger
            elif node_type == "module":
                node_sizes_viz.append(100)
            elif node_type == "external":
                node_sizes_viz.append(200)
            elif node_type == "file":
                node_sizes_viz.append(30)
            else:  # Default size
                node_sizes_viz.append(20)

            node_types_viz.append(node_type)
            node_subjects_viz.append(subject)

        # Create plot
        print("Creating plot...")
        plt.figure(figsize=figsize)

        # Draw nodes by type
        unique_node_types = sorted(list(set(node_types_viz)))
        for nt in unique_node_types:
            # Get indices of nodes with this type
            indices = [i for i, t in enumerate(node_types_viz) if t == nt]
            if not indices:
                continue

            # Get coordinates, sizes and colors for these nodes
            xs = embedding_2d[indices, 0]
            ys = embedding_2d[indices, 1]
            sizes = [node_sizes_viz[i] for i in indices]
            colors = [node_colors_viz[i] for i in indices]

            plt.scatter(
                xs,
                ys,
                s=sizes,
                c=colors,
                alpha=0.6,
                label=nt,
                edgecolors="w",
                linewidths=0.2,
            )

        # Add legend for node types
        type_legend = plt.legend(title="Node Types", loc="upper right")
        plt.gca().add_artist(type_legend)

        # Add color legend for subjects
        # Count occurrences of each subject to sort by prevalence
        subject_counts = {}
        for subject in node_subjects_viz:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        # Sort all subjects by count
        all_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)

        # Create color legend handles for all subjects
        subject_handles = []
        subject_labels = []

        for subject, count in all_subjects:
            if subject in subject_colors:
                proxy = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=subject_colors[subject],
                    markersize=10,
                    label=f"{subject.capitalize()} ({count})",
                )
                subject_handles.append(proxy)
                subject_labels.append(f"{subject.capitalize()} ({count})")

        # Add the color legend at a different position
        if subject_handles:
            # Adjust the number of columns based on the number of subjects
            ncol = min(4, max(1, len(subject_handles) // 8))

            # Create a custom legend
            subject_legend = plt.legend(
                handles=subject_handles,
                labels=subject_labels,
                title="Subject Colors",
                loc="lower right",
                bbox_to_anchor=(1, 0),
                ncol=ncol,
                fontsize=9,
            )

            plt.gca().add_artist(subject_legend)

        # Add title
        if title:
            plot_title = title
        else:
            plot_title = f"""Graph Embedding Visualization (UMAP projection)
            {len(nodes_to_visualize_ids)} nodes"""

        plt.title(plot_title)
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.xticks([])
        plt.yticks([])

        # Save figure
        print(f"Saving figure to {output_file}...")
        plt.tight_layout()
        # Ensure plots directory exists
        os.makedirs(Path(output_file).parent, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Graph visualization saved to {output_file}")
        plt.close()

    def run(
        self,
        save=True,
        embedding_methods=None,
        output_dir="plots",
        output_data="graph_info.npz",
        dimensions=64,
        workers=None,
        min_degree=1,
        graph_cache="graph_cache.npz",
        reduction_method="umap",
        reduction_params=None,
    ):
        """Run the full pipeline with graph visualization for multiple embeddings.

        Args:
            save: Whether to save the graph data
            embedding_methods: List of embedding methods to use,
                or None to use the default 'ggvec'. Options:
                'node2vec', 'ggvec', 'prone', 'grarep', 'glove', 'umap'
            output_dir: Directory to save visualization images
            output_data: Path to save graph data
            dimensions: Dimensionality of embeddings
            workers: Number of worker processes (None = use all available)
            min_degree: Minimum node degree for filtering (currently unused)
            graph_cache: Path to save/load the graph cache
            reduction_method: Method for reduction ('umap', 'tsne', or 'pca')
            reduction_params: Dictionary of parameters for the reduction method
        """
        # Initialize reduction parameters if None
        if reduction_params is None:
            reduction_params = {}

        # Build the graph with parallelization and caching
        self.graph_builder.build_graph(workers=workers, graph_cache=graph_cache)

        stats = self.graph_builder.calculate_graph_stats()
        self.graph_builder.print_stats(stats)

        # Default to using GGVec if no methods specified
        if embedding_methods is None:
            # GGVec is faster and more reliable than other methods
            embedding_methods = ["ggvec"]

        # Create the main output directory structure
        base_output_dir = Path(output_dir) / "graph_emb"
        if isinstance(reduction_method, str):
            reduction_methods = [reduction_method]
        else:
            reduction_methods = reduction_method

        # Create directories for each reduction method
        for method in reduction_methods:
            method_dir = base_output_dir / method
            os.makedirs(method_dir, exist_ok=True)
            print(f"Created output directory for {method}: {method_dir}")

        # Keep track of failed methods to try a fallback
        all_methods_failed = True

        # Generate embeddings and visualizations for each method
        for emb_method in embedding_methods:
            print(f"\n===== Generating {emb_method.upper()} embeddings =====")

            # Generate embeddings using the specified method
            embeddings = self.graph_builder.generate_embeddings(
                method=emb_method, dimensions=dimensions, workers=workers
            )

            if embeddings:
                all_methods_failed = False

                # Create visualizations with each reduction method
                for red_method in reduction_methods:
                    print(f"\n----- Visualizing with {red_method.upper()} -----")

                    # Create appropriate output directory
                    method_dir = base_output_dir / red_method

                    # Set appropriate parameters for this reduction method
                    current_params = reduction_params.copy()

                    # Create output filename based on embedding method only
                    output_file = method_dir / f"{emb_method}_viz.png"

                    # Visualize the embeddings
                    self.visualize_node2vec(
                        embeddings=embeddings,
                        output_file=output_file,
                        reduction_method=red_method,
                        title=f"""{emb_method.upper()} Embedding with
                        {red_method.upper()}
                        {self.graph_builder.graph.nnodes} nodes""",
                        **current_params,
                    )
            else:
                print(
                    f"""Skipping visualization for {emb_method} due to
                    embedding generation failure."""
                )

        # If all methods failed, try to fall back to a simpler method
        if all_methods_failed and embedding_methods != ["ggvec"]:
            print("\n===== All specified methods failed, falling back to GGVec =====")
            embeddings = self.graph_builder.generate_embeddings(
                method="ggvec", dimensions=dimensions, workers=workers
            )

            if embeddings:
                for red_method in reduction_methods:
                    method_dir = base_output_dir / red_method
                    output_file = method_dir / "ggvec_fallback_viz.png"

                    self.visualize_node2vec(
                        embeddings=embeddings,
                        output_file=output_file,
                        reduction_method=red_method,
                        title=f"""GGVec Fallback with {red_method.upper()}
                        {self.graph_builder.graph.nnodes} nodes""",
                        **reduction_params,
                    )

        if save and self.graph_builder.graph:
            # Save graph data to the base directory
            output_data_path = base_output_dir / output_data
            print(f"Saving graph data to {output_data_path}")
            self.graph_builder.save_graph_info(output_file=output_data_path)
        elif not self.graph_builder.graph:
            print("Skipping saving graph data as graph building failed.")


def main():
    """Main function to parse arguments and run the graph builder."""
    parser = argparse.ArgumentParser(
        description="Visualize markdown resources with graph embeddings"
    )
    parser.add_argument(
        "--resources",
        default="cache/md_resources",
        help="Path to markdown resources",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--output-data",
        default="graph_info.npz",
        help="Output graph data file (.npz format)",
    )
    parser.add_argument(
        "--graph-cache",
        default="graph_cache.npz",
        help="Path to save/load the graph cache",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving graph data")
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        choices=["node2vec", "ggvec", "prone", "grarep", "glove", "umap", "all"],
        default=["ggvec"],
        help='Embedding methods to use (specify multiple or "all" for all methods)',
    )
    parser.add_argument(
        "--dimensions", type=int, default=64, help="Dimensions for graph embeddings"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--reduction-methods",
        nargs="+",
        choices=["umap", "tsne", "pca", "all"],
        default=["umap"],
        help='Methods for reduction (specify multiple or "all" for all methods)',
    )

    # UMAP specific parameters
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="UMAP parameter: n_neighbors"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="UMAP parameter: min_dist"
    )

    # t-SNE specific parameters
    parser.add_argument(
        "--perplexity", type=float, default=30.0, help="t-SNE parameter: perplexity"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE parameter: learning_rate",
    )

    args = parser.parse_args()

    # Handle 'all' option for embedding methods
    if "all" in args.embedding_methods:
        embedding_methods = ["node2vec", "ggvec", "prone", "grarep", "glove"]
    else:
        embedding_methods = args.embedding_methods

    # Handle 'all' option for reduction methods
    if "all" in args.reduction_methods:
        reduction_methods = ["umap", "tsne", "pca"]
    else:
        reduction_methods = args.reduction_methods

    # Collect reduction parameters based on the selected methods
    reduction_params = {}

    # If UMAP is used, add its parameters
    if "umap" in reduction_methods:
        reduction_params["n_neighbors"] = args.n_neighbors
        reduction_params["min_dist"] = args.min_dist

    # If t-SNE is used, add its parameters
    if "tsne" in reduction_methods:
        reduction_params["perplexity"] = args.perplexity
        reduction_params["learning_rate"] = args.learning_rate

    visualizer = MarkdownGraphVisualizer(resources_dir=args.resources)

    # Run the visualization with multiple embedding methods and reduction methods
    visualizer.run(
        save=not args.no_save,
        embedding_methods=embedding_methods,
        output_dir=args.output_dir,
        output_data=args.output_data,
        dimensions=args.dimensions,
        workers=args.workers,
        graph_cache=args.graph_cache,
        reduction_method=reduction_methods,
        reduction_params=reduction_params,
    )


if __name__ == "__main__":
    main()

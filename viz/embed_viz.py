import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # Use non-interactive backend
import hashlib
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from utils.data import DATASET, get_dataset


class CodeEmbeddingVisualizer:
    """Class to handle embedding and visualization of code datasets."""

    def __init__(self, cache_dir="cache/embeddings", plot_dir="plots/embeddings"):
        """
        Initialize the visualizer.

        Args:
            cache_dir: Directory to store cache files
            plot_dir: Directory to store plots
        """
        # Set style for better visualizations
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 10)

        self.cache_dir = Path(cache_dir)
        self.plot_dir = Path(plot_dir)
        self.df = None
        self.embeddings = None
        self.reduced_embeddings_3d = None
        self.reduced_embeddings_2d = None
        self.data_hash = None

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self):
        """Load the dataset and compute a hash for caching."""
        print(f"Loading dataset from {DATASET}")
        self.df = get_dataset()

        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of unique languages: {self.df['language_name'].n_unique()}")
        print(f"Number of unique tasks: {self.df['task_name'].n_unique()}")

        # Calculate a hash of the dataset to use for caching
        self.data_hash = hashlib.md5(
            str(self.df.shape).encode() + str(self.df.head(10).to_dict()).encode()
        ).hexdigest()

        return self.df

    def generate_embeddings(self):
        """Generate or load cached embeddings for the code samples."""
        embeddings_cache_file = (
            self.cache_dir / f"embeddings_cache_{self.data_hash}.npy"
        )

        # Check if embeddings cache exists
        if embeddings_cache_file.exists():
            print(f"Loading embeddings from cache file: {embeddings_cache_file}")
            self.embeddings = np.load(embeddings_cache_file)
        else:
            # Load CodeBERT model and tokenizer
            print("Cache not found. Loading CodeBERT model...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base")
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            model = model.to(device)
            model.compile()
            model.eval()

            # Generate embeddings
            print("Generating embeddings...")
            embeddings = []

            # Process each code sample
            for code in tqdm(self.df.select("code").fill_null("").to_numpy().flatten()):
                embeddings.append(self._get_embedding(code, model, tokenizer, device))

            self.embeddings = np.array(embeddings)

            # Save embeddings to cache file
            print(f"Saving embeddings to cache file: {embeddings_cache_file}")
            np.save(embeddings_cache_file, self.embeddings)

        print(f"Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def _get_embedding(self, code_text, model, tokenizer, device):
        """
        Generate an embedding for a code sample.

        Args:
            code_text: The code text to embed
            model: The CodeBERT model
            tokenizer: The CodeBERT tokenizer
            device: The device to run inference on

        Returns:
            numpy array: The embedding vector
        """
        # Truncate and clean the code
        code_text = str(code_text)[:1024]  # CodeBERT has token limits

        # Tokenize and get model output
        with torch.no_grad():
            inputs = tokenizer(
                code_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            # Use CLS token embedding as the representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        return embedding

    def reduce_dimensions(
        self,
        n_components: int = 2,
        method: Literal["umap", "pca"] = "umap",
        supervised: bool = False,
        target_column: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply dimensionality reduction to embeddings.

        Args:
            n_components: Number of dimensions to reduce to (2 or 3)
            method: Dimensionality reduction method ('umap' or 'pca')
            supervised: Whether to use supervised reduction (only for UMAP)
            target_column: Column to use for supervised reduction
            **kwargs: Additional arguments to pass to the reducer

        Returns:
            numpy array: Reduced embeddings
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")

        kind = f"{method}_{'supervised' if supervised else 'unsupervised'}"
        cache_file = self.cache_dir / f"{kind}_{n_components}d_{self.data_hash}.npy"

        if cache_file.exists():
            print(f"Loading {method.upper()} results from cache: {cache_file}")
            reduced_embeddings = np.load(cache_file)
        else:
            print(f"Computing {method.upper()} dimensionality reduction...")

            if method == "umap":
                if supervised and target_column:
                    # Get target labels and encode them if they're strings
                    target_labels = self.df[target_column].to_numpy()
                    if target_labels.dtype == object:  # If labels are strings
                        encoder = LabelEncoder()
                        target_labels = encoder.fit_transform(target_labels)

                    reducer = UMAP(
                        n_components=n_components,
                        n_neighbors=kwargs.get("n_neighbors", 15),
                        min_dist=kwargs.get("min_dist", 0.1),
                        metric=kwargs.get("metric", "cosine"),
                        target_metric="categorical",
                        target_weight=kwargs.get("target_weight", 0.5),
                    )
                    reduced_embeddings = reducer.fit_transform(
                        self.embeddings, y=target_labels
                    )
                else:
                    reducer = UMAP(
                        n_components=n_components,
                        n_neighbors=kwargs.get("n_neighbors", 15),
                        min_dist=kwargs.get("min_dist", 0.1),
                        metric=kwargs.get("metric", "cosine"),
                    )
                    reduced_embeddings = reducer.fit_transform(self.embeddings)
            else:  # PCA
                reducer = PCA(n_components=n_components)
                reduced_embeddings = reducer.fit_transform(self.embeddings)

            print(f"Saving {method.upper()} results to cache: {cache_file}")
            np.save(cache_file, reduced_embeddings)

        # Store the reduced embeddings
        if n_components == 3:
            self.reduced_embeddings_3d = reduced_embeddings
        else:
            self.reduced_embeddings_2d = reduced_embeddings

        return reduced_embeddings

    def get_top_languages(self, n=20):
        """Get the top n most frequent languages in the dataset."""
        language_counts = (
            self.df.group_by("language_name").agg(pl.len()).sort("len", descending=True)
        )
        return language_counts.head(n)["language_name"].to_list()

    def get_top_tasks(self, n=10):
        """Get the top n most frequent tasks in the dataset."""
        task_counts = (
            self.df.group_by("task_name").agg(pl.len()).sort("len", descending=True)
        )
        return task_counts.head(n)["task_name"].to_list()

    def visualize_3d_by_language(
        self,
        n_languages: int = 20,
        method: Literal["umap", "pca"] = "umap",
        supervised: bool = False,
        target_column: str = "language_name",
    ):
        """
        Create a 3D visualization of embeddings colored by programming language.

        Args:
            n_languages: Number of top languages to include
            method: Dimensionality reduction method ('umap' or 'pca')
            supervised: Whether to use supervised reduction
            target_column: Column to use for supervised reduction
        """
        # Get top languages
        top_languages = self.get_top_languages(n_languages)

        # Apply dimensionality reduction
        reduced_embeddings = self.reduce_dimensions(
            n_components=3,
            method=method,
            supervised=supervised,
            target_column=target_column,
        )

        # Create a DataFrame with reduced embeddings and language info
        viz_df = pl.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "z": reduced_embeddings[:, 2],
                "language": self.df["language_name"],
            }
        ).filter(pl.col("language").is_in(top_languages))

        # Create 3D scatter plot
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Create color palette
        palette = sns.color_palette("husl", n_colors=len(top_languages))

        # Plot each language
        for i, lang in enumerate(top_languages):
            lang_data = viz_df.filter(pl.col("language") == lang)

            ax.scatter(
                lang_data["x"].to_numpy(),
                lang_data["y"].to_numpy(),
                lang_data["z"].to_numpy(),
                label=lang,
                color=palette[i],
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"3D {method.upper()} Visualization by Language")

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        kind = f"3d_{method}_{'supervised' if supervised else 'unsupervised'}"
        # Save plot
        plt.tight_layout()
        plt.savefig(
            self.plot_dir / f"{kind}_language_visualization.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def visualize_2d_by_language(
        self,
        n_languages: int = 20,
        method: Literal["umap", "pca"] = "umap",
        supervised: bool = False,
        target_column: str = "language_name",
    ):
        """
        Create a 2D visualization of embeddings colored by programming language.

        Args:
            n_languages: Number of top languages to include
            method: Dimensionality reduction method ('umap' or 'pca')
            supervised: Whether to use supervised reduction
            target_column: Column to use for supervised reduction
        """
        # Get top languages
        top_languages = self.get_top_languages(n_languages)

        # Apply dimensionality reduction
        reduced_embeddings = self.reduce_dimensions(
            n_components=2,
            method=method,
            supervised=supervised,
            target_column=target_column,
        )

        # Create a DataFrame with reduced embeddings and language info
        viz_df = pl.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "language": self.df["language_name"],
            }
        ).filter(pl.col("language").is_in(top_languages))

        # Create 2D scatter plot
        plt.figure(figsize=(15, 12))

        # Create color palette
        palette = sns.color_palette("husl", n_colors=len(top_languages))

        # Plot each language
        for i, lang in enumerate(top_languages):
            lang_data = viz_df.filter(pl.col("language") == lang)

            plt.scatter(
                lang_data["x"].to_numpy(),
                lang_data["y"].to_numpy(),
                label=lang,
                color=palette[i],
                alpha=0.6,
                s=50,
            )

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"2D {method.upper()} Visualization by Language")

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        kind = f"2d_{method}_{'supervised' if supervised else 'unsupervised'}"
        # Save plot
        plt.tight_layout()
        plt.savefig(
            self.plot_dir / f"{kind}_language_visualization.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def visualize_2d_by_task(
        self,
        n_tasks: int = 10,
        method: Literal["umap", "pca"] = "umap",
        supervised: bool = False,
        target_column: str = "task_name",
    ):
        """
        Create a 2D visualization of embeddings colored by task.

        Args:
            n_tasks: Number of top tasks to include
            method: Dimensionality reduction method ('umap' or 'pca')
            supervised: Whether to use supervised reduction
            target_column: Column to use for supervised reduction
        """
        # Get top tasks
        top_tasks = self.get_top_tasks(n_tasks)

        # Apply dimensionality reduction
        reduced_embeddings = self.reduce_dimensions(
            n_components=2,
            method=method,
            supervised=supervised,
            target_column=target_column,
        )

        # Create a DataFrame with reduced embeddings and task info
        viz_df = pl.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "task": self.df["task_name"],
            }
        ).filter(pl.col("task").is_in(top_tasks))

        # Create 2D scatter plot
        plt.figure(figsize=(15, 12))

        # Create color palette
        palette = sns.color_palette("viridis", n_colors=len(top_tasks))

        # Plot each task
        for i, task in enumerate(top_tasks):
            task_data = viz_df.filter(pl.col("task") == task)

            plt.scatter(
                task_data["x"].to_numpy(),
                task_data["y"].to_numpy(),
                label=task,
                color=palette[i],
                alpha=0.6,
                s=50,
            )

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"2D {method.upper()} Visualization by Task")

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save plot
        kind = f"2d_{method}_{'supervised' if supervised else 'unsupervised'}"
        plt.tight_layout()
        plt.savefig(
            self.plot_dir / f"{kind}_task_visualization.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def run_pipeline(self):
        """Run the full visualization pipeline."""
        self.load_dataset()
        self.generate_embeddings()

        # Unsupervised UMAP
        self.visualize_3d_by_language()
        self.visualize_2d_by_language()
        self.visualize_2d_by_task()

        # Supervised UMAP
        self.visualize_3d_by_language(supervised=True)
        self.visualize_2d_by_language(supervised=True)
        self.visualize_2d_by_task(supervised=True)

        # PCA
        self.visualize_3d_by_language(method="pca")
        self.visualize_2d_by_language(method="pca")
        self.visualize_2d_by_task(method="pca")

        print("All visualizations generated successfully!")


def main():
    """Main function to run the visualization pipeline."""
    visualizer = CodeEmbeddingVisualizer(
        cache_dir="../cache", plot_dir="../plots/embeddings"
    )
    visualizer.run_pipeline()


if __name__ == "__main__":
    main()

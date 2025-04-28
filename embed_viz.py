import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import os
import hashlib


class CodeEmbeddingVisualizer:
    """Class to handle embedding and visualization of code datasets."""
    
    def __init__(self, data_path, cache_dir='cache'):
        """
        Initialize the visualizer.
        
        Args:
            data_path: Path to the dataset
            cache_dir: Directory to store cache files
        """
        # Set style for better visualizations
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 10)
        
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.df = None
        self.embeddings = None
        self.reduced_embeddings_3d = None
        self.reduced_embeddings_2d = None
        self.data_hash = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_dataset(self):
        """Load the dataset and compute a hash for caching."""
        print(f"Loading dataset from {self.data_path}")
        self.df = pl.read_parquet(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of unique languages: {self.df['language_name'].n_unique()}")
        print(f"Number of unique tasks: {self.df['task_name'].n_unique()}")
        
        # Calculate a hash of the dataset to use for caching
        self.data_hash = hashlib.md5(
            str(self.df.shape).encode() + 
            str(self.df.head(10).to_dict()).encode()
        ).hexdigest()
        
        return self.df
    
    def generate_embeddings(self):
        """Generate or load cached embeddings for the code samples."""
        embeddings_cache_file = os.path.join(self.cache_dir, f"embeddings_cache_{self.data_hash}.npy")
        
        # Check if embeddings cache exists
        if os.path.exists(embeddings_cache_file):
            print(f"Loading embeddings from cache file: {embeddings_cache_file}")
            self.embeddings = np.load(embeddings_cache_file)
        else:
            # Load CodeBERT model and tokenizer
            print("Cache not found. Loading CodeBERT model...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.compile()
            model.eval()

            # Generate embeddings
            print("Generating embeddings...")
            embeddings = []

            # Process each code sample
            for code in tqdm(self.df.select('code').fill_null("").to_numpy().flatten()):
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
            inputs = tokenizer(code_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Use CLS token embedding as the representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        return embedding
    
    def apply_umap_3d(self):
        """Apply UMAP dimensionality reduction to get 3D embeddings."""
        umap_cache_file = os.path.join(self.cache_dir, f"umap_3d_{self.data_hash}.npy")
        
        if os.path.exists(umap_cache_file):
            print(f"Loading 3D UMAP results from cache: {umap_cache_file}")
            self.reduced_embeddings_3d = np.load(umap_cache_file)
        else:
            print("Computing 3D UMAP dimensionality reduction...")
            reducer = UMAP(n_neighbors=15, n_components=3, min_dist=0.1, metric='cosine')
            self.reduced_embeddings_3d = reducer.fit_transform(self.embeddings)
            print(f"Saving 3D UMAP results to cache: {umap_cache_file}")
            np.save(umap_cache_file, self.reduced_embeddings_3d)
            
        return self.reduced_embeddings_3d
    
    def apply_umap_2d(self):
        """Apply UMAP dimensionality reduction to get 2D embeddings."""
        umap_2d_cache_file = os.path.join(self.cache_dir, f"umap_2d_{self.data_hash}.npy")
        
        if os.path.exists(umap_2d_cache_file):
            print(f"Loading 2D UMAP results from cache: {umap_2d_cache_file}")
            self.reduced_embeddings_2d = np.load(umap_2d_cache_file)
        else:
            print("Computing 2D UMAP dimensionality reduction...")
            reducer_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine')
            self.reduced_embeddings_2d = reducer_2d.fit_transform(self.embeddings)
            print(f"Saving 2D UMAP results to cache: {umap_2d_cache_file}")
            np.save(umap_2d_cache_file, self.reduced_embeddings_2d)
            
        return self.reduced_embeddings_2d
    
    def get_top_languages(self, n=20):
        """Get the top n most frequent languages in the dataset."""
        language_counts = self.df.group_by('language_name').agg(pl.len()).sort('count', descending=True)
        return language_counts.head(n)['language_name'].to_list()
    
    def get_top_tasks(self, n=10):
        """Get the top n most frequent tasks in the dataset."""
        task_counts = self.df.group_by('task_name').agg(pl.len()).sort('count', descending=True)
        return task_counts.head(n)['task_name'].to_list()
    
    def visualize_3d_by_language(self, output_file='code_embeddings_3d.png'):
        """Create a 3D visualization of code embeddings colored by language."""
        if self.reduced_embeddings_3d is None:
            self.apply_umap_3d()
        
        # Select subset of popular languages
        top_languages = self.get_top_languages()
        language_mask = self.df['language_name'].is_in(top_languages)
        filtered_df = self.df.filter(language_mask)
        
        # Get the indices where language_mask is True
        language_indices = np.where(language_mask.to_numpy())[0]
        filtered_embeddings = self.reduced_embeddings_3d[language_indices]
        
        # Create a DataFrame for visualization
        viz_df = pl.DataFrame({
            'x': filtered_embeddings[:, 0],
            'y': filtered_embeddings[:, 1],
            'z': filtered_embeddings[:, 2],
            'language': filtered_df['language_name'],
            'task': filtered_df['task_name']
        })
        
        # 3D visualization with matplotlib
        print("Creating 3D visualization...")
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a color palette for languages
        unique_languages = viz_df['language'].unique().to_list()
        palette = sns.color_palette("husl", len(unique_languages))
        color_dict = dict(zip(unique_languages, palette))
        
        # Plot each language with a different color
        for language in unique_languages:
            group = viz_df.filter(pl.col('language') == language)
            ax.scatter(
                group['x'].to_numpy(),
                group['y'].to_numpy(),
                group['z'].to_numpy(),
                label=language,
                color=color_dict[language],
                s=30, alpha=0.7
            )
        
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        ax.set_title('Code Embeddings by Programming Language (CodeBERT + UMAP)', fontsize=14)
        
        # Add a legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D visualization saved to {output_file}")
    
    def visualize_2d_by_language(self, output_file='code_embeddings_2d.png'):
        """Create a 2D visualization of code embeddings colored by language."""
        if self.reduced_embeddings_2d is None:
            self.apply_umap_2d()
        
        # Select subset of popular languages
        top_languages = self.get_top_languages()
        language_mask = self.df['language_name'].is_in(top_languages)
        filtered_df = self.df.filter(language_mask)
        
        # Get the indices where language_mask is True
        language_indices = np.where(language_mask.to_numpy())[0]
        filtered_embeddings_2d = self.reduced_embeddings_2d[language_indices]
        
        # Create a DataFrame for 2D visualization
        viz_df_2d = pl.DataFrame({
            'x': filtered_embeddings_2d[:, 0],
            'y': filtered_embeddings_2d[:, 1],
            'language': filtered_df['language_name'],
            'task': filtered_df['task_name']
        })
        
        # 2D visualization with seaborn - convert to numpy arrays for plotting
        print("Creating 2D visualization by language...")
        plt.figure(figsize=(16, 12))
        
        # For seaborn plotting, convert to dict of numpy arrays
        plot_data = {
            'x': viz_df_2d['x'].to_numpy(),
            'y': viz_df_2d['y'].to_numpy(),
            'language': viz_df_2d['language'].to_numpy()
        }
        
        sns.scatterplot(
            x='x', y='y', hue='language',
            data=plot_data,
            palette='husl',
            s=100,
            alpha=0.7
        )
        
        plt.title('Code Embeddings by Programming Language (2D)', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2D language visualization saved to {output_file}")
    
    def visualize_2d_by_task(self, output_file='code_embeddings_by_task.png'):
        """Create a 2D visualization of code embeddings colored by task."""
        if self.reduced_embeddings_2d is None:
            self.apply_umap_2d()
        
        # Get top tasks
        top_tasks = self.get_top_tasks()
        task_mask = self.df['task_name'].is_in(top_tasks)
        task_filtered_df = self.df.filter(task_mask)
        
        # Get the indices where task_mask is True
        task_indices = np.where(task_mask.to_numpy())[0]
        task_filtered_embeddings_2d = self.reduced_embeddings_2d[task_indices]
        
        # Create a DataFrame for task visualization
        task_viz_df = pl.DataFrame({
            'x': task_filtered_embeddings_2d[:, 0],
            'y': task_filtered_embeddings_2d[:, 1],
            'language': task_filtered_df['language_name'],
            'task': task_filtered_df['task_name']
        })
        
        plt.figure(figsize=(16, 12))
        
        # For seaborn plotting, convert to dict of numpy arrays
        task_plot_data = {
            'x': task_viz_df['x'].to_numpy(),
            'y': task_viz_df['y'].to_numpy(),
            'task': task_viz_df['task'].to_numpy()
        }
        
        print("Creating 2D visualization by task...")
        sns.scatterplot(
            x='x', y='y', hue='task',
            data=task_plot_data,
            palette='viridis',
            s=100,
            alpha=0.7
        )
        
        plt.title('Code Embeddings by Task (2D)', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Task visualization saved to {output_file}")
    
    def run_pipeline(self):
        """Run the full visualization pipeline."""
        self.load_dataset()
        self.generate_embeddings()
        self.apply_umap_3d()
        self.apply_umap_2d()
        self.visualize_3d_by_language()
        self.visualize_2d_by_language()
        self.visualize_2d_by_task()
        print("All visualizations generated successfully!")


def main():
    """Main function to run the visualization pipeline."""
    visualizer = CodeEmbeddingVisualizer(
        'hf://datasets/christopher/rosetta-code/data/train-00000-of-00001-8b4da49264116bbf.parquet',
        cache_dir='cache'
    )
    visualizer.run_pipeline()


if __name__ == "__main__":
    main()

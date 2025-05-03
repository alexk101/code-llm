import re
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from urllib.parse import urlparse
import argparse
import os
from collections import defaultdict


class MarkdownGraphVisualizer:
    """Class to create and visualize a graph of markdown files and their links."""
    
    def __init__(self, resources_dir='cache/md_resources'):
        """Initialize the visualizer with the path to the markdown resources."""
        self.resources_dir = Path(resources_dir)
        self.graph = nx.DiGraph()
        self.http_node = "external-http"  # Special node for all external HTTP links
        self.md_files = []
        self.subjects = {}  # Map of directory names to normalized subjects
        self.modules = set()  # Set of module directories
        
        # Regular expressions for extracting links
        self.md_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.html_link_pattern = re.compile(r'<a\s+(?:[^>]*?)href="([^"]*)"[^>]*>(.*?)</a>', re.IGNORECASE)
        
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
            raise FileNotFoundError(f"Resources directory not found: {self.resources_dir}")
            
        # Recursively find all markdown files
        self.md_files = list(self.resources_dir.glob('**/*.md'))
        print(f"Found {len(self.md_files)} markdown files")
        
        # Find all subdirectories (modules)
        all_dirs = set()
        for md_file in self.md_files:
            # Add all parent directories up to resources_dir
            current = md_file.parent
            while current != self.resources_dir and current != current.parent:
                all_dirs.add(current)
                current = current.parent
        
        # Convert to relative paths
        self.modules = {d.relative_to(self.resources_dir) for d in all_dirs}
        print(f"Found {len(self.modules)} module directories")
        
    def extract_links_from_md(self, md_file):
        """Extract links from a markdown file."""
        try:
            with open(md_file, 'r', encoding='utf-8', errors='replace') as f:
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
        return parsed.scheme in ('http', 'https')
    
    def normalize_internal_link(self, link, source_file):
        """Normalize an internal link to a Path object relative to resources_dir."""
        # Remove URL fragments
        link = link.split('#')[0]
        
        # If empty after removing fragment, it's a self-reference
        if not link:
            return source_file
            
        # Handle relative links
        source_dir = source_file.parent
        if link.startswith('./'):
            link = link[2:]
            target = source_dir / link
        elif link.startswith('../'):
            # Handle multiple parent directory references
            target_path = source_dir
            while link.startswith('../'):
                target_path = target_path.parent
                link = link[3:]
            target = target_path / link
        else:
            target = source_dir / link
            
        # Ensure the link has .md extension
        if not target.suffix and not target.exists():
            target = target.with_suffix('.md')
            
        # Convert to relative path from resources_dir
        try:
            return target.relative_to(self.resources_dir)
        except ValueError:
            # If we can't get a relative path, return the absolute path
            return target
    
    def get_subject_for_path(self, path):
        """Determine the subject for a given file or directory path."""
        if isinstance(path, Path):
            path = str(path)
            
        # For the external HTTP node, return 'external'
        if path == self.http_node:
            return 'external'
            
        # Check the top-level directory - this is the subject
        top_dir = path.split(os.sep)[0] if os.sep in path else path
        
        # Use the normalized subject name if available
        return self.subjects.get(top_dir, top_dir.lower())
    
    def add_node_with_attributes(self, node_id, node_type, parent=None):
        """Add a node to the graph with appropriate attributes."""
        # Determine subject based on path
        subject = self.get_subject_for_path(node_id)
        
        # Add the node with attributes - avoid None values since GraphML doesn't support them
        attrs = {
            'type': node_type, 
            'subject': subject
        }
        
        # Only add parent if it's not None
        if parent is not None:
            attrs['parent'] = parent
            
        self.graph.add_node(node_id, **attrs)
        
        # If there's a parent, add an edge from parent to node
        if parent:
            self.graph.add_edge(parent, node_id, type='hierarchy')
    
    def build_graph(self):
        """Build the graph from markdown files and their links."""
        self.load_subjects()
        self.find_markdown_files()
        
        # Print debugging information about directory structure
        print(f"\nMapped subjects and their raw directory names:")
        for dir_name, subj in self.subjects.items():
            print(f"  Directory: '{dir_name}' → Subject: '{subj}'")
        
        # Add special node for external links
        self.graph.add_node(self.http_node, type='external', subject='external')
        
        # First add all module directories as nodes
        for module in self.modules:
            module_str = str(module)
            # Find parent module if any
            parent = None
            if os.sep in module_str:
                parent_path = os.path.dirname(module_str)
                if parent_path:
                    parent = parent_path
            
            self.add_node_with_attributes(module_str, 'module', parent)
        
        # Add all markdown files as nodes
        for md_file in self.md_files:
            rel_path = str(md_file.relative_to(self.resources_dir))
            parent = os.path.dirname(rel_path)
            if parent == "":
                parent = None
            
            self.add_node_with_attributes(rel_path, 'file', parent)
        
        # Process links and add edges
        for md_file in tqdm(self.md_files, desc="Building graph"):
            rel_source = str(md_file.relative_to(self.resources_dir))
            links = self.extract_links_from_md(md_file)
            
            for link in links:
                if self.is_external_link(link):
                    # Add edge to the HTTP node
                    self.graph.add_edge(rel_source, self.http_node, type='external')
                else:
                    # Process internal link
                    target = self.normalize_internal_link(link, md_file)
                    target_str = str(target)
                    
                    # Skip self-references
                    if target_str == rel_source:
                        continue
                        
                    # Only add edge if target exists in our files
                    try:
                        target_file = self.resources_dir / target
                        if target_file.exists():
                            self.graph.add_edge(rel_source, target_str, type='internal')
                    except Exception as e:
                        print(f"Error processing link {link} from {rel_source}: {e}")
    
    def calculate_graph_stats(self):
        """Calculate and return statistics about the graph."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_files': sum(1 for _, attr in self.graph.nodes(data=True) if attr.get('type') == 'file'),
            'num_modules': sum(1 for _, attr in self.graph.nodes(data=True) if attr.get('type') == 'module'),
            'num_internal_edges': sum(1 for _, _, data in self.graph.edges(data=True) if data['type'] == 'internal'),
            'num_external_edges': sum(1 for _, _, data in self.graph.edges(data=True) if data['type'] == 'external'),
            'num_hierarchy_edges': sum(1 for _, _, data in self.graph.edges(data=True) if data['type'] == 'hierarchy'),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'isolated_nodes': list(nx.isolates(self.graph)),
            'num_isolated_nodes': len(list(nx.isolates(self.graph))),
        }
        
        # Count subjects
        subject_counts = defaultdict(int)
        for _, attr in self.graph.nodes(data=True):
            subject = attr.get('subject', 'unknown')
            subject_counts[subject] += 1
        
        stats['subjects'] = dict(subject_counts)
        
        # Find nodes with highest in-degree (most linked to)
        in_degrees = sorted([(n, d) for n, d in self.graph.in_degree() if n != self.http_node], 
                           key=lambda x: x[1], reverse=True)
        stats['top_linked_to'] = in_degrees[:10] if in_degrees else []
        
        # Find nodes with highest out-degree (most links from)
        out_degrees = sorted(self.graph.out_degree(), key=lambda x: x[1], reverse=True)
        stats['top_linking_from'] = out_degrees[:10] if out_degrees else []
        
        return stats
    
    def get_node_colors(self):
        """Generate node colors based on subject."""
        # Use seaborn's tab10 color palette
        default_colors = sns.color_palette("tab10").as_hex()
        
        # Special colors
        special_colors = {
            'external': '#FF5733',  # Red for external
            'unknown': '#CCCCCC',   # Gray for unknown
        }
        
        # Extract all unique subjects from the graph nodes
        all_subjects = set()
        for _, attr in self.graph.nodes(data=True):
            subject = attr.get('subject', 'unknown')
            if subject not in ['external', 'unknown']:
                all_subjects.add(subject)
        
        # Create a color map for all subjects
        subject_colors = dict(special_colors)  # Start with special colors
        
        # Assign colors to subjects in sorted order for consistency
        color_idx = 0
        for subject in sorted(all_subjects):
            if subject not in subject_colors:
                subject_colors[subject] = default_colors[color_idx % len(default_colors)]
                color_idx += 1
        
        # Generate colors for each node based on their subject
        node_colors = {}
        for node, attr in self.graph.nodes(data=True):
            subject = attr.get('subject', 'unknown')
            node_colors[node] = subject_colors.get(subject, subject_colors['unknown'])
            
        return node_colors, subject_colors
    
    def draw_graph(self, output_file='graph.png', figsize=(24, 20)):
        """Draw and save the graph visualization."""
        plt.figure(figsize=figsize)
        
        # Use graphviz twopi layout
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        
        # Node colors based on subject/language
        node_colors, subject_colors = self.get_node_colors()
        
        # Define node styles based on type
        node_types = {
            'module': {'shape': 's', 'size': 300, 'alpha': 0.8},  # Square for modules
            'file': {'shape': 'o', 'size': 50, 'alpha': 0.7},     # Circle for files
            'external': {'shape': 'd', 'size': 800, 'alpha': 1.0}  # Diamond for external
        }
        
        # Draw nodes by type, with appropriate styles and colors
        for node_type, style in node_types.items():
            # Get nodes of this type
            nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == node_type]
            
            if not nodes:
                continue
                
            # Get colors for these nodes
            colors = [node_colors.get(n, '#CCCCCC') for n in nodes]
            
            if style['shape'] == 'o':  # Circle (default)
                nx.draw_networkx_nodes(
                    self.graph, pos, 
                    nodelist=nodes,
                    node_color=colors, 
                    node_size=style['size'],
                    alpha=style['alpha']
                )
            else:  # Other shapes
                for i, node in enumerate(nodes):
                    plt.scatter(
                        pos[node][0], pos[node][1],
                        s=style['size'],
                        c=colors[i],
                        marker=style['shape'],
                        alpha=style['alpha']
                    )
        
        # Draw edges by type
        edge_styles = {
            'internal': {'color': 'black', 'width': 0.5, 'alpha': 0.5, 'arrows': True},
            'external': {'color': 'red', 'width': 0.7, 'alpha': 0.7, 'arrows': True},
            'hierarchy': {'color': 'grey', 'width': 1.0, 'alpha': 0.3, 'arrows': False, 'style': 'dashed'}
        }
        
        for edge_type, style in edge_styles.items():
            edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data.get('type') == edge_type]
            if edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=edges,
                    width=style.get('width', 0.5),
                    alpha=style.get('alpha', 0.5),
                    arrows=style.get('arrows', True),
                    edge_color=style.get('color', 'black'),
                    style=style.get('style', 'solid')
                )
        
        # Draw labels for important nodes
        important_nodes = {}
        
        # Add the HTTP node
        important_nodes[self.http_node] = "External HTTP"
        
        # Add module nodes
        for node, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'module':
                # Use just the leaf directory name for the label
                label = os.path.basename(node)
                important_nodes[node] = label
                
        # Add highly connected file nodes
        for node, degree in sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:10]:
            if node != self.http_node and self.graph.nodes[node].get('type') == 'file':
                # Use the filename without extension for the label
                label = os.path.basename(node)
                if label.endswith('.md'):
                    label = label[:-3]
                important_nodes[node] = label
        
        # Draw the labels
        nx.draw_networkx_labels(self.graph, pos, important_nodes, font_size=8, font_weight='bold')
        
        # Add a legend for node types and subjects
        legend_elements = []
        
        # Add node type legend with shapes
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker='d', color='w', 
                                      label='External', markerfacecolor=subject_colors.get('external', '#FF5733'), markersize=15))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                      label='Module', markerfacecolor='#CCCCCC', markersize=12))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      label='File', markerfacecolor='#CCCCCC', markersize=8))
        
        # Add subject/language color legend
        subject_stats = self.calculate_graph_stats()['subjects']
        
        # Add legends for each subject (excluding special ones)
        for subject, color in sorted(subject_colors.items()):
            # Skip special categories
            if subject in ['external', 'unknown']:
                continue
                
            # Get count if available
            count = subject_stats.get(subject, 0)
            if count > 0:  # Only show subjects with nodes
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                             label=f'{subject.capitalize()} ({count})', 
                                             markerfacecolor=color, 
                                             markersize=10))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title and remove axis
        plt.title(f"Resource Documentation Graph\n{len(self.md_files)} files, {len(self.modules)} modules, {len(subject_stats)} subjects")
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_file}")
        plt.close()
        
    def save_graph(self, output_file='graph.graphml'):
        """Save the graph in GraphML format for use with other tools."""
        try:
            # Clean graph attributes to ensure it's compatible with GraphML
            graph_copy = self.graph.copy()
            
            # Replace None values with empty strings which GraphML can handle
            for node, attrs in graph_copy.nodes(data=True):
                for key, value in list(attrs.items()):
                    if value is None:
                        attrs[key] = ""
            
            # Do the same for edge attributes
            for u, v, attrs in graph_copy.edges(data=True):
                for key, value in list(attrs.items()):
                    if value is None:
                        attrs[key] = ""
            
            # Now save the cleaned graph
            nx.write_graphml(graph_copy, output_file)
            print(f"Graph data saved to {output_file}")
        except Exception as e:
            print(f"Warning: Could not save graph to GraphML format due to: {e}")
            print("Continuing without saving graph data.")
        
    def print_stats(self, stats=None):
        """Print statistics about the graph."""
        if stats is None:
            stats = self.calculate_graph_stats()
            
        print("\n=== Graph Statistics ===")
        print(f"Nodes: {stats['num_nodes']} (Files: {stats['num_files']}, Modules: {stats['num_modules']}, External: 1)")
        print(f"Edges: {stats['num_edges']} (Internal: {stats['num_internal_edges']}, External: {stats['num_external_edges']}, Hierarchy: {stats['num_hierarchy_edges']})")
        print(f"Connected Components: {stats['connected_components']}")
        print(f"Isolated Nodes: {stats['num_isolated_nodes']}")
        
        print("\nSubjects:")
        for subject, count in sorted(stats['subjects'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {subject.capitalize()}: {count} nodes")
            
        print("\nTop Linked-To Files:")
        for node, degree in stats['top_linked_to'][:5]:
            if self.graph.nodes[node].get('type') == 'file':
                print(f"  - {node}: {degree} incoming links")
            
        print("\nTop Linking-From Files:")
        for node, degree in stats['top_linking_from'][:5]:
            if node != self.http_node and self.graph.nodes[node].get('type') == 'file':
                print(f"  - {node}: {degree} outgoing links")
    
    def run(self, draw=True, save=True, output_image='graph.png', output_data='graph.graphml'):
        """Run the full pipeline."""
        self.build_graph()
        stats = self.calculate_graph_stats()
        self.print_stats(stats)
        
        if draw:
            print(f"Drawing graph to {output_image}")
            self.draw_graph(output_file=output_image)
        
        if save:
            print(f"Saving graph to {output_data}")
            self.save_graph(output_file=output_data)


def main():
    """Main function to parse arguments and run the graph builder."""
    parser = argparse.ArgumentParser(description='Build a graph of markdown resources')
    parser.add_argument('--resources', default='cache/md_resources', 
                       help='Path to markdown resources')
    parser.add_argument('--output-image', default='plots/graph.png',
                       help='Output image file')
    parser.add_argument('--output-data', default='graph.graphml',
                       help='Output graph data file')
    parser.add_argument('--no-draw', action='store_true',
                       help='Skip drawing the graph image')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving graph data')
    
    args = parser.parse_args()
    
    visualizer = MarkdownGraphVisualizer(
        resources_dir=args.resources
    )
    visualizer.run(
        draw=not args.no_draw, 
        save=not args.no_save,
        output_image=args.output_image, 
        output_data=args.output_data
    )


if __name__ == "__main__":
    main()

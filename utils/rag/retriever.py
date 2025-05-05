"""
Retriever module for hybrid Graph RAG operations.

This module implements the hybrid retrieval approach that combines
vector similarity search with graph-based filtering and reranking.
"""

import time
from typing import Dict, List

import requests

from .config import GraphRAGConfig
from .embedding import EmbeddingManager
from .graph_processor import GraphProcessor
from .vectordb import MilvusManager


class HybridRetriever:
    """Hybrid retriever combining vector search with graph operations."""

    def __init__(
        self,
        vector_db: MilvusManager,
        graph_processor: GraphProcessor,
        embedding_manager: EmbeddingManager,
        config: GraphRAGConfig,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_db: Vector database manager
            graph_processor: Graph processor
            embedding_manager: Embedding manager
            config: GraphRAG configuration
        """
        self.vector_db = vector_db
        self.graph_processor = graph_processor
        self.embedding_manager = embedding_manager
        self.config = config
        self.retriever_config = config.retriever
        self.llm_config = config.llm

        # LLM for generation
        self.llm = None

    def retrieve(
        self,
        query: str,
        query_embedding: List[float] = None,
        top_k: int = None,
        filters: Dict = None,
        rerank: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """Perform hybrid retrieval using vector search and graph relationships.

        Args:
            query: The query string
            query_embedding: Pre-computed query embedding (optional)
            top_k: Number of results to return
            filters: Optional filters to apply to vector search
            rerank: Whether to rerank results using graph relationships
            verbose: Whether to print verbose retrieval information

        Returns:
            Dictionary with retrieval results
        """
        start_time = time.time()

        # Get configuration for retrieval
        top_k_semantic = self.retriever_config["top_k_semantic"]
        top_k_final = top_k or self.retriever_config["top_k_final"]
        include_relationships = self.retriever_config["include_relationships"]

        # Generate embedding if not provided
        if query_embedding is None:
            if verbose:
                print("Generating query embedding...")
            query_embedding = self.embedding_manager.embed_query(query)

        # Step 1: Initial vector similarity search
        if verbose:
            print(f"Performing initial vector search with top_k={top_k_semantic}...")
        vector_results = self.vector_db.search_passages(
            query_embedding=query_embedding, top_k=top_k_semantic, filters=filters
        )

        if verbose:
            print(f"Found {len(vector_results)} initial matches from vector search")

        # Step 2: Retrieve relevant relationships if enabled
        relationship_results = []
        if include_relationships:
            if verbose:
                print("Retrieving relevant relationships...")
            relationship_results = self.vector_db.search_relationships(
                query_embedding=query_embedding, top_k=top_k_semantic
            )

            # Extract node IDs from relationships
            related_node_ids = set()
            for rel in relationship_results:
                related_node_ids.add(rel["source_id"])
                related_node_ids.add(rel["target_id"])

            if verbose:
                print(
                    f"Found {len(relationship_results)} ",
                    f"relevant relationships involving {len(related_node_ids)} nodes",
                )

        # Step 3: Rerank and filter results if enabled
        final_results = vector_results
        if rerank:
            if verbose:
                print("Reranking results using graph relationships...")
            final_results = self._rerank_results(
                vector_results=vector_results,
                relationship_results=relationship_results,
                verbose=verbose,
            )

        # Step 4: Limit to top_k_final results
        final_results = final_results[:top_k_final]

        # Build the response
        elapsed_time = time.time() - start_time
        response = {
            "query": query,
            "passages": final_results,
            "total_results": len(final_results),
            "elapsed_time": elapsed_time,
        }

        if include_relationships and relationship_results:
            response["relationships"] = relationship_results[
                :10
            ]  # Limit relationships in response

        if verbose:
            print(
                f"Retrieval completed in {elapsed_time:.3f} seconds, ",
                f"returning {len(final_results)} results",
            )

        return response

    def _rerank_results(
        self,
        vector_results: List[Dict],
        relationship_results: List[Dict],
        verbose: bool = False,
    ) -> List[Dict]:
        """Rerank the results using graph relationship information.

        Args:
            query: The original query
            query_embedding: The query embedding
            vector_results: Results from vector search
            relationship_results: Results from relationship search
            verbose: Whether to print verbose information

        Returns:
            Reranked list of results
        """
        # Map node IDs to their vector search scores for quick lookup
        node_id_to_score = {result["id"]: result["score"] for result in vector_results}

        # Get the set of all relevant node IDs from vector search
        vector_node_ids = set(node_id_to_score.keys())

        # Create a set of expanded node IDs from relationships
        expanded_node_ids = set()
        # Maps node IDs to their related nodes and relationship types
        relationship_map = {}

        if relationship_results:
            for rel in relationship_results:
                source_id = rel["source_id"]
                target_id = rel["target_id"]
                rel_type = rel["relation_type"]
                rel_weight = rel.get("weight", 1.0)

                # Add to expanded set
                expanded_node_ids.add(source_id)
                expanded_node_ids.add(target_id)

                # Build relationship map
                if source_id not in relationship_map:
                    relationship_map[source_id] = []
                if target_id not in relationship_map:
                    relationship_map[target_id] = []

                # Add bidirectional relationships
                relationship_map[source_id].append(
                    {
                        "node_id": target_id,
                        "type": rel_type,
                        "weight": rel_weight,
                        "direction": "outgoing",
                    }
                )
                relationship_map[target_id].append(
                    {
                        "node_id": source_id,
                        "type": rel_type,
                        "weight": rel_weight,
                        "direction": "incoming",
                    }
                )

        # Find nodes that are in the expanded set but not in the vector search results
        missing_node_ids = expanded_node_ids - vector_node_ids

        # Fetch information about these missing nodes
        missing_nodes = {}
        for node_id in missing_node_ids:
            node = self.graph_processor.get_node_by_id(node_id)
            if node:
                missing_nodes[node_id] = node

        if verbose and missing_nodes:
            print(f"Found {len(missing_nodes)} additional nodes from relationships")

        # Combine original vector results with missing nodes
        combined_results = []

        # Add original vector results with their scores
        for result in vector_results:
            node_id = result["id"]
            result["retrieval_source"] = "vector"
            result["relationships"] = relationship_map.get(node_id, [])
            combined_results.append(result)

        # Add missing nodes from relationships
        for node_id, node in missing_nodes.items():
            # Compute a score for this node based on its relationships
            rel_score = 0.0
            related_nodes = relationship_map.get(node_id, [])

            for related in related_nodes:
                related_id = related["node_id"]
                # If the related node has a vector score,
                # use it to influence this node's score
                if related_id in node_id_to_score:
                    rel_weight = related["weight"]
                    rel_score += node_id_to_score[related_id] * rel_weight

            # Average the score if there are multiple relationships
            if related_nodes:
                rel_score /= len(related_nodes)

            # Create a result entry for this missing node
            result = {
                "id": node_id,
                "title": node.get("title", ""),
                "text": node.get("text", ""),
                "subject": node.get("subject", "unknown"),
                "node_type": node.get("type", "unknown"),
                "score": rel_score
                * 0.8,  # Slightly penalize relationship-based entries
                "retrieval_source": "relationship",
                "relationships": related_nodes,
            }
            combined_results.append(result)

        # Sort the combined results by score
        reranked_results = sorted(
            combined_results, key=lambda x: x["score"], reverse=True
        )

        # Apply additional graph-based ranking if there are relationships
        if relationship_map:
            # Give a boost to nodes that form connected subgraphs
            node_boost = {}

            for result in reranked_results:
                node_id = result["id"]
                relationships = result.get("relationships", [])

                # Count how many relationships this node
                # has with other nodes in the results
                connected_count = 0
                for rel in relationships:
                    related_id = rel["node_id"]
                    if related_id in node_id_to_score or related_id in missing_nodes:
                        connected_count += 1

                # Apply a boost based on connectedness
                connectedness_boost = min(0.1 * connected_count, 0.3)  # Cap at 0.3
                node_boost[node_id] = connectedness_boost

            # Apply the boost
            for result in reranked_results:
                node_id = result["id"]
                if node_id in node_boost:
                    result["score"] += node_boost[node_id]
                    if "original_score" not in result:
                        result["original_score"] = result["score"] - node_boost[node_id]

            # Re-sort after applying boosts
            reranked_results = sorted(
                reranked_results, key=lambda x: x["score"], reverse=True
            )

        return reranked_results

    def generate_response(self, query: str, context: str, provider: str = None) -> str:
        """Generate a response using an LLM with the retrieved context.

        Args:
            query: The user query
            context: The retrieved context formatted as a string
            provider: Override for the LLM provider

        Returns:
            Generated response
        """
        # Use default provider from config if not specified
        llm_provider = provider or self.llm_config["provider"]
        model_name = self.llm_config["model_name"]
        temperature = self.llm_config["temperature"]
        max_tokens = self.llm_config["max_tokens"]
        system_prompt = self.llm_config["system_prompt"]

        # Initialize the LLM if not already done
        if self.llm is None:
            self._initialize_llm(llm_provider)

        # Generate response based on provider
        if llm_provider == "local":
            return self._generate_with_local_llm(query, context, system_prompt)
        elif llm_provider == "openai":
            return self._generate_with_openai(
                query, context, model_name, system_prompt, temperature, max_tokens
            )
        elif llm_provider == "http":
            return self._generate_with_http_api(
                query, context, system_prompt, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def _initialize_llm(self, provider: str):
        """Initialize the LLM based on the provider.

        Args:
            provider: The LLM provider
        """
        if provider == "local":
            try:
                # Try to import LLaMA.cpp Python bindings
                from llama_cpp import Llama

                model_name = self.llm_config["model_name"]
                model_path = self.llm_config.get(
                    "model_path", f"models/{model_name}.gguf"
                )

                print(f"Loading local LLM model: {model_path}")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,  # Context window size
                    n_batch=512,  # Batch size for prompt processing
                    n_gpu_layers=-1,  # Auto-detect number of layers to offload to GPU
                )
                print("Local LLM loaded successfully")
            except ImportError as err:
                raise ImportError(
                    "llama-cpp-python package is required for local LLM functionality. "
                    "Please install it in your environment using: "
                    "pip install llama-cpp-python"
                ) from err
        elif provider == "http":
            # For HTTP API-based LLMs, we don't need special initialization
            # We'll just use the requests library directly
            # Check if we can connect to the API
            api_url = self.llm_config.get("api_url")
            if not api_url:
                raise ValueError("API URL not provided for HTTP LLM provider")

            try:
                # Try to get info on the API endpoint
                base_url = "/".join(
                    api_url.split("/")[:-2]
                )  # Remove /chat/completions part
                models_url = f"{base_url}/models"
                response = requests.get(models_url)

                if response.status_code == 200:
                    print(f"Successfully connected to HTTP LLM API at {base_url}")
                else:
                    print(f"Warning: API status code {response.status_code}")
            except Exception as e:
                print(f"Warning: Could not connect to HTTP LLM API: {e}")
                print("Will attempt to use the API when generating a response")

    def _generate_with_local_llm(
        self, query: str, context: str, system_prompt: str
    ) -> str:
        """Generate a response using a local LLM.

        Args:
            query: The user query
            context: The retrieved context
            system_prompt: The system prompt

        Returns:
            Generated response
        """
        # Format the prompt for the local LLM
        prompt = f"""
{system_prompt}

CONTEXT:
{context}

USER QUERY:
{query}

RESPONSE:
"""

        try:
            # Generate completion
            completion = self.llm(
                prompt,
                max_tokens=self.llm_config["max_tokens"],
                temperature=self.llm_config["temperature"],
                stop=["USER QUERY:", "CONTEXT:"],
            )

            return completion["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating response with local LLM: {e}")
            return f"Error generating response: {str(e)}"

    def _generate_with_openai(
        self,
        query: str,
        context: str,
        model_name: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response using OpenAI.

        Args:
            query: The user query
            context: The retrieved context
            model_name: The model name to use
            system_prompt: The system prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate

        Returns:
            Generated response
        """
        try:
            response = self.llm.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"

    def _generate_with_http_api(
        self,
        query: str,
        context: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response using an HTTP API.

        This method handles HTTP API endpoints that follow the OpenAI-compatible format,
        like those provided by LLaMA.cpp, local LLM servers, or LMStudio.

        Args:
            query: The user query
            context: The retrieved context
            system_prompt: The system prompt
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate

        Returns:
            Generated response
        """
        api_url = self.llm_config.get("api_url")
        model_name = self.llm_config.get("model_name", "local-model")
        api_key = self.llm_config.get("api_key")

        headers = {"Content-Type": "application/json"}

        # Add authorization header if api_key is provided
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Prepare the request data following OpenAI API format
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            # Make the request
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data,
                timeout=60,  # Set a generous timeout
            )

            # Check for successful response
            if response.status_code == 200:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Unexpected response format: {result}")
                    return "Error: Unexpected response format from API"
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error: API returned status code {response.status_code}"

        except Exception as e:
            print(f"Error generating response with HTTP API: {e}")
            return f"Error generating response: {str(e)}"

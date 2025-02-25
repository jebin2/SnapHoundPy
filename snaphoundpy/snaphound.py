from .model import load_model
import os
from PIL import Image
import numpy as np
import faiss
import torch
from .db_processor import DatabaseManager
import pickle
from typing import Optional, List, Tuple, Set
import json
from dotenv import load_dotenv

class PathParser:
	@staticmethod
	def normalize_path(path: str) -> str:
		"""Normalize path format removing trailing slashes and stars."""
		path = path.rstrip('/*').rstrip('/')
		return os.path.normpath(path)

	@staticmethod
	def get_path_pattern(path: str) -> str:
		"""Convert path to glob pattern if needed."""
		if path.endswith('*'):
			base_path = path.rstrip('*')
			return os.path.join(base_path, '**')
		return path

	@staticmethod
	def expand_paths(paths: List[str]) -> Set[str]:
		"""Expand paths containing wildcards and return only directories."""
		expanded_paths = set()

		for path in paths:
			try:
				is_full_recursive = path.endswith("/**")
				is_recursive = path.endswith("/*") and not is_full_recursive
				base_path = path.rstrip("/**").rstrip("/*")
				abs_path = os.path.abspath(os.path.expanduser(base_path))

				if os.path.isdir(abs_path):
					expanded_paths.add(abs_path)  # Always include the base path

					if is_recursive or is_full_recursive:
						for root, dirs, _ in os.walk(abs_path):
							expanded_paths.update(os.path.join(root, d) for d in dirs)
							if not is_full_recursive:
								break  # Stop after the first level if not fully recursive
			except:
				pass

		return expanded_paths

class SnapHound:
	PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
	FULL_DB_INFO = f'''{{
		"DATABASE": "{PACKAGE_DIR}/snaphound.db",
		"BACKUP_DATABASE": "{PACKAGE_DIR}/snaphound.db.bak",
		"TABLE_NAME": "snaphound",
		"COLUMNS": {{
			"id": {{"index": 0, "name": "id", "type": "integer"}},
			"image_path": {{"index": 1, "name": "image_path", "type": "text"}},
			"embedding": {{"index": 2, "name": "embedding", "type": "BLOB"}}
		}}
	}}'''

	def __init__(self, paths: List[str] = [], priority_paths: List[str] = [], exclude_paths: List[str] = [], search_value: str = None, search_path: str = None, index=True):
		load_dotenv()

		self.__process_path(paths, priority_paths, exclude_paths)
		
		# Database connection
		self.conn = DatabaseManager(json.loads(self.FULL_DB_INFO))
		self.model, self.processor = load_model()
		self.search_value = search_value
		self.search_path = search_path
		self.all_searched_once = False
		# Start indexing
		if index:
			self.__index_images()

	def _is_excluded(self, path: str) -> bool:
		"""Check if a path should be excluded based on exclude_paths."""
		normalized_path = self.path_parser.normalize_path(path)
		for exclude in self.exclude_paths:
			if normalized_path.startswith(exclude):
				return True
		return False

	def _is_valid_image_path(self, path: str) -> bool:
		"""Check if path is a valid image file and not in excluded paths."""
		if not os.path.isfile(path):
			return False
		if self._is_excluded(path):
			return False
		return path.lower().endswith((".jpg", ".jpeg", ".png"))

	def __process_path(self, paths: List[str] = [], priority_paths: List[str] = [], exclude_paths: List[str] = []):
		self.path_parser = PathParser()

		self.exclude_paths = []
		if exclude_paths:
			self.exclude_paths.extend(self.path_parser.expand_paths(exclude_paths))

		self.all_paths = []
		if paths:
			self.all_paths.extend(self.path_parser.expand_paths(paths))
		if priority_paths:
			self.all_paths.extend(self.path_parser.expand_paths(priority_paths))
		
		# Remove excluded paths
		self.all_paths = [p for p in self.all_paths if not self._is_excluded(p)]

		print(f"Selected paths: {self.all_paths}")

	def __index_images(self):
		"""Index images in a background thread."""
		indexed_files = set()  # Track already indexed files to avoid duplicates
		
		for base_path in self.all_paths:
			try:
				# Handle directory
				if os.path.isdir(base_path):
					for root, _, files in os.walk(base_path):
						for filename in files:
							full_path = os.path.normpath(os.path.join(root, filename))
							
							# Skip if already processed or invalid
							if full_path in indexed_files:
								continue
								
							self._process_image(full_path)
							indexed_files.add(full_path)
				
				# Handle single file
				elif base_path not in indexed_files:
					self._process_image(base_path)
					indexed_files.add(base_path)

			except Exception as e:
				print(f"Error processing path {base_path}: {str(e)}")

	def _process_image(self, img_path: str):
		"""Process single image file."""
		try:
			if not self._is_valid_image_path(img_path):
				return

			if self.__check_if_indexed(img_path):
				if not self.all_searched_once:
					if self.search_value:
						print(f"Searching with text: {self.search_value}")
						self.search_with_text(self.search_value)

					if self.search_path:
						print(f"Searching with images for: {self.search_path}")
						self.search_with_image(self.search_path)

					self.all_searched_once = True
				return

			print(f"processing... {img_path}")
			image = Image.open(img_path).convert("RGB")
			# Convert image to embedding
			inputs = self.processor(images=image, return_tensors="pt")
			with torch.no_grad():
				img_embedding = self.model.get_image_features(**inputs)
			
			# Normalize embedding
			img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
			embedding_np = img_embedding.squeeze().numpy()
			
			# Store in DB and notify search functions
			self.__store_embedding(img_path, embedding_np)

			print(f"processed {img_path}")

			# Perform search if required
			if self.search_value:
				print(f"Searching with text: {self.search_value}")
				self.search_with_text(self.search_value, [embedding_np], [img_path])

			if self.search_path:
				print(f"Searching with images for: {self.search_path}")
				self.search_with_image(self.search_path, [embedding_np], [img_path])

		except Exception as e:
			print(f"Critical error with {img_path} [{type(e).__name__}]: {str(e)}")

	def __check_if_indexed(self, image_path: str) -> bool:
		"""Check if an image is already indexed in the database."""
		result = self.conn.execute(
			f"SELECT 1 FROM {self.conn.table_name} WHERE image_path = ? LIMIT 1",
			(image_path,),
			"get"
		)
		return bool(result)

	def __store_embedding(self, image_path: str, embedding: np.ndarray):
		"""Stores image path and embedding in SQLite database."""
		embedding_blob = pickle.dumps(embedding)
		self.conn.execute(
			f"INSERT OR REPLACE INTO {self.conn.table_name} (image_path, embedding) VALUES (?, ?)",
			(image_path, embedding_blob)
		)

	def __load_embeddings(self, image_paths_filter=None) -> Tuple[np.ndarray, List[str]]:
		"""Loads embeddings from the database and includes any newly indexed items."""
		# Get all stored embeddings
		data = None
		if image_paths_filter:
			placeholders = ','.join('?' for _ in image_paths_filter)  # Create the correct number of placeholders
			query = f"SELECT image_path, embedding FROM {self.conn.table_name} WHERE image_path IN ({placeholders})"
			print(query)
			data = self.conn.execute(query, image_paths_filter)
		else:
			data = self.conn.execute(f"SELECT image_path, embedding FROM {self.conn.table_name}")
		
		# Convert to lists for easier manipulation
		image_vectors = []
		image_paths = []
		
		# Add stored embeddings
		for path, emb_blob in data:
			# Skip excluded paths
			if self._is_excluded(path):
				continue
			image_vectors.append(pickle.loads(emb_blob))
			image_paths.append(path)
		
		return np.array(image_vectors, dtype="float32"), image_paths

	def __create_faiss_index(self, embeddings, use_cosine=True):
		"""Creates a FAISS index for searching embeddings."""
		dim = embeddings.shape[1]

		if use_cosine:
			# For cosine similarity, we use inner product with normalized vectors
			index = faiss.IndexFlatIP(dim)
			
			# Make a copy before normalizing to avoid modifying the original
			embeddings_copy = embeddings.copy()
			
			# Normalize vectors properly for inner product search
			faiss.normalize_L2(embeddings_copy)
			index.add(embeddings_copy)
		else:
			# For Euclidean distance
			index = faiss.IndexFlatL2(dim)
			index.add(embeddings)
			
		return index

	def __build_faiss(self, use_cosine=True, image_paths=None) -> Tuple[Optional[faiss.Index], List[str]]:
		"""Builds or updates FAISS index with current embeddings."""
		image_vectors, image_paths = self.__load_embeddings(image_paths)
		
		if len(image_vectors) == 0:
			return None, []

		image_vectors = np.array(image_vectors, dtype="float32")

		index = self.__create_faiss_index(image_vectors, use_cosine)
		
		return index, image_paths

	def search_with_text(self, query: str, image_vectors=None, image_paths=None, top_k: int = 5, use_cosine: bool = True, threshold: float = 0.7) -> List[str]:
		inputs = self.processor(text=[query], return_tensors="pt")
		query_embedding = None
		with torch.no_grad():
			query_embedding = self.model.get_text_features(**inputs)

		return self.search(query_embedding, image_vectors, image_paths, top_k, use_cosine, threshold)

	def search_with_image(self, query: str, image_vectors=None, image_paths=None, top_k: int = 5, use_cosine: bool = True, threshold: float = 0.7) -> List[str]:
		inputs = self.processor(images=query, return_tensors="pt")
		query_embedding = None
		with torch.no_grad():
			query_embedding = self.model.get_image_features(**inputs)

		return self.search(query_embedding, image_vectors, image_paths, top_k, use_cosine, threshold)

	def search(
		self, 
		query_embedding, 
		image_vectors=None, 
		image_paths=None, 
		top_k: int = 5, 
		use_cosine: bool = True, 
		threshold: float = 0.3  # Higher threshold for better matches
	) -> List[str]:
		"""Generalized search function for text or image queries."""
		
		# Always search against the full database
		index, full_image_paths = self.__build_faiss(use_cosine, image_paths)
		
		if not index or not full_image_paths:
			return []
		
		# Normalize query embedding
		query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
		query_embedding_np = query_embedding.squeeze().numpy().astype("float32")
		
		# Ensure query embedding is 2D
		if len(query_embedding_np.shape) == 1:
			query_embedding_np = np.expand_dims(query_embedding_np, axis=0)
		
		# Perform FAISS search - search against all indexed images
		# Use a larger k to find more potential matches
		search_k = min(20, len(full_image_paths))
		distances, indices = index.search(query_embedding_np, search_k)
		
		return self.__process_result(distances, indices, full_image_paths, threshold)

	def __process_result(self, distances, indices, image_paths, threshold):
		"""Processes FAISS search results based on similarity threshold."""
		result = []
		similarities = []
		
		# Calculate global stats for normalization if needed
		max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
		min_dist = np.min(distances[0]) if distances[0].size > 0 else 0.0
		range_dist = max_dist - min_dist
		
		for i, idx in enumerate(indices[0]):
			if idx >= len(image_paths):
				continue
				
			raw_similarity = distances[0][i]
			
			# For SigLIP model specifically:
			# Normalize to a more interpretable scale (0-1)
			# This helps set a consistent threshold
			normalized_similarity = (raw_similarity - min_dist) / range_dist if range_dist > 0 else 0
			
			similarities.append((normalized_similarity, raw_similarity, image_paths[idx]))
			print(f"Normalized: {normalized_similarity:.4f}, Raw: {raw_similarity:.4f} :: {image_paths[idx]}")
		
		# Sort by normalized similarity (highest first)
		similarities.sort(key=lambda x: x[0], reverse=True)
		
		# Apply threshold to normalized similarity
		# Adjust this threshold based on your specific needs
		normalized_threshold = 0.05  # Only return results in the top half of similarity range
		
		for norm_sim, raw_sim, path in similarities:
			if norm_sim >= normalized_threshold or raw_sim >= normalized_threshold:
				result.append(path)
		
		print(f"""{{"searched_result":{json.dumps(result)}}}""")
		return result
from .model import load_model
import os
from PIL import Image
import numpy as np
import faiss
import torch
from .db_processor import DatabaseManager
import pickle
from queue import Queue
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

	def __init__(self, paths: List[str] = [], priority_paths: List[str] = [], exclude_paths: List[str] = []):
		load_dotenv()

		self.__process_path(paths, priority_paths, exclude_paths)
		
		# Database connection
		self.conn = DatabaseManager(json.loads(self.FULL_DB_INFO))
		self._newly_indexed = Queue()
		self.model, self.processor = load_model()
		# Start indexing
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
			# Skip if already indexed
			if self.__check_if_indexed(img_path) or not self._is_valid_image_path(img_path):
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
			self._newly_indexed.put((img_path, embedding_np))

			print(f"processed {img_path}")

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

	def __load_embeddings(self) -> Tuple[np.ndarray, List[str]]:
		"""Loads embeddings from the database and includes any newly indexed items."""
		# Get all stored embeddings
		self._newly_indexed = Queue()
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
		
		# Add any newly indexed items not yet in DB
		while not self._newly_indexed.empty():
			path, embedding = self._newly_indexed.get()
			if path not in image_paths and not self._is_excluded(path):
				image_vectors.append(embedding)
				image_paths.append(path)
		
		return np.array(image_vectors, dtype="float32"), image_paths

	def __build_faiss(self) -> Tuple[Optional[faiss.Index], List[str]]:
		"""Builds or updates FAISS index with current embeddings."""
		image_vectors, image_paths = self.__load_embeddings()
		
		if len(image_vectors) == 0:
			return None, []
			
		dim = image_vectors.shape[1]
		index = faiss.IndexFlatL2(dim)
		index.add(image_vectors)
		
		return index, image_paths

	def search_with_text(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
		"""Search images using text query."""
		index, image_paths = self.__build_faiss()
		if not index:
			print("Not Indexed yet.")
			return [], []
			
		inputs = self.processor(text=[query], return_tensors="pt")
		with torch.no_grad():
			text_embedding = self.model.get_text_features(**inputs)
		
		text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
		text_embedding = text_embedding.numpy().astype("float32")
		
		distances, indices = index.search(text_embedding, min(top_k, len(image_paths)))
		result = [image_paths[i] for i in indices[0]]
		print(f"""{{
			"still_indexing": {len(self._newly_indexed) != 0}
			"searched_result":{result}
		}}""")
		return result

	def search_with_image(self, query_image_path: str, top_k: int = 5) -> List[str]:
		"""Search similar images using an image query."""
		index, image_paths = self.__build_faiss()
		if not index:
			return []
			
		image = Image.open(query_image_path).convert("RGB")
		inputs = self.processor(images=image, return_tensors="pt")
		
		with torch.no_grad():
			query_embedding = self.model.get_image_features(**inputs)
			
		query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
		query_np = query_embedding.squeeze().numpy().astype("float32")
		
		distances, indices = index.search(np.array([query_np]), min(top_k, len(image_paths)))
		result = [image_paths[i] for i in indices[0]]
		print(f"""{{
			"still_indexing": {len(self._newly_indexed) != 0}
			"searched_result":{result}
		}}""")
		return result
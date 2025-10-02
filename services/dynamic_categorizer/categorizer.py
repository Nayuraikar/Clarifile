from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

try:
	from sentence_transformers import SentenceTransformer
	from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
	from keybert import KeyBERT
	external_libs_available = True
except Exception:
	external_libs_available = False
	SentenceTransformer = None  # type: ignore
	KMeans = DBSCAN = AgglomerativeClustering = None  # type: ignore
	KeyBERT = None  # type: ignore


class FileType(Enum):
	TEXT = "text"
	IMAGE = "image"
	AUDIO = "audio"
	UNKNOWN = "unknown"


@dataclass
class DocumentResult:
	file_path: str
	content: str


@dataclass
class DiscoveredCategory:
	name: str
	description: str
	keywords: List[str]
	cluster_id: int
	example_documents: List[str]


class DynamicCategorizer:
	def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', clustering_algorithm: str = 'kmeans', n_categories: int = 5, n_clusters: Optional[int] = None, min_cluster_size: int = 5):
		self.embedding_model_name = embedding_model
		self.clustering_algorithm = clustering_algorithm
		self.n_categories = n_categories
		self.n_clusters = n_clusters or n_categories
		self.min_cluster_size = min_cluster_size
		self._embed_model: Optional[SentenceTransformer] = None
		self._kw_model: Optional[KeyBERT] = None

	def _ensure_models(self) -> None:
		if not external_libs_available:
			return
		if self._embed_model is None:
			self._embed_model = SentenceTransformer(f'sentence-transformers/{self.embedding_model_name}')
		if self._kw_model is None:
			self._kw_model = KeyBERT()

	def detect_file_type(self, file_path: Union[str, Path]) -> FileType:
		mime = str(file_path).lower()
		if mime.endswith(('.txt', '.md')):
			return FileType.TEXT
		if mime.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
			return FileType.IMAGE
		if mime.endswith(('.mp3', '.wav', '.m4a', '.aac')):
			return FileType.AUDIO
		return FileType.UNKNOWN

	def _extract_text_from_text_file(self, file_path: Union[str, Path]) -> str:
		try:
			with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
				return f.read()
		except Exception:
			return ""

	def _extract_text_from_image(self, file_path: Union[str, Path]) -> str:
		# Placeholder; reuse project's OCR if needed
		return ""

	def _extract_text_from_audio(self, file_path: Union[str, Path]) -> str:
		# Placeholder; reuse project's ASR if needed
		return ""

	def extract_text(self, file_path: Union[str, Path]) -> str:
		ft = self.detect_file_type(file_path)
		if ft == FileType.TEXT:
			return self._extract_text_from_text_file(file_path)
		if ft == FileType.IMAGE:
			return self._extract_text_from_image(file_path)
		if ft == FileType.AUDIO:
			return self._extract_text_from_audio(file_path)
		return ""

	def discover_categories(self, documents: List[DocumentResult]) -> List[DiscoveredCategory]:
		if not external_libs_available:
			return []
		self._ensure_models()
		if not documents:
			return []

		texts = [d.content for d in documents]
		embs = self._embed_model.encode(texts, show_progress_bar=False)

		if self.clustering_algorithm == 'kmeans':
			labels = KMeans(n_clusters=max(2, min(self.n_clusters, len(documents))), random_state=42, n_init=10).fit_predict(embs)
		elif self.clustering_algorithm == 'dbscan':
			labels = DBSCAN(min_samples=self.min_cluster_size).fit_predict(embs)
		else:
			labels = AgglomerativeClustering(n_clusters=max(2, min(self.n_clusters, len(documents)))).fit_predict(embs)

		cats: List[DiscoveredCategory] = []
		for cluster_id in sorted(set(labels)):
			if cluster_id == -1:
				continue
			cluster_docs = [d for d, c in zip(documents, labels) if c == cluster_id]
			cluster_text = " ".join(d.content for d in cluster_docs)
			keywords: List[str] = []
			if self._kw_model:
				try:
					kws = self._kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
					keywords = [k for k, _ in kws]
				except Exception:
					keywords = []
			cats.append(DiscoveredCategory(
				name=f"Category_{cluster_id}",
				description="Automatically discovered category",
				keywords=keywords,
				cluster_id=cluster_id,
				example_documents=[d.file_path for d in cluster_docs[:3]]
			))

		return cats



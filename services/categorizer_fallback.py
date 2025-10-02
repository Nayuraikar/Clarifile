from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np

try:
	from sentence_transformers import SentenceTransformer
	from sklearn.metrics.pairwise import cosine_similarity
	external_libs_available = True
except Exception:
	# Graceful degradation if optional libs are unavailable at runtime
	external_libs_available = False
	SentenceTransformer = None  # type: ignore
	cosine_similarity = None  # type: ignore


# Predefined categories and indicative keywords
CATEGORY_KEYWORDS: Dict[str, list[str]] = {
	# Education & Learning
	"School Notes": ["school", "class notes", "homework", "syllabus", "teacher"],
	"College/University Courses": ["course", "lecture", "semester", "credits", "assignment"],
	"Tutorials & Study Guides": ["tutorial", "guide", "how to", "study", "steps"],
	"Exam Preparation": ["exam", "mock test", "practice", "questions", "answers"],
	"Summaries & Cheat Sheets": ["summary", "cheat sheet", "key points", "revision"],
	"Flashcards": ["flashcard", "q&a", "term", "definition"],
	"Research Notes": ["research notes", "hypothesis", "method", "findings"],
	"Scientific Papers": ["abstract", "introduction", "methodology", "results", "references"],
	"Thesis Drafts": ["thesis", "dissertation", "proposal", "committee"],
	"Lab Reports": ["lab report", "experiment", "materials", "procedure", "observations"],
	"Lecture Notes": ["lecture", "slides", "ppt", "handout"],
	"Learning Roadmaps": ["roadmap", "learning path", "milestones"],
	"Skill Progress Trackers": ["progress", "tracker", "skills", "milestone"],
	"Language Learning": ["vocabulary", "grammar", "conjugation", "pronunciation"],
	"Book Summaries": ["book summary", "synopsis", "review", "chapters"],
	"Concept Maps": ["concept map", "mind map", "diagram"],

	# Science & Technology
	"Physics": ["mechanics", "thermodynamics", "quantum", "optics"],
	"Chemistry": ["organic", "inorganic", "stoichiometry", "reaction"],
	"Biology": ["cell", "genetics", "evolution", "anatomy"],
	"Computer Science": ["algorithm", "data structure", "complexity", "compiler"],
	"Data Science": ["dataset", "pandas", "visualization", "statistics"],
	"Artificial Intelligence": ["ai", "knowledge graph", "reasoning", "planning"],
	"Machine Learning": ["training", "model", "regression", "classification"],
	"Deep Learning": ["neural network", "cnn", "rnn", "transformer"],
	"Cybersecurity": ["vulnerability", "penetration", "encryption", "threat"],
	"Networking": ["tcp/ip", "routing", "switch", "latency"],
	"Cloud Computing": ["kubernetes", "aws", "gcp", "deployment"],
	"Blockchain": ["smart contract", "ledger", "ethereum", "wallet"],
	"Robotics": ["robot", "actuator", "sensor", "path planning"],
	"Internet of Things (IoT)": ["iot", "mqtt", "edge", "gateway"],
	"Software Engineering": ["design pattern", "testing", "requirements", "architecture"],
	"Web Development": ["html", "css", "javascript", "react"],
	"App Development": ["android", "ios", "flutter", "apk"],
	"DevOps": ["ci/cd", "docker", "pipeline", "monitoring"],
	"Algorithms & DAA": ["divide and conquer", "greedy", "dp", "graph"],
	"Operating Systems": ["process", "thread", "scheduling", "filesystem"],
	"Embedded Systems": ["firmware", "microcontroller", "rtos", "spi"],

	# Engineering & Technical
	"Mechanical Engineering": ["thermodynamics", "fluid mechanics", "machine design"],
	"Electrical Engineering": ["circuit", "power systems", "signal"],
	"Electronics": ["semiconductor", "analog", "digital", "pcb"],
	"Civil Engineering": ["structure", "concrete", "surveying", "load"],
	"Chemical Engineering": ["reactor", "process", "distillation"],
	"Aerospace": ["aerodynamics", "propulsion", "aircraft", "trajectory"],
	"Automotive": ["engine", "transmission", "vehicle dynamics"],
	"Materials Science": ["crystal", "polymer", "alloy", "composite"],
	"Control Systems": ["pid", "state space", "stability"],
	"CAD/Design Docs": ["cad", "solidworks", "autocad", "design"],
	"Project Diagrams": ["diagram", "flowchart", "uml", "block diagram"],
	"Circuit Diagrams": ["schematic", "netlist", "circuit", "wiring"],
	"System Workflows": ["workflow", "pipeline", "process flow"],

	# Business, Finance & Management
	"Business Plans": ["executive summary", "market", "financials", "strategy"],
	"Marketing Strategies": ["campaign", "seo", "brand", "positioning"],
	"Sales Funnels": ["lead", "conversion", "crm", "pipeline"],
	"Project Management": ["gantt", "kanban", "milestone", "risk"],
	"Roadmaps & Timelines": ["roadmap", "timeline", "goals"],
	"OKRs & KPIs": ["okr", "kpi", "metrics", "targets"],
	"SWOT Analysis": ["strengths", "weaknesses", "opportunities", "threats"],
	"Competitive Analysis": ["competitor", "benchmark", "market share"],
	"Meeting Notes": ["meeting", "minutes", "action items"],
	"Startup Ideas": ["mvp", "pitch", "validation"],
	"Product Development": ["spec", "backlog", "release"],
	"Financial Planning": ["budget", "forecast", "cash flow"],
	"Personal Finance": ["expense", "savings", "budget", "ledger"],
	"Investment Strategies": ["portfolio", "equity", "mutual fund"],
	"Stock Market Analysis": ["stock", "candlestick", "analysis", "price"],
	"Real Estate Notes": ["mortgage", "rent", "property", "lease"],
	"Accounting & Bookkeeping": ["invoice", "balance sheet", "ledger", "tax"],
	"Supply Chain": ["logistics", "inventory", "warehouse", "procurement"],
	"E-commerce": ["shop", "cart", "checkout", "catalog"],

	# Professional Work
	"Reports & Documentation": ["report", "documentation", "summary"],
	"SOPs": ["standard operating procedure", "sop", "steps"],
	"Workflows & Pipelines": ["workflow", "pipeline", "automation"],
	"Flowcharts for Processes": ["flowchart", "process", "diagram"],
	"Meeting Minutes": ["minutes", "attendees", "agenda"],
	"Client Notes": ["client", "requirements", "notes"],
	"Interview Notes": ["interview", "candidate", "assessment"],
	"Hiring Processes": ["recruitment", "onboarding", "offer"],
	"Company Policies": ["policy", "compliance", "guidelines"],
	"Product Specs": ["specification", "requirements", "acceptance"],
	"API Documentation": ["api", "endpoint", "request", "response"],

	# Health, Medicine & Psychology
	"Medical Notes": ["diagnosis", "symptoms", "treatment"],
	"Patient Records": ["patient", "record", "prescription", "visit"],
	"Biology Research": ["assay", "protocol", "biomarker"],
	"Anatomy Charts": ["anatomy", "organ", "muscle"],
	"Disease Flowcharts": ["disease", "progression", "stages"],
	"Therapy Plans": ["therapy", "sessions", "cbt"],
	"Nutrition Guides": ["diet", "calorie", "nutrition"],
	"Exercise & Fitness Plans": ["workout", "reps", "routine"],
	"Mental Health Trackers": ["mood", "journal", "stress"],

	# Creative & Artistic
	"Story Planning": ["outline", "plot", "story arc"],
	"Character Profiles": ["character", "backstory", "traits"],
	"Scriptwriting Notes": ["script", "screenplay", "dialogue"],
	"Novel/Book Drafts": ["draft", "chapter", "manuscript"],
	"Poetry Notes": ["poem", "verse", "stanza"],
	"Art & Design Ideas": ["sketch", "color", "composition"],
	"Moodboards": ["moodboard", "palette", "theme"],
	"Comic Planning": ["panel", "storyboard", "comic"],
	"Music Composition Notes": ["composition", "chords", "melody"],

	# Personal & Lifestyle
	"Personal Journal": ["journal", "diary", "reflection"],
	"Daily Logs": ["daily", "log", "notes"],
	"Habit Tracking": ["habit", "tracker", "streak"],
	"Goal Setting": ["goal", "objective", "plan"],
	"Life Planning": ["life plan", "vision", "timeline"],
	"Travel Planning": ["itinerary", "flight", "hotel"],
	"Gratitude Journal": ["gratitude", "thankful", "reflection"],

	# Cooking & Food
	"Recipes": ["ingredients", "instructions", "servings"],
	"Cooking Flowcharts": ["cook", "steps", "flowchart"],
	"Meal Plans": ["meal plan", "breakfast", "lunch", "dinner"],
	"Nutrition Info": ["calories", "protein", "fat", "carbs"],
	"Baking Guides": ["bake", "oven", "temperature"],

	# Nature, Environment & Farming
	"Farming": ["farming", "agriculture", "crop", "harvest"],
	"Gardening Plans": ["garden", "soil", "planting"],
	"Composting Guides": ["compost", "organic", "waste"],
	"Environmental Research": ["environment", "sustainability", "impact"],

	# History, Culture & Social Studies
	"Historical Timelines": ["timeline", "era", "period"],
	"Cultural Studies": ["culture", "anthropology", "tradition"],
	"Political Science": ["policy", "government", "election"],
	"Law & Legal Notes": ["legal", "case", "statute", "contract"],

	# Productivity Tools
	"Task Flows": ["task", "workflow", "steps"],
	"Gantt Charts": ["gantt", "timeline", "schedule"],
	"Kanban Boards": ["kanban", "board", "backlog", "wip"],
	"Sprint Planning": ["sprint", "planning", "retrospective"],
	"Checklists": ["checklist", "todo", "tasks"],

	# Tech Architecture (Expanded)
	"AI Architectures": ["transformer", "attention", "encoder", "decoder"],
	"Data Pipeline Flows": ["etl", "elt", "ingestion", "pipeline"],
	"Database Schemas": ["schema", "er diagram", "tables"],
	"API Call Flows": ["api", "request", "response", "auth"],
	"Microservices Architecture": ["microservice", "service", "mesh", "grpc"],
	"DevOps CI/CD Pipelines": ["ci", "cd", "pipeline", "deployment"],
	"System Design Diagrams": ["system design", "scalability", "availability"],
	"Cloud Deployment Guides": ["terraform", "helm", "manifest"],

	# Misc / Meta
	"Brain Dumps": ["brain dump", "scratch", "notes"],
	"Whiteboard Ideas": ["whiteboard", "sketch", "idea"],
	"Mind Maps": ["mind map", "nodes", "branches"],
	"FAQ Sheets": ["faq", "questions", "answers"],
}


def _clean_text(text: str) -> str:
	if not text:
		return ""
	# Lowercase and remove excessive whitespace/non-text noise
	text = re.sub(r"\s+", " ", text).strip().lower()
	# Keep words, numbers, common punctuation
	text = re.sub(r"[^a-z0-9\s.,:;()\-_'\"]+", " ", text)
	return text


def keyword_based_category(text: str, keywords_map: Dict[str, list[str]] = CATEGORY_KEYWORDS) -> Tuple[Optional[str], dict]:
	t = text.lower()
	for category, keywords in keywords_map.items():
		for kw in keywords:
			if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t):
				return category, {"method": "keyword", "matched_keyword": kw}
	return None, {}


_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_CATEGORY_PROTOTYPES: Dict[str, np.ndarray] = {}


def _get_model() -> Optional[SentenceTransformer]:
	global _EMBEDDING_MODEL
	if not external_libs_available:
		return None
	if _EMBEDDING_MODEL is None:
		# Keep model name aligned with the rest of the project
		_EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	return _EMBEDDING_MODEL


def _init_prototypes() -> None:
	if not external_libs_available:
		return
	global _CATEGORY_PROTOTYPES
	if _CATEGORY_PROTOTYPES:
		return

	model = _get_model()
	if model is None:
		return

	# Create simple textual prototypes per category
	prototype_texts = {
		cat: f"This document is about {cat.lower()}. Keywords: " + ", ".join(words)
		for cat, words in CATEGORY_KEYWORDS.items()
	}
	labels = list(prototype_texts.keys())
	embs = model.encode([prototype_texts[c] for c in labels], convert_to_numpy=True, normalize_embeddings=True)
	for i, label in enumerate(labels):
		_CATEGORY_PROTOTYPES[label] = embs[i]


def embedding_based_category(text: str, threshold: float = 0.45) -> Tuple[Optional[str], dict]:
	if not external_libs_available:
		return None, {"error": "embeddings_unavailable"}

	_init_prototypes()
	if not _CATEGORY_PROTOTYPES:
		return None, {"error": "no_prototypes"}

	model = _get_model()
	if model is None:
		return None, {"error": "model_unavailable"}

	text_clean = _clean_text(text)
	if not text_clean:
		return None, {"error": "empty_text"}

	emb = model.encode([text_clean], convert_to_numpy=True, normalize_embeddings=True)
	cats = list(_CATEGORY_PROTOTYPES.keys())
	proto_embs = np.stack([_CATEGORY_PROTOTYPES[c] for c in cats])
	sims = np.dot(emb, proto_embs.T)[0]
	score_map = {cats[i]: float(sims[i]) for i in range(len(cats))}
	best_idx = int(np.argmax(sims))
	best_cat = cats[best_idx]
	best_score = float(sims[best_idx])
	debug = {"method": "embedding", "scores": score_map, "top": (best_cat, best_score)}
	return (best_cat, debug) if best_score >= threshold else (None, debug)


def categorize_with_fallback(text: str, existing_categorizer=None) -> Tuple[Optional[str], Dict]:
	"""Hybrid categorization with keyword → embedding → external fallback.

	Returns (category_name | None, debug_info)
	"""
	text_clean = _clean_text(text)
	if not text_clean:
		return None, {"error": "empty_text"}

	# 1) Keyword pass
	kw_cat, kw_dbg = keyword_based_category(text_clean)
	if kw_cat:
		return kw_cat, kw_dbg

	# 2) Embedding pass (threshold adapts to text length)
	word_count = len(text_clean.split())
	threshold = 0.45 if word_count >= 6 else 0.60
	emb_cat, emb_dbg = embedding_based_category(text_clean, threshold=threshold)
	if emb_cat:
		return emb_cat, emb_dbg

	# 3) External fallback if provided
	if existing_categorizer:
		try:
			return existing_categorizer(text_clean), {"method": "existing_categorizer"}
		except Exception as exc:
			return None, {"error": "existing_categorizer_failed", "exc": str(exc)}

	return None, {"method": "none"}



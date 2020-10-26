from typing import List, Optional, TypedDict, Literal, Any, Dict
from datetime import datetime
import networkx as nx


LANGUAGES = Literal['en', 'fr', 'multi']
CODE_TO_LANG = {"en": "english", "fr": "french"}
LANG_TO_CODE = {v: k for k, v in CODE_TO_LANG.items()}

CHUNK_SIZE = 1000  # number of word characters
NB_KEYWORDS = 6  # number of keywords to keep per chunk
EMBED_DIM = 768
MODEL_NAMES: Dict[LANGUAGES, str] = {
        "multi": "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",  # noqa E501
        "fr": 'None',  # noqa E501
        "en": 'None'  # noqa E501
    }


class RawEntry(TypedDict):
    type: Literal['page', 'section', 'pdf']
    path: str
    title: str
    content: str
    language: Literal['fr', 'en']
    first_seen_date: Any
    children: Optional[List['RawEntry']]


class Link(TypedDict):
    path: str
    start: int
    name: str


class MetaData(TypedDict, total=False):
    links: List[Link]
    keywords: List[str]
    title_embedding: List[float]
    content_embedding: List[float]


class Chunk(RawEntry, MetaData):
    chunk_hash: str
    original_hash: str
    chunk_start: int


class Indexes(TypedDict):
    graph: nx.DiGraph
    db: Any
    faiss: Any
    faiss_idx: List


class Models(TypedDict, total=False):
    embedder: Dict[LANGUAGES, Any]
    keyworder: Dict[LANGUAGES, Any]
    answerer: Dict[LANGUAGES, Any]
    processor: Dict[LANGUAGES, Any]
    dim_reducer: Any

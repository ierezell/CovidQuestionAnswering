"""
This file is made to have configuration variables at the same place
as well with the types
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from elasticsearch import Elasticsearch

from embedders.answerer import Answerer
from embedders.embedders import Embedder

LANGUAGES = Literal['en', 'fr', 'multi', 'qa_fr', 'qa_en', 'qa_multi']
CODE_TO_LANG = {"en": "english", "fr": "french"}
LANG_TO_CODE = {v: k for k, v in CODE_TO_LANG.items()}

CHUNK_SIZE = 1000  # number of word per paragraph
NB_KEYWORDS = 6  # number of keywords to keep per chunk
EMBED_DIM = 768
MODEL_NAMES: Dict[LANGUAGES, str] = {
        "fr": "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",  # noqa E501
        "multi": 'facebook/mbart-large-cc25',
        "en": 'None',
        "qa_en": 'None',
        "qa_fr": 'etalab-ia/camembert-base-squadFR-fquad-piaf',
        "qa_multi": 'deepset/xlm-roberta-large-squad2',
    }

SPACY_MODEL_NAMES: Dict[LANGUAGES, str] = {"multi": "xx_ent_wiki_sm",
                                           "en": "en_core_web_sm",
                                           "fr": "fr_dep_news_trf"}


class RawEntry(TypedDict):
    type: Literal['page', 'section', 'pdf']
    path: str
    title: str
    content: str
    language: Literal['fr', 'en']
    first_seen_date: str
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
    page_content: str
    first_seen_date: datetime
    parent_content_embedding: List[float]
    parent_title_embedding: List[float]
    parent_content: str
    parent_title: str


class Indexes(TypedDict):
    db: Elasticsearch


class Models(TypedDict, total=False):
    embedder: Dict[LANGUAGES, Embedder]
    answerer: Dict[LANGUAGES, Answerer]
    processor: Dict[LANGUAGES, Any]


class Answer(TypedDict):
    score: float
    content: str
    answer: str
    start: int
    end: int
    elected: Literal['qa', 'kw', 'n/a']

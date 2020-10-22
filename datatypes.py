from typing import List, Optional, TypedDict, Literal
from datetime import date


class RawEntry(TypedDict):
    type: Literal['page', 'section', 'pdf']
    path: str
    title: str
    content: str
    language: Literal['fr', 'en']
    first_seen_date: date
    children: Optional[List['RawEntry']]


class Link(TypedDict):
    path: str
    start: int
    name: str


class MetaData(TypedDict, total=False):
    links: List[Link]
    keywords: List[str]
    title_embedding: List[float]
    chunk_embedding: List[float]


class Chunk(RawEntry):
    chunk_hash: str
    original_hash: str
    chunk_start: int


class Entry(TypedDict):
    chunk: Chunk
    metadatas: MetaData


CODE_TO_LANG = {"en": "english", "fr": "french"}
LANG_TO_CODE = {v: k for k, v in CODE_TO_LANG.items()}

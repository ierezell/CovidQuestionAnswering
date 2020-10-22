import hashlib
from typing import Tuple, List
from utils import hash_text
from datatypes import Chunk, Link, RawEntry, MetaData, CODE_TO_LANG
from rake_nltk import Rake


def create_metadata(chunk: Chunk, links: List[Link]) -> MetaData:
    new_metadatas: MetaData = {}

    chunk_links = []
    chunk_start = chunk['chunk_start']
    chunk_end = chunk_start + len(chunk['content'])

    for link in links:
        if chunk_start < link['start'] < chunk_end:
            chunk_links.append(link)

    new_metadatas['keywords'] = extract_keywords(chunk)
    new_metadatas['links'] = chunk_links

    return new_metadatas


def extract_keywords(entry) -> List[str]:
    r = Rake(language=CODE_TO_LANG[entry['language']])
    r.extract_keywords_from_text(entry['content'])
    keywords: List[str] = r.get_ranked_phrases()
    return keywords[:6]


def embed_entry(entry: RawEntry) -> List[float]:
    content_embedding = embed(entry['content'])
    title_embedding = embed(entry['title'])
    return {}

from typing import Dict, List, Tuple

from datatypes import (LANGUAGES, NB_KEYWORDS, Chunk, Link, MetaData, Models)
from embedders.embedders import Embedder


def create_metadata(chunk: Chunk, links: List[Link], models: Models
                    ) -> MetaData:
    new_metadatas: MetaData = {}

    chunk_links = []
    chunk_start = chunk['chunk_start']
    chunk_end = chunk_start + len(chunk['content'])

    for link in links:
        if chunk_start < link['start'] < chunk_end:
            chunk_links.append(link)

    # new_metadatas['keywords'] = extract_keywords(chunk, models['keyworder'])
    new_metadatas['links'] = chunk_links

    embeddings = embed_chunk(chunk, models['embedder'])
    new_metadatas['title_embedding'] = embeddings[0]
    new_metadatas['content_embedding'] = embeddings[1]

    return new_metadatas


def extract_keywords(chunk, keyworder) -> List[str]:
    r = keyworder.get(chunk['language'], None)
    if r is None:
        return []
    r.extract_keywords_from_text(chunk['content'])
    keywords: List[str] = r.get_ranked_phrases()
    return keywords[:NB_KEYWORDS]


def embed_chunk(chunk: Chunk, embedders: Dict[LANGUAGES, Embedder]
                ) -> Tuple[List[float], List[float]]:
    embedder = embedders.get(chunk['language'], embedders['fr'])

    content_embedding = embedder.embed(chunk['content'])
    title_embedding = embedder.embed(chunk['title'])

    return (title_embedding, content_embedding)

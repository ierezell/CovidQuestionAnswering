from typing import Dict, List, Tuple

from datatypes import LANGUAGES, Chunk, Link, MetaData, Models
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

    new_metadatas['links'] = chunk_links

    embeddings = embed_chunk(chunk, models['embedder'])
    new_metadatas['title_embedding'] = embeddings[0]
    new_metadatas['content_embedding'] = embeddings[1]

    return new_metadatas


def embed_chunk(chunk: Chunk, embedders: Dict[LANGUAGES, Embedder]
                ) -> Tuple[List[float], List[float]]:

    embedder = embedders.get(chunk['language'], embedders['fr'])

    content_embedding = embedder.embed(chunk['content'], method='sentence')
    title_embedding = embedder.embed(chunk['title'], method='sentence')

    return (title_embedding, content_embedding)

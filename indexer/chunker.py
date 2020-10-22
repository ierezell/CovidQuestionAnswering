from copy import deepcopy
from utils import hash_text
from datatypes import Chunk, MetaData, RawEntry, Link
from typing import List
from .metabuilder import create_metadata
import spacy

CHUNK_SIZE = 1000  # number of word characters
nlp = spacy.load("xx_ent_wiki_sm")
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def chunker(entry: RawEntry) -> List[Chunk]:
    chunks: List[Chunk] = []

    chunk_content: List[str] = []
    chunk_len = 0
    chunk_offset = 0
    splitted_entry = split(entry['content'])

    for paragraph in splitted_entry:
        paragraph_len = len(paragraph)

        if chunk_len + paragraph_len < CHUNK_SIZE:
            chunk_content.append(paragraph)
            chunk_len += paragraph_len

        else:
            chunks.append(create_chunk(entry, "".join(chunk_content),
                                       chunk_offset))

            chunk_content = []
            chunk_offset += chunk_len
            chunk_len = 0

    if len(chunks) < 1:
        chunks.append(create_chunk(entry, "".join(entry['content']), 0))

    return chunks


def create_chunk(entry: RawEntry, content: str, chunk_len: int) -> Chunk:
    chunk: Chunk = deepcopy(entry)  # type: ignore
    chunk['content'] = content
    chunk['chunk_hash'] = hash_text(content)
    chunk['original_hash'] = hash_text(entry['content'])
    chunk['chunk_start'] = chunk_len
    return chunk


def split(text: str) -> List[str]:
    chunks: List[str] = []
    for piece in nlp(text).sents:
        chunks.append(piece.text)
    return chunks
    # chunked_text = text.split('\n')
    # for chunk in chunked_text:
    #     if len(chunk) > CHUNK_SIZE:
    #         # Resplit with other method

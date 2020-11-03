from copy import deepcopy
from typing import List

from config import CHUNK_SIZE
from datatypes import Chunk, Models, RawEntry
from utils import hash_text


def chunker(entry: RawEntry, models: Models) -> List[Chunk]:
    """Create chunks by first splitting the content in sentences and then
    adding each sentence until reaching the desired size. If adding the
    sentence make the chunk bigger than the allowed size, start a new chunk.
    """
    chunks: List[Chunk] = []

    chunk_content: List[str] = []
    chunk_len = 0
    chunk_offset = 0

    splitted_entry = split(entry['content'],
                           models['processor'].get(entry['language'], 'multi'))

    for paragraph in splitted_entry:
        paragraph_len = len(paragraph)

        if chunk_len + paragraph_len < CHUNK_SIZE:
            chunk_content.append(paragraph)
            chunk_len += paragraph_len

        else:
            chunks.append(create_chunk(entry, "".join(chunk_content),
                                       chunk_offset))

            chunk_content = [paragraph]
            chunk_offset += chunk_len
            chunk_len = paragraph_len

    if chunk_content:
        chunks.append(create_chunk(entry, "".join(chunk_content),
                                   chunk_offset))

    if len(chunks) < 1:
        chunks.append(create_chunk(entry, "".join(entry['content']), 0))

    return chunks


def create_chunk(entry: RawEntry, content: str, chunk_len: int) -> Chunk:
    """Add content and it's informations to the chunk"""
    # Copy an entry which is not of type chunk but then add all the fields
    chunk: Chunk = deepcopy(entry)  # type: ignore
    chunk['content'] = content
    chunk['chunk_hash'] = hash_text(content)
    chunk['original_hash'] = hash_text(entry['content'])
    chunk['chunk_start'] = chunk_len
    return chunk


def split(text: str, processor) -> List[str]:
    """Split a chunk of text to make chunk consistant (not splitting them in
    the middle of a word or a sentence)"""
    chunks: List[str] = []
    for piece in processor(text).sents:
        chunks.append(piece.text)
    return chunks

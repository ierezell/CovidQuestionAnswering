import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import elasticsearch
import faiss
import networkx as nx
import numpy as np
import spacy
from datatypes import (EMBED_DIM, LANGUAGES, SPACY_MODEL_NAMES, Chunk, Indexes,
                       Models, RawEntry)
from embedders.embedders import Embedder
from embedders.answerer import Answerer
from rake_nltk import Rake
from sklearn.decomposition import PCA
# from umap import UMAP
# from sklearn.manifold import TSNE
from utils import remove_links, sanitize_text

from .chunker import chunker
from .metabuilder import create_metadata


def recurse_add(raw_entry: RawEntry, level: int,
                indexes: Indexes, models: Models,
                parents: List[Chunk] = []):
    # Clean entry
    raw_entry['content'], links = remove_links(raw_entry['content'])
    raw_entry['title'] = sanitize_text(raw_entry['title'])
    raw_entry['content'] = sanitize_text(raw_entry['content'])
    raw_entry['first_seen_date'] = datetime.strptime(
        raw_entry['first_seen_date'], "%m/%d/%Y")

    # Chunk entry content
    current_parents: List[Chunk] = []
    for chunk in chunker(raw_entry, models):
        chunk.pop('children', None)

        metadatas = create_metadata(chunk, links, models)
        chunk.update(metadatas)  # type: ignore

        current_parents.append(chunk)
        if chunk['content'] != chunk['title']:
            add_chunk_to_indexes(chunk, indexes, parents, level)

    for child in raw_entry.get('children', []):
        recurse_add(child, parents=current_parents,
                    indexes=indexes, models=models, level=level+1)


def add_chunk_to_indexes(chunk: Chunk, indexes: Indexes, parents: List[Chunk],
                         level: int):

    if parents:
        chunk['parent_content_embedding'] = np.sum(
            np.array([p['content_embedding'] for p in parents]), axis=0
        ).tolist()

        chunk['parent_title_embedding'] = np.sum(
            np.array([p['title_embedding'] for p in parents]), axis=0
        ).tolist()

        chunk['parent_content'] = " ".join([p['content'] for p in parents])
        chunk['parent_title'] = parents[0]['title']

    else:
        chunk['parent_content_embedding'] = np.full(
            EMBED_DIM, np.finfo(float).eps).tolist()
        chunk['parent_title_embedding'] = np.full(
            EMBED_DIM, np.finfo(float).eps).tolist()

    indexes['db'].index(index='gouv', id=chunk['chunk_hash'], body=chunk)

    indexes['faiss'].add(np.array([chunk['content_embedding']],
                                  dtype=np.float32))

    indexes['faiss_idx'].append(chunk['chunk_hash'])

    indexes['graph'].add_node(chunk['chunk_hash'], **chunk)

    for dad in parents:
        if dad['path'] in chunk['path']:
            indexes['graph'].add_edge(dad['chunk_hash'],
                                      chunk['chunk_hash'])


def create_graph_links(graph: nx.DiGraph):
    all_path = []
    all_links = []
    for node_hash, node_data in graph.nodes(data=True):
        all_path.append(node_data['path'])

        for other_node_hash, other_node_data in graph.nodes(data=True):
            if node_hash == other_node_hash:
                continue

            for link in node_data['links']:
                all_links.append(link['path'])
                if other_node_data['path'] == link['path']:
                    graph.add_edge(node_hash, other_node_hash)


def create_models() -> Models:

    keyworder: Dict[LANGUAGES, Any] = {'en': Rake(language='english'),
                                       'fr': Rake(language='french')}

    dim_reducer = PCA(n_components=3)

    processor: Dict[LANGUAGES, Any] = {}

    plop: Dict[LANGUAGES, str] = {"fr": "fr_core_news_lg"}
    for lang, model in plop.items():
        nlp = spacy.load(model)

        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)

        processor.update({lang: nlp})

    embedder: Dict[LANGUAGES, Any] = {"fr": Embedder(processor['fr'], 'fr')}

    answerer: Dict[LANGUAGES, Any] = {"fr": Answerer('qa_fr')}

    models: Models = {"embedder": embedder, "keyworder": keyworder,
                      "answerer": answerer, "processor": processor,
                      "dim_reducer": dim_reducer}

    return models


def create_indexes() -> Tuple[Indexes, bool]:
    need_creation = False

    es = elasticsearch.Elasticsearch()

    indexes: Indexes = {"graph": nx.DiGraph(), "db": es, "faiss_idx": [],
                        "faiss": faiss.IndexFlatL2(EMBED_DIM)}

    if not es.indices.exists(index='gouv'):
        index_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            # "filter": ["stop"],
                            # "type": "standard",
                            "type": "french",
                            "language": "french",
                            "stopwords": "_french_"
                        },
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIM
                    },
                    "content_embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIM
                    },
                    "parent_content_embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIM
                    },
                    "parent_title_embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIM
                    },
                    "keywords": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "parent_content": {"type": "text"},
                    "parent_title": {"type": "text"},
                    "first_seen_date": {"type": "date"},
                }
            }
        }

        es.indices.create(index='gouv', ignore=400, body=index_body)
        need_creation = True

    return indexes, need_creation


def compute_pca(indexes: Indexes, models: Models):
    vectors = []
    for i in range(indexes['faiss'].ntotal):
        vectors.append(indexes['faiss'].reconstruct(i))
    models['dim_reducer'].fit(vectors)


def load(indexes: Indexes, models: Models):
    if os.path.exists('./indexer/models/graph.json'):
        with open('./indexer/models/graph.json', 'r')as file:
            graph_data = json.load(file)
        indexes['graph'] = nx.cytoscape_graph(graph_data)

    if os.path.exists('./indexer/models/faiss'):
        indexes['faiss'] = faiss.read_index('./indexer/models/faiss')

        with open('./indexer/models/faiss_idx', 'rb')as file:
            indexes['faiss_idx'] = pickle.load(file)

    if os.path.exists('./indexer/models/pca'):
        with open('./indexer/models/pca', 'rb') as file:
            models['dim_reducer'] = pickle.load(file)


def save(indexes: Indexes, models: Models):
    c_graph = nx.cytoscape_data(indexes['graph'])

    for node in c_graph['elements']['nodes']:
        # node['data']['keywords'] = " ".join(node['data']["keywords"])
        node['data']['links'] = " ".join([lk['path'] for lk in
                                          node['data']["links"]])
        node['data']['first_seen_date'] = node['data'][
            "first_seen_date"].strftime("%m/%d/%Y")

    with open('./indexer/models/graph.json', 'w') as file:
        json.dump(c_graph, file)

    faiss.write_index(indexes['faiss'], './indexer/models/faiss')

    with open('./indexer/models/faiss_idx', 'wb')as file:
        pickle.dump(indexes['faiss_idx'], file)

    with open('./indexer/models/pca', 'wb') as file:
        pickle.dump(models['dim_reducer'], file)


def preprocess(raw_entry: RawEntry) -> Tuple[Indexes, Models]:
    start = time.time()
    models = create_models()
    print(f"Models created in {time.time()-start}s")

    start = time.time()
    indexes, need_creation = create_indexes()
    print(f"Indexes created in {time.time()-start}s")

    if need_creation:
        print("Doing all : need creation")
        start = time.time()
        recurse_add(raw_entry, parents=[], level=0,
                    indexes=indexes, models=models)
        print(f"Entries added in {time.time()-start}s")

        create_graph_links(indexes['graph'])
        print("Created all links")

        start = time.time()
        compute_pca(indexes, models)
        print(f"Pca computed in {time.time()-start}s")
        save(indexes, models)
    else:
        load(indexes, models)

    return indexes, models

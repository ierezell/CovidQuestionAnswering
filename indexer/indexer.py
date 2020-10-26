import os
import json
from typing import Dict, List, Any, Tuple

import networkx as nx
import plotly.graph_objects as go
from datatypes import Chunk, Models, RawEntry, Indexes, LANGUAGES, EMBED_DIM
from utils import remove_links, sanitize_text
import elasticsearch
from embedders.embedders import Embedder
import spacy
from rake_nltk import Rake
from .chunker import chunker
from .metabuilder import create_metadata
import faiss
import numpy as np
from datetime import datetime
import time
# from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def recurse_add(raw_entry: RawEntry, indexes: Indexes, models: Models,
                parents: List[Chunk] = []):
    # Clean entry
    raw_entry['content'], links = remove_links(raw_entry['content'])
    raw_entry['title'] = sanitize_text(raw_entry['title'])
    raw_entry['first_seen_date'] = datetime.strptime(
        raw_entry['first_seen_date'], "%m/%d/%Y")

    # Chunk entry content
    current_parents: List[Chunk] = []
    for chunk in chunker(raw_entry, models):
        metadatas = create_metadata(chunk, links, models)
        chunk.pop('children', None)
        chunk.update(metadatas)  # type: ignore
        current_parents.append(chunk)

        # Add to indexes
        add_chunk_to_indexes(chunk, indexes, parents)

    for child in raw_entry.get('children', []):
        recurse_add(child, parents=current_parents,
                    indexes=indexes, models=models)


def add_chunk_to_indexes(chunk: Chunk, indexes: Indexes, parents: List[Chunk]):
    indexes['graph'].add_node(chunk['chunk_hash'], **chunk)
    indexes['db'].index(index='gouv', id=chunk['chunk_hash'], body=chunk)
    # add vectors to the index
    indexes['faiss'].add(np.array([chunk['content_embedding']],
                                  dtype=np.float32))
    indexes['faiss_idx'].append(chunk['chunk_hash'])

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

    print("Intersect", set(all_path).intersection(set(all_links)))


def create_models() -> Models:
    embedder: Dict[LANGUAGES, Any] = {"multi": Embedder('multi')}  # noqa E501

    answerer: Dict[LANGUAGES, Any] = {}

    keyworder: Dict[LANGUAGES, Any] = {'en': Rake(language='english'),
                                       'fr': Rake(language='french')}

    processor: Dict[LANGUAGES, Any] = {}

    spacy_models: Dict[LANGUAGES, Any] = {"multi": "xx_ent_wiki_sm",
                                          "en": "en_core_web_sm",
                                          "fr": "fr_core_news_sm"}

    dim_reducer = PCA(n_components=3)

    for lang, model in spacy_models.items():
        nlp = spacy.load(model)
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)

        processor.update({lang: nlp})

    models: Models = {"embedder": embedder, "keyworder": keyworder,
                      "answerer": answerer, "processor": processor,
                      "dim_reducer": dim_reducer}

    return models


def create_indexes() -> Tuple[Indexes, bool]:
    need_creation = False

    es = elasticsearch.Elasticsearch()

    indexes: Indexes = {
        "graph": nx.DiGraph(), "db": es,
        "faiss": faiss.IndexFlatL2(EMBED_DIM),
        "faiss_idx": []
    }

    if not es.indices.exists(index='gouv'):
        index_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "filter": ["stop"],
                            "type": "standard",
                            "language": "french",
                            "stopwords": "_french_"
                        },
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title_embedding": {"type": "dense_vector",
                                        "dims": EMBED_DIM},
                    "content_embedding": {"type": "dense_vector",
                                          "dims": EMBED_DIM},
                    "keywords": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "first_seen_date": {"type": "date"},
                }
            }
        }

        es.indices.create(index='gouv', ignore=400, body=index_body)
        need_creation = True

    if os.path.exists('./graph.json') and not need_creation:
        with open('./graph.json', 'r')as file:
            graph_data = json.load(file)
        indexes['graph'] = nx.cytoscape_graph(graph_data)

    return indexes, need_creation


def compute_pca(indexes: Indexes, models: Models):
    vectors = []
    for i in range(indexes['faiss'].ntotal):
        vectors.append(indexes['faiss'].reconstruct(i))
    models['dim_reducer'].fit(vectors)


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
        recurse_add(raw_entry, parents=[], indexes=indexes, models=models)
        print(f"Entries added in {time.time()-start}s")

        create_graph_links(indexes['graph'])
        print("Created all links")

        dump_cytoscape(indexes['graph'])
        print("Exported")

        start = time.time()
        compute_pca(indexes, models)
        print(f"Pca computed in {time.time()-start}s")
    # visualise_graph(graph)

    return indexes, models


def dump_cytoscape(graph: nx.Graph):
    c_graph = nx.cytoscape_data(graph)

    for node in c_graph['elements']['nodes']:
        node['data']['keywords'] = " ".join(node['data']["keywords"])
        node['data']['links'] = " ".join([lk['path'] for lk in
                                          node['data']["links"]])
        node['data']['first_seen_date'] = node['data'][
            "first_seen_date"].strftime("%m/%d/%Y")

    with open('./graph.json', 'w') as file:
        json.dump(c_graph, file)


def visualise_graph(graph: nx.Graph):
    pos = nx.spring_layout(graph, k=10)
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, hoverinfo='none', mode='lines',
                            line=dict(width=2, color='#888'))

    node_x = []
    node_y = []
    node_text = []
    for node_hash, node_data in graph.nodes(data=True):
        x, y = pos[node_hash]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            node_data['chunk_hash'] + "<br>" +
            node_data['original_hash'] + "<br>" + node_data['path'] + "<br>" +
            "<br>".join([lk['path']
                         for lk in node_data['metadatas']['links']]) +
            "<br>" + node_data['title'] + "<br>" +
            node_data['content'][:30] + "<br>"
        )

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                            hoverinfo='text', text=node_text,
                            marker=dict(showscale=True, colorscale='YlGnBu',
                                        reversescale=True, color=[], size=10,
                                        line_width=2))

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Gouv graph', hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    for idx in range(0, len(edge_x), 3):
        fig.add_annotation(dict(
            x=edge_x[idx+1], y=edge_y[idx + 1], ax=edge_x[idx], ay=edge_y[idx],
            xref='x', yref='y', axref='x', ayref='y', arrowcolor='black',
            showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=2, text=''
        ))

    fig.show()

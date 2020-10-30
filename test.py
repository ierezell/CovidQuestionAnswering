"""
This file is made to check the pipeline end to end
in one shot just to verify for code mistakes
"""
import json
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from datatypes import Indexes, Models, RawEntry
from indexer.indexer import preprocess
from qa.refinder import answer_question
from qa.retriever import retrieve_docs


def ask_question(indexes: Indexes, models: Models, question: str):
    return retrieve_docs(indexes, models, question)


def preprocess_data() -> Tuple[Indexes, Models]:
    with open('datasets/gouv/DOCUMENTS.json', 'r') as file:
        # with open('datasets/gouv/dummy.json', 'r') as file:
        data: RawEntry = json.load(file)
    indexes, models = preprocess(data)

    return indexes, models


def clear_indexes(indexes: Indexes):
    for index in indexes['db'].indices.get('*'):
        indexes['db'].indices.delete(index=index, ignore=[400, 404])


def get_all_docs_vectors(indexes, support_hash) -> List[List[float]]:
    all_docs: List[List[float]] = []
    for i in range(indexes['faiss'].ntotal):
        if indexes['faiss_idx'][i] in support_hash:
            continue
        else:
            all_docs.append(indexes['faiss'].reconstruct(i))
    return all_docs


def get_fig_embeddings(indexes, models, support, question_embed):
    support_embed = []
    support_hash = []
    for sup in support:
        sup.pop('type', None)
        sup.pop('first_seen_date', None)
        sup.pop('language', None)
        sup.pop('title_embedding', None)
        support_embed.append(sup.pop('content_embedding', None))
        support_hash.append(sup.pop('content_embedding', None))

    all_docs = get_all_docs_vectors(indexes, support_hash)

    support_PCA = models['dim_reducer'].transform(support_embed)
    docs_PCA = models['dim_reducer'].transform(all_docs)
    question_PCA = models['dim_reducer'].transform(
        np.array(question_embed).reshape(1, -1))

    all_data = np.concatenate((docs_PCA, support_PCA,  question_PCA), axis=0)

    fig = go.Figure(data=[go.Scatter3d(
        x=[plop[0] for plop in all_data], y=[plop[1] for plop in all_data],
        z=[plop[2] for plop in all_data], mode='markers', marker=dict(
            opacity=0.8,
            color=['green']*len(docs_PCA) + ['blue'] *
            len(support_PCA) + ['red']))])

    return fig


indexes, models = preprocess_data()
user_input = "Les femmes enceintes sont-elles plus a risque ?"
question_embed, support, max_score, hits = ask_question(indexes, models,
                                                        user_input)
answer = answer_question(question_embed, user_input, support, models)
clear_indexes(indexes)

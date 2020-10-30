import json
from typing import List, Tuple

import elasticsearch
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from datatypes import Indexes, Models, RawEntry
from indexer.indexer import preprocess
from qa.refinder import answer_question
from qa.retriever import retrieve_docs
import logging

st.title('Gouv bot')
logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)


# @st.cache(hash_funcs={elasticsearch.Elasticsearch: id,
#   "builtins.SwigPyObject": id,
#   "preshed.maps.PreshMap": id,
#   "cymem.cymem.Pool": id})
def ask_question(indexes: Indexes, models: Models, question: str):
    return retrieve_docs(indexes, models, question)


@st.cache(allow_output_mutation=True)
def preprocess_data() -> Tuple[Indexes, Models]:
    with open('datasets/gouv/DOCUMENTS.json', 'r') as file:
        # with open('datasets/gouv/dummy.json', 'r') as file:
        data: RawEntry = json.load(file)
    indexes, models = preprocess(data)

    return indexes, models


def clear_indexes(indexes: Indexes):
    for index in indexes['db'].indices.get('*'):
        indexes['db'].indices.delete(index=index, ignore=[400, 404])


# @st.cache(hash_funcs={elasticsearch.Elasticsearch: id,
#                       "builtins.SwigPyObject": id,
#                       "preshed.maps.PreshMap": id,
#                       "cymem.cymem.Pool": id})
def get_all_docs_vectors(indexes: Indexes) -> List[Tuple[List[float], str]]:
    all_docs: List[Tuple[List[float], str]] = []
    for i in range(indexes['faiss'].ntotal):
        all_docs.append((indexes['faiss'].reconstruct(i),
                         indexes['faiss_idx'][i]))
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

    all_docs = [doc_emb for doc_emb, doc_hash in get_all_docs_vectors(indexes)
                if doc_hash not in support_hash]

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


indexes = None  # type: ignore
user_input = st.text_input(
    "Posez une question", "Les femmes enceintes sont-elles plus a risque ?")

if st.button('ask'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.success('Done!')
    question_embed, supports, max_score, hits = ask_question(indexes, models,
                                                             user_input)

    answer = answer_question(question_embed, user_input, supports, models)

    st.write(answer)

    with st.beta_expander("See the supports documents"):
        st.write("Max Score")
        st.write(max_score)
        st.write("Hits")
        st.write(hits)
        st.write("Docs")
        st.write([
            {
                k: v for k, v in sup.items()
                if k in ['title', 'content', 'parent_title', 'path']
            }
            for sup in supports
        ])

    with st.beta_expander("See embedded documents"):
        st.plotly_chart(get_fig_embeddings(indexes, models,
                                           supports, question_embed))


if st.button('clear database'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.write("Clearing")
    clear_indexes(indexes)
    st.success('Done!')

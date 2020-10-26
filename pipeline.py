import json
from retriever.retriever import retrieve_docs

import streamlit as st
from typing import Tuple
from datatypes import Indexes, RawEntry, Models
from indexer.indexer import preprocess
import elasticsearch
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# "https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html#vector-functions"
st.title('Gouv bot')


# @st.cache(hash_funcs={elasticsearch.Elasticsearch: id,
#   builtins.SwigPyObject: id})
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


indexes = None  # type: ignore
user_input = st.text_input(
    "Posez une question", "Les femmes enceintes sont-elles plus a risque ?")

if st.button('ask'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.success('Done!')
    question_embed, support, max_score, hits = ask_question(indexes,
                                                            models, user_input)
    support_embed = []
    support_hash = []
    for sup in support:
        sup.pop('type', None)
        sup.pop('first_seen_date', None)
        sup.pop('language', None)
        sup.pop('title_embedding', None)
        support_embed.append(sup.pop('content_embedding', None))
        support_hash.append(sup['chunk_hash'])

    all_docs = []
    for i in range(indexes['faiss'].ntotal):
        if indexes['faiss_idx'][i] in support_hash:
            continue
        else:
            all_docs.append(indexes['faiss'].reconstruct(i))

    support_PCA = models['dim_reducer'].transform(support_embed)
    docs_PCA = models['dim_reducer'].transform(all_docs)
    question_PCA = models['dim_reducer'].transform(
        np.array(question_embed).reshape(1, -1))

    all_data = np.concatenate((docs_PCA, support_PCA,  question_PCA), axis=0)
    print(len(docs_PCA))
    print(len(support_PCA))
    # px.scatter_3d(
    #     color_discrete_sequence=,
    # )

    fig = go.Figure(data=[go.Scatter3d(
        x=[plop[0] for plop in all_data],
        y=[plop[1] for plop in all_data],
        z=[plop[2] for plop in all_data],
        mode='markers',
        marker=dict(
            # size=12,
            color=['green']*len(docs_PCA) + ['blue'] *
            len(support_PCA) + ['red'],
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    st.plotly_chart(fig)

    st.write("Max Score")
    st.write(max_score)
    st.write("Hits")
    st.write(hits)
    st.write("Docs")
    st.write(support)


if st.button('clear database'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.write("Clearing")
    clear_indexes(indexes)
    st.success('Done!')

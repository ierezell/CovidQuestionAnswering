import plotly.graph_objects as go
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from datatypes import Entry, MetaData, RawEntry
from utils import remove_links, sanitize_text

from .chunker import chunker
from .metabuilder import create_metadata
CHECK_URL = "https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/informations-pour-les-femmes-enceintes-coronavirus-covid-19/"


def recurse_add(raw_entry: RawEntry, parents: List[Entry] = [], graph=None):
    # Clean entry
    raw_entry['content'], links = remove_links(raw_entry['content'])
    raw_entry['title'] = sanitize_text(raw_entry['title'])

    # Chunk entry content
    entries: List[Entry] = []
    for chunk in chunker(raw_entry):
        metadatas = create_metadata(chunk, links)
        entry: Entry = {"chunk": chunk, "metadatas": metadatas}
        entries.append(entry)

        graph.add_node(entry['chunk']['chunk_hash'], **entry)

        if parents:
            for dad in parents:
                graph.add_edge(dad['chunk']['chunk_hash'],
                               entry['chunk']['chunk_hash'])

    for child in raw_entry.get('children', []):
        recurse_add(child, parents=entries, graph=graph)


def create_links(graph: nx.DiGraph):
    all_path = []
    all_links = []
    for node_hash, node_data in graph.nodes(data=True):
        all_path.append(node_data['chunk']['path'])
        if node_data['chunk']['path'] == "https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/informations-pour-les-femmes-enceintes-coronavirus-covid-19/":
            print('sdf')
        links = node_data['metadatas']['links']
        for other_node_hash, other_node_data in graph.nodes(data=True):
            if node_hash == other_node_hash:
                continue
            for link in links:
                all_links.append(link['path'])
                if other_node_data['chunk']['path'] == link['path']:
                    print('added sge')
                    graph.add_edge(node_hash, other_node_hash)
    # print(set(all_path))
    # print(set(all_links))
    # print("inter  ", set(all_links).intersection(set(all_path)))


def create_index(raw_entry: RawEntry):
    index: Dict[str, List[str]] = {}
    graph = nx.DiGraph()

    recurse_add(raw_entry, parents=[], graph=graph)
    create_links(graph)
    nx.write_gexf(graph, './graph.gexf')
    # nx.write_gml(graph, './graph.gml')
    with open('./graph.json', 'w') as file:
        json.dump(nx.cytoscape_data(graph), file)
    with open('graph_data.json', 'w') as outfile1:
        json.dump(nx.node_link_data(graph), outfile1)
    # nx.draw(graph)
    # plt.show()
    visualise(graph)


def visualise(graph: nx.Graph):
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
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node_hash, node_data in graph.nodes(data=True):
        x, y = pos[node_hash]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_data['chunk']['chunk_hash'] + "<br>" +
                         node_data['chunk']['original_hash'] + "<br>" +
                         node_data['chunk']['path'] + "<br>" +
                         "<br>".join([l['path'] for l in node_data['metadatas']['links']]) + "<br>" +
                         node_data['chunk']['title'] + "<br>" +
                         node_data['chunk']['content'][:30] + "<br>"
                         )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
        title='Gouv graph',
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    for idx in range(0, len(edge_x), 3):
        fig.add_annotation(dict(
            x=edge_x[idx+1],  # arrows' head
            y=edge_y[idx+1],  # arrows' head
            ax=edge_x[idx],  # arrows' tail
            ay=edge_y[idx],  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # if you want only the arrow
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor='black'
        ))

    fig.show()

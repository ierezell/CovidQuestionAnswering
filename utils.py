import hashlib
import re
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from datatypes import Link

# regex_links_markdown = re.compile(
# r"(\[([\w\-'’ \xa0\(\)]+)\]\(((?:[\w\.\/:-])+)\))")

regex_links_markdown = re.compile(r"(\[(.*?)\]\((.*?)\))")
# Match 0 all => [Text](url)
# Match 1 all => Text
# Match 2 all => Url


def remove_links(text: str) -> Tuple[str, List[Link]]:
    cleaned_text = deepcopy(text)
    links: List[Link] = []

    matched_links = regex_links_markdown.findall(text)
    for match in list(dict.fromkeys(matched_links)):
        link: Link = {'path': match[2],
                      'start': cleaned_text.index(match[0]),
                      'name': match[1]}
        links.append(link)
        cleaned_text = cleaned_text.replace(match[0], match[1])

    return cleaned_text, links


def sanitize_text(text: str) -> str:
    new_text = deepcopy(text)
    new_text = new_text.strip()
    new_text = new_text.replace(';', '. ')
    new_text = new_text.replace('\xa0', ' ')
    new_text = new_text.replace('à', 'à')
    new_text = new_text.replace('\t', ' ')
    new_text = new_text.replace('\r', ' ')
    new_text = new_text.replace('\\[', '')
    new_text = new_text.replace('\\]', '')
    new_text = new_text.replace('_', '')
    new_text = new_text.replace('\n', ' ')
    new_text = new_text.replace('*', ' ')
    new_text = re.sub(r'\s+', ' ', new_text)
    new_text = re.sub(r'\.(\w)', r'. \g<1>', new_text)
    return new_text


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm > 0:
        return np.dot(a, b) / norm
    else:
        return 0


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

    return fig

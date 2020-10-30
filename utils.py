"""
Utils file which regroup miscalenous fonctions use in the rest of the code
"""
import hashlib
import re
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from datatypes import Link

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

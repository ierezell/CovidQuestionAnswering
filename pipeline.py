import json


from datatypes import RawEntry
from indexer.indexer import create_index

# with open('datasets/gouv/DOCUMENTS.json', 'r') as file:
with open('datasets/gouv/dummy.json', 'r') as file:
    gouv_data: RawEntry = json.load(file)

create_index(gouv_data)

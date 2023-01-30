import os
import requests

# Script for testing the implementation of the conversational entity linking API
#
# To run:
#
#    python .\src\REL\server.py $REL_BASE_URL wiki_2019 --ner_models ner-fast ner-fast-with-lowercase
# or
#    python .\src\REL\server.py $env:REL_BASE_URL wiki_2019 --ner_models ner-fast ner-fast-with-lowercase
#
# Set $REL_BASE_URL to where your data are stored (`base_url`)
#
# These paths must exist:
# - `$REL_BASE_URL/bert_conv`
# - `$REL_BASE_URL/s2e_ast_onto `
#
# (see https://github.com/informagi/conversational-entity-linking-2022/tree/main/tool#step-1-download-models)
#


host = 'localhost'
port = '5555'

inputs = (
    {
        'tagger': 'ner-fast',
        'text': 'REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.',
        'spans': [],
    },
    {
        'mode': 'ne',
        'tagger': 'ner-fast-with-lowercase',
        'text': 'REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.',
        'spans': [],
    },
    {
        'mode': 'conv',
        'tagger': 'default',
        'text':
        [
            {
                'speaker': 'USER',
                'utterance': 'I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.',
            },
            {
                'speaker': 'SYSTEM',
                'utterance': 'Some people are allergic to histamine in tomatoes.',
            },
            {
                'speaker': 'USER',
                'utterance': 'Talking of food, can you recommend me a restaurant in my city for our anniversary?',
            },
        ],
    },
    {
        'mode': 'ne_concept',
    },
    {
        'mode': 'fail',
    },
    {
        'text': 'Hello world.',
        'this-argument-does-not-exist': None,
    },
)

for payload in inputs:
    endpoint = ''

    print('Input API:')
    print(payload)
    print()
    print('Output API:')
    print(requests.post(f'http://{host}:{port}/{endpoint}', json=payload).json())
    print()
    print('----------------------------')
    print()


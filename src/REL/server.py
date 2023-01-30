from REL.response_model import ResponseModel

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Annotated

app = FastAPI()

@app.get("/")
def root():
    """Returns server status."""
    return {
        "schemaVersion": 1,
        "label": "status",
        "message": "up",
        "color": "green",
    }

class NamedEntityConfig(BaseModel):
    mode: Literal['ne']

    text: str = Field(..., description="Text for entity linking or disambiguation.")
    spans: List[str] = Field(..., description=(
        "For EL: the spans field needs to be set to an empty list. "
        "For ED: spans should consist of a list of tuples, where each "
        "tuple refers to the start position and length of a mention."))
    model: Literal[
        "ner-fast", 
        "ner-fast-with-lowercase",
    ] = Field("ner-fast", description="NER model to use.")


class NamedEntityConceptConfig(BaseModel):
    mode: Literal['ne_concept']


class ConversationTurn(BaseModel):
    speaker: Literal["USER", "SYSTEM"] = Field(..., description="Speaker for this turn.")
    utterance: str = Field(..., description="Input utterance.")


class ConversationConfig(BaseModel):
    mode: Literal['conv']

    text: List[ConversationTurn] = Field(..., description="Conversation as list of turns between two speakers.")
    model: Literal[
        "default", 
    ] = Field("default", description="NER model to use.")


class DefaultConfig(NamedEntityConfig):
    mode: str = 'ne'


Config = Annotated[Union[DefaultConfig,
                         NamedEntityConfig,
                         NamedEntityConceptConfig,
                         ConversationConfig],
                       Field(discriminator='mode')]


@app.post("/")
def root(config: Config):
    """Submit your text here for entity disambiguation or linking."""

    print(config)

    return config

    if config.mode == 'conv':
        text = config.dict()['text']

        conv_handler = conv_handlers[config.model]
        response = conv_handler.annotate(text)

    elif config.mode == 'ne':
        handler = handlers[config.model]
        response = handler.generate_response(text=config.text, spans=config.spans)

    elif config.mode == 'ne_concept':
        raise NotImplementedError

    return response


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("wiki_version")
    p.add_argument("--ed-model", default="ed-wiki-2019")
    p.add_argument("--ner-model", default="ner-fast", nargs="+")
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    # from REL.crel.conv_el import ConvEL
    # from REL.entity_disambiguation import EntityDisambiguation
    # from REL.ner import load_flair_ner

    # ed_model = EntityDisambiguation(
    #     args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    # )

    # handlers = {}

    # for ner_model_name in args.ner_model:
    #     print('Loading NER model:', ner_model_name)
    #     ner_model = load_flair_ner(ner_model_name)
    #     handler = ResponseModel(args.base_url, args.wiki_version, ed_model, ner_model)
    #     handlers[ner_model_name] = handler

    # conv_handlers = {'default': ConvEL(args.base_url, args.wiki_version, ed_model=ed_model)}

    uvicorn.run(app, port=args.port, host=args.bind)

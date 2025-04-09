from typing import Literal, Union

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


EmbeddingsModel = Union[OpenAIEmbeddings, HuggingFaceEmbeddings,SentenceTransformer]


def get_embedding_model(
    model_id: str,
    model_type: str,
    device: str = "cpu",
) -> EmbeddingsModel:

    if model_type == "openai":
        return get_openai_embedding_model(model_id)
    elif model_type == "huggingface":
        return get_huggingface_embedding_model(model_id, device)
    elif model_type == "bert":
        return get_sentence_transformer_model(model_id, device)
    else:
        raise ValueError(f"Invalid embedding model type: {model_type}")


def get_openai_embedding_model(model_id: str) -> OpenAIEmbeddings:
    """
    Note: needs an openai key
    """
    pass 

def get_huggingface_embedding_model(
    model_id: str, device: str
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )
def get_sentence_transformer_model (model_id: str,
                                    device:str) -> SentenceTransformer:
    return SentenceTransformer(device = device,
                               model_name_or_path = model_id)
    
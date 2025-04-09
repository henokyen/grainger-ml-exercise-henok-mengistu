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
    """Gets an instance of the configured embedding model.

    The function returns either an OpenAI or HuggingFace embedding model based on the
    provided model type.

    Args:
        model_id (str): The ID/name of the embedding model to use
        model_type (EmbeddingModelType): The type of embedding model to use.
            Must be either "openai" or "huggingface". Defaults to "huggingface"
        device (str): The device to use for the embedding model. Defaults to "cpu"

    Returns:
        EmbeddingsModel: An embedding model instance based on the configuration settings

    Raises:
        ValueError: If model_type is not "openai" or "huggingface"
    """

    if model_type == "openai":
        return get_openai_embedding_model(model_id)
    elif model_type == "huggingface":
        return get_huggingface_embedding_model(model_id, device)
    elif model_type == "bert":
        return get_sentence_transformer_model(model_id, device)
    else:
        raise ValueError(f"Invalid embedding model type: {model_type}")


def get_openai_embedding_model(model_id: str) -> OpenAIEmbeddings:
    """Gets an OpenAI embedding model instance.

    Args:
        model_id (str): The ID/name of the OpenAI embedding model to use

    Returns:
        OpenAIEmbeddings: A configured OpenAI embeddings model instance with
            special token handling enabled
    Note: needs an openai key
    """
    pass 


def get_huggingface_embedding_model(
    model_id: str, device: str
) -> HuggingFaceEmbeddings:
    """Gets a HuggingFace embedding model instance.

    Args:
        model_id (str): The ID/name of the HuggingFace embedding model to use
        device (str): The compute device to run the model on (e.g. "cpu", "cuda")

    Returns:
        HuggingFaceEmbeddings: A configured HuggingFace embeddings model instance
            with remote code trust enabled and embedding normalization disabled
    """
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )
def get_sentence_transformer_model (model_id: str,
                                    device:str) -> SentenceTransformer:
    """Gets a  SentenceTransformer model instance.

    Args:
        model_id (str): The ID/name of the SentenceTransformer embedding model to use
        device (str): The compute device to run the model on (e.g. "cpu", "cuda")

    Returns:
        SentenceTransformer: A configured SentenceTransformer embeddings model instance
    """
    return SentenceTransformer(device = device,
                               model_name_or_path = model_id)
    
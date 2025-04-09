from loguru import logger
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError
from commons.embeddings import get_embedding_model
import pandas as pd
import os
import json


def build_multi_index(
    data_dir: str,
    train_data_name: str,
    embedding_model_id: str,
    embedding_model_type: str,
    embedding_model_dim: int,
    index_name:  str,
    mapping_path: str
)->None:
    product_data = pd.read_csv(os.path.join(data_dir, train_data_name)).to_dict(orient='records')
    # Connect to Elasticsearch
    es = Elasticsearch("http://127.0.0.1:9200")
    logger.info(f"Reading index maapping from {mapping_path}")

    with open(mapping_path, 'r') as f:
        index_mappings = json.load(f)

    logger.info (f"Getting embeding id {embedding_model_id} which is of type {embedding_model_type}")
    embedding_model = get_embedding_model(embedding_model_id,embedding_model_type)
    
    
    logger.info(f"Creating indices in Elasticsearch")

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        logger.info(f"Index '{index_name}' deleted.")

    es.indices.create(index=index_name, body=index_mappings)
    logger.info(f"Index '{index_name}' created.")
    
    logger.info (f"Compute embeddings and inserting..")

    for prod in product_data:
        prod_id = prod["product_id"]
        product_description_embedding = embedding_model.encode(prod["product_description"] if pd.notna(prod["product_description"]) else '').tolist()
        product_bullets_embedding = embedding_model.encode(prod["product_bullet_point"] if pd.notna(prod["product_bullet_point"]) else '').tolist()
        product_title_embedding = embedding_model.encode(prod["product_title"] if pd.notna(prod["product_title"]) else '').tolist()

        doc = {
            "product_description": product_description_embedding,
            "product_bullets": product_bullets_embedding,
            "product_title_embed": product_title_embedding,
            "product_color": prod['product_color'] if pd.notna(prod['product_color']) else '',
            "product_brand": prod['product_brand'] if pd.notna(prod['product_brand']) else '',
            "product_title_text": prod['product_title'] if pd.notna(prod['product_title']) else '',
            "product_id": prod['product_id']
        }
        es.index(index=index_name, id =prod_id,  document = doc)
    logger.info (f"index creation is completed")
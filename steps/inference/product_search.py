from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from commons.embeddings import get_embedding_model

KNOWN_COLORS = ["black", "brown", "blonde", "red", "white", "blue", "green", "gray"]
KNOWN_BRANDS = ["L'Oréal", "Clairol", "Revlon", "Schwarzkopf", "Naturtint"]

def extract_color_and_brand(query):
    query_lower = query.lower()
    color = next((c for c in KNOWN_COLORS if c in query_lower), None)
    brand = next((b for b in KNOWN_BRANDS if b.lower() in query_lower), None)
    return color, brand


def build_query(user_query, query_vector):
    color, brand = extract_color_and_brand(user_query)

    filters = []
    if color:
        filters.append({ "term": { "product_color": color } })
    if brand:
        filters.append({ "term": { "product_brand": brand } })

    return {
        "size": 10,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            { "match": { "product_title_text": { "query": user_query, "boost": 10.0 } } }
                        ],
                        "filter": filters if filters else [],
                        "minimum_should_match": 1
                    }
                },
                "script": {
                    "source": """
                        0.9 * cosineSimilarity(params.query_vector, 'product_title_embed') +
                        0.1 * cosineSimilarity(params.query_vector, 'product_bullets') + 1.0
                    """,
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

def search(embedding_model_id: str,
           embedding_model_type: str,
           index_name: str,
           user_query: str
           ) -> None:
    
    es = Elasticsearch("http://localhost:9200")
    #model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    embedding_model = get_embedding_model(embedding_model_id,embedding_model_type)
    query_vector = embedding_model.encode(user_query).tolist()
    elastic_search_query  = build_query(user_query, query_vector)
    response = es.search(index=index_name, body=elastic_search_query)
    results = []

    for hit in response["hits"]["hits"]:
        score = hit['_score']
        title = hit["_source"].get("product_title_text", "")
        brand = hit["_source"].get("product_brand", "")
        color = hit["_source"].get("product_color", "")
        product_id= hit["_source"].get("product_id", "")
        results.append({'Score': score,
                        'Title': title,
                        'Brand':brand,
                        'Color': color,
                        'Product_id': product_id
                        })
        
    return results
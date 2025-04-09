import pandas as pd
from pathlib import Path
import os
from collections import defaultdict
import json
from steps.inference import product_search

def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def evaluate_mrr(results, ground_truth):
    total_rr = 0.0
    total_rr += reciprocal_rank(results, ground_truth)
    return total_rr / len(results)

def hits_at_n(retrieved, relevant, n):
    return int(any(doc_id in relevant for doc_id in retrieved[:n]))



def evaluate_hits_and_mrr(results, ground_truth, n_values=[1, 5, 10]):
    num_queries = len(results)
    hits_metrics = {f"HITS@{n}": 0 for n in n_values}
    total_rr = 0.0

    for qid, retrieved_docs in results.items():
        relevant_docs = ground_truth.get(qid, set())

        for n in n_values:
            hits_metrics[f"HITS@{n}"] += hits_at_n(retrieved_docs, relevant_docs, n)

        total_rr += reciprocal_rank(retrieved_docs, relevant_docs)

    # Normalize hits
    for n in n_values:
        hits_metrics[f"HITS@{n}"] /= num_queries

    mrr = total_rr / num_queries

    return hits_metrics, mrr
def evaluate_hitsN_MRR( data_dir: Path,
                        embedding_model_id: str,
                        embedding_model_type: str,
                        index_name: str,
                       ):
    # print('what is data dir', Path(data_dir))
    # print(os.path.join(Path(data_dir),'test/test.csv'))
    test_df = pd.read_csv(os.path.join(data_dir,'test/test.csv'))
    with open(os.path.join(data_dir, 'all_pairs.json'), 'r') as f:
        all_pairs = json.load(f)

    test_data = test_df.to_dict(orient='records')
    
    query_id_to_query = {item['query_id']: item['query'] for item in test_data}

    query_to_product_id = defaultdict(list)
    for query_id in query_id_to_query.keys():
        query_id = str(query_id)
        if query_id in all_pairs:
            query_to_product_id[query_id] = all_pairs[query_id]
        else:
            query_to_product_id.setdefault(query_id,[])

    search_results = defaultdict(list)
    for qid, query in query_id_to_query.items():
        results = product_search.search(embedding_model_id,
                                embedding_model_type,
                                index_name,
                                query
                                )
        results = [item['Product_id'] for item in results]
        search_results.setdefault(str(qid), []).extend(results)

    hits_metrics, mrr = evaluate_hits_and_mrr(search_results, query_to_product_id)
    print("Hits@N:", hits_metrics)
    print("MRR:", mrr)
# Semantic Product Search

The goal of this project is to build a vector index to enable semantic product discovery. It also aims to evaluate the effectiveness of the index by measuring its performance against a set of provided search queries.

---


##  Project Steps

### 1. Data Ingestion & Preprocessing
The dataset used for this project consists of product listings and search queries, derived from the Amazon’s [ESCI dataset](https://github.com/amazon-science/esci-data). The goal of preprocessing is to prepare clean, high-quality text inputs suitable for embedding and semantic search.

- **Cleaning Missing Values**  
  Removed invalid placeholder values such as `"", "nan", "none"` across all columns to ensure consistency and data quality.

- **Stripping HTML Tags**  
  The `product_description` and `product_bullet_point` fields contained HTML markup. These were parsed and cleaned to extract the raw text content.  
  This step was necessary because embedding models (e.g., SentenceTransformers, OpenAI Embeddings) are trained on clean, natural language text. HTML tags can degrade embedding quality and negatively impact search relevance.

- **Locale and Label Filtering**  
  Filtered the dataset to include only products with:
  - `product_locale = 'us'` (U.S. market)
  - `esci_label = 'E'` (Exact match label)

- **Query-Product Pair Management**  
  In some cases, a single `query_id` may correspond to multiple `product_ids`. Since the training set is capped at 500 rows, I retained additional `product_ids` for those `query_ids` separately. These are reserved for evaluation and testing to measure generalization on unseen examples.


### 2. Build Vector Database

###  Fields Considered

- **`product_description`**: Contains detailed product information. However, it can be long and include irrelevant content like instructions or contact details that don’t contribute to search relevance.
- **`product_title`**: Often reflects the main keywords and product identity. This field typically captures what users are actually looking for (e.g., "wireless headphones").
- **`product_brand`**: Important for brand-based queries such as "Nike shoes".
- **`product_color`**: Useful for queries like "red dress" or "black sneakers".
- **`product_bullet_point`**: May contain concise selling points and features, which can be useful in text-based matching.

###  Design Decision

After evaluating the fields, I chose `product_title` as the primary representation of the product's meaning or intent, as it most closely aligns with how users phrase their search queries. However, to support experimentation and flexibitly I also embed and index `product_description` and `product_bullet_point`

### Hybrid Search Strategy

To balance relevance and coverage, I implemented a hybrid search approach:

- **Semantic Search** on `product_title` and `product_bullets` using vector embeddings to capture intent and meaning.
- **Keyword Search** on `product_brand` and `product_color` to support exact matches and filtering.
- **Search Boosting**: To emphasize queries that directly overlap with the product title, I also indexed the `product_title` as text to enable text match search.

### 3. Inference

During inference, the system takes a user’s search query and processes it using the same embedding model used during vector index creation. This ensures consistency between the query representation and the indexed product embeddings.


### Elasticsearch Hybrid Search

A hybrid search is performed by combining **semantic similarity** with **text-based relevance**. The Elasticsearch `script_score` query is configured as follows:

- **Keyword Match**:
  - Uses `match` on the `product_title_text` field with a **boost of 10.0** to reward exact or near-exact matches.
- **Vector Similarity**:
  - Combines cosine similarity between the embedded query and:
    - `product_title_embed` (weighted at 90%)
    - `product_bullets` (weighted at 10%)
- **Optional Filters**:
  - Applied via a `filter` clause to restrict results based on attributes like brand or color.

### 4. Evaluation

To assess the quality of the search results, I used two standard information retrieval metrics:

### HITS@N

**HITS@N** measures whether at least one relevant product appears in the top-N retrieved results for a given query.

- **HITS@1**: Did the first result contain a relevant product?
- **HITS@5**: Was a relevant product retrieved in the top 5 results?
- **HITS@10**: Was it found within the top 10?

This metric evaluates **recall** within the top-N ranked results.

### MRR (Mean Reciprocal Rank)

**MRR** evaluates how high the first relevant product appears in the result list.

- The reciprocal rank is `1 / rank_of_first_relevant_result`
- MRR is the **mean** of these reciprocal ranks over all queries.

A higher MRR indicates that relevant results appear earlier in the search output, which reflects better **ranking performance**.

---

## Running the code
Each step can be maintained, debugged, or scaled independently, making the system suitable for iterative development or production deployment. Follow the steps below to set up and run the entire pipeline.
Note: I used Python 3.12.7 to run the steps
### 1 Get raw Dataset
1. Download the dataset from 

    - [Examples Dataset](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet)
    - [Products Dataset](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet)


2. Place the files inside the `/data/` directory at the root of the project.
### 2. Clone the follwoing repository and install necessary libraries

```bash
git https://github.com/henokyen/grainger-ml-exercise-henok-mengistu.git
cd grainger-ml-exercise-henok-mengistu
pip install -r requirements.txt
```

### 3. Start Elasticsearch (via Docker)
Elasticsearch is required for indexing and search. If not already installed, you can spin it up using Docker:
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```
### 4. Running the code
To run each stage of the pipeline, simply run the following make commands one after the other. Note: the product-search command will launch a gradio local webserver. Follow that link to start the product searching.
```bash
make generate-dataset
make build-vector-index
make product-search
make evaluate
```
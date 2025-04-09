export PYTHONPATH = .
# Step 1: Data ingestion and pre-processing
generate-dataset:
	@echo "Running data ingestion and generation..."
	@python -m tools.run --run-generate-dataset

# Step 2: Vector database building
build-vector-index:
	@echo "Building vector index..."
	@python -m tools.run --run-build-vector-index

# Step 3: Search 
product-search:
	@echo "Running product search engine..."
	@python -m tools.run --run-search

# Srep 4: Evaluation 
evaluate:
	@echo "Running Evaluation..."
	@python -m tools.run --run-evaluate
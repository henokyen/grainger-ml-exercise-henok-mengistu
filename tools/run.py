import argparse
import click
from pathlib import Path
import yaml
from utilis.utilis import load_config
from steps.data_ingestion import generation_dataset
from steps.vector_index import build_vector
from steps.inference import product_search
from steps.evaluation import evaluate
import gradio as gr
import pandas as pd

def gradio_search(user_query):
    root_dir = Path(__file__).resolve().parent.parent
    vector_args = load_config(root_dir / "configs" / "vector_index.yaml")
    keys_to_pass = ["embedding_model_id", "embedding_model_type", "index_name"]
    args_dict = {k: vector_args[k] for k in keys_to_pass}
    args_dict.update({'user_query':user_query })
    results = product_search.search(**args_dict)
    return  "\n".join(
        f"Score: {item['Score']:.2f}\n"
        f"Title: {item['Title']}\n"
        f"Brand: {item['Brand']}\n"
        f"Color: {item['Color']}\n"
        + "-" * 30
        for item in results
        )
    #return "\n\n".join(results)

@click.command()

@click.option(
    "--run-generate-dataset",
    is_flag=True,
    default=False,
    help="Generating and pre-processing."
)
@click.option(
    "--run-build-vector-index",
    is_flag=True,
    default=False,
    help="compute and build a vector database."
)
@click.option(
    "--run-search",
    is_flag=True,
    default=False,
    help="running product search."
)
@click.option(
    "--run-evaluate",
    is_flag=True,
    default=False,
    help="running evalution."
)

def main(
    run_generate_dataset: bool = False,
    run_build_vector_index: bool = False,
    run_search: bool = False,
    run_evaluate: bool =False
) -> None:
    assert (
        run_generate_dataset
        or run_build_vector_index
        or run_search
        or run_evaluate
    ), "Please specify an action to run."

    root_dir = Path(__file__).resolve().parent.parent
    if run_generate_dataset:
        generate_dataset_args = load_config(root_dir / "configs" / "data_generation.yaml")
        generation_dataset.get_data_from_disk(**generate_dataset_args)
    if run_build_vector_index:
        vector_args = load_config(root_dir / "configs" / "vector_index.yaml")
        build_vector.build_multi_index(**vector_args)
    if run_search:
        iface = gr.Interface(
            fn=gradio_search,
            inputs=gr.Textbox(label="Search Query", placeholder="e.g. fabric storage bins"),
            outputs=gr.Textbox(label="Search Results"),
            title="Semantic Search",
            description="Type a query and get semantic search results from Elasticsearch."
        )
        iface.launch()
    if run_evaluate:
        vector_args = load_config(root_dir / "configs" / "vector_index.yaml")
        keys_to_pass = ["embedding_model_id", "embedding_model_type", "index_name"]
        args_dict = {k: vector_args[k] for k in keys_to_pass}
        
        evaluate.evaluate_hitsN_MRR('data/', **args_dict)
    
if __name__=="__main__":
    main()
import pandas as pd
import shutil
from pathlib import Path
import os
from loguru import logger
from utilis.utilis import clean_and_filter
from collections import defaultdict
import json


def save_data_to_disck(df, path, file_name):
   if Path(path).exists():
        shutil.rmtree(path)
    
   Path(path).mkdir(parents=True)
   df.to_csv(os.path.join(path, file_name))
   
   logger.info (f"Data generation completed: {os.path.join(path,file_name)}")

def preprocess_fields(df, columns):
    for col in columns:
        df[col] = df[col].apply(clean_and_filter)
    return df

def get_data_from_disk(data_dir: Path,
                       output_dir_train:Path,
                       output_dir_test:Path,
                       train_data_name: str,
                       test_data_name: str,
                       max_rows:int,
                       unique_queires: int
                       ) -> None:

   logger.info (f"Reading raw data from {data_dir}")
   df_examples = pd.read_parquet(os.path.join(data_dir, 'shopping_queries_dataset_examples.parquet'))
   df_products = pd.read_parquet(os.path.join(data_dir, 'shopping_queries_dataset_products.parquet'))
  
   all_product_columns = df_products.columns.to_list()
   all_query_columns = df_examples.columns.to_list()
   invalid_values = ["", "nan", "none"]
   df_products[all_product_columns] = df_products[all_product_columns].astype('str')
   df_examples[all_query_columns] = df_examples[all_query_columns].astype('str')

   clean_mask = ~df_products[all_product_columns].apply(lambda col: col.str.strip().str.lower().isin(invalid_values) | col.isna()).any(axis=1)
   df_products_cleaned = df_products[clean_mask].reset_index(drop=True)

   logger.info (f"Cleansing ..")
   df_products_cleaned = preprocess_fields(df_products_cleaned, ['product_description','product_bullet_point'])
   
   df_examples_products = pd.merge(
        df_examples,
        df_products_cleaned,
        how='inner',
       left_on=['product_locale','product_id'],
       right_on=['product_locale', 'product_id']
    )
   df_examples_products_us_E = df_examples_products[(df_examples_products['product_locale'] == 'us') & (df_examples_products['esci_label'] == 'E')]
   df_examples_products_train = df_examples_products_us_E[df_examples_products_us_E['split'] == "train"]

   unique_queries = set(df_examples_products_train['query_id'].sample(n=unique_queires, random_state=42, replace=False).to_list())
   filtered_df = df_examples_products_train[df_examples_products_train['query_id'].isin(unique_queries)]
   filtered_df = filtered_df.drop_duplicates(subset=['query_id', 'product_id']) 
   df_train = filtered_df.sample(n=max_rows, random_state=42)

   #curating test dataset
   train_pairs = set(zip(df_train['query_id'], df_train['product_id']))
   all_pairs = set(zip(filtered_df['query_id'], filtered_df['product_id']))
   test_pairs = all_pairs - train_pairs
   test_pairs_df = pd.DataFrame(list(test_pairs), columns=['query_id', 'product_id'])
   df_test = test_pairs_df.merge(
        df_examples_products_train,
        on=['query_id', 'product_id'],
        how='left'
    )
   
   all_pair_dict = defaultdict(list)
   for key, value in all_pairs:
        all_pair_dict.setdefault(key, []).append(value)

   with open(os.path.join(data_dir,'all_pairs.json'), 'w') as f:
       json.dump(all_pair_dict, f, indent=2)

   save_data_to_disck(df_train, output_dir_train, train_data_name)
   save_data_to_disck(df_test, output_dir_test,test_data_name)
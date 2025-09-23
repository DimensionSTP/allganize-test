#!/bin/bash

indices_name="data_base_only.faiss"
items_name="data_base_only.csv"

python set_vector_store.py \
    indices_name=$indices_name \
    items_name=$items_name
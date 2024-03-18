# Databricks notebook source
# Databricks notebook source
import torch

if 'config' not in locals():
  config = {}

# Use Case
config['use-case']="insurance_qa_bot"
config['use-case-raw']="insurance_qa_bot_raw"

# Define the model we would like to use
config['model_id'] = 'meta-llama/Llama-2-13b-chat-hf'

# Get the username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')


# Create catalog nd database 
config['catalog'] = 'genaibootcamp'
_ = spark.sql(f"create catalog if not exists {config['catalog']}")
_ = spark.catalog.setCurrentCatalog(config['catalog'])

# create database if not exists
config['database_name'] = username.split("@")[0] + 'qabot'
_ = spark.sql(f"create database if not exists {config['catalog']}.{config['database_name']}")
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# Set Environmental Variables for tokens
import os
config['PAT_TOKEN'] = 'pat_token'


# Set document path
config['loc'] = "/Volumes/genaibootcamp/data/raw_data/"

# Set vector store name 
config['vector_endpoint'] = "vector-search-demo-endpoint"
config['vector_index'] = "insurance_docs"
config["embedding_model_endpoint"] = "databricks-bge-large-en"


# Set model configs
config['llm_model'] = 'databricks-llama-2-70b-chat' 
config['model_name'] = f"{config['catalog']}.{config['database_name'] }.insurance_chatbot_model"
config['serving_endpoint_name']=username.split("@")[0]+"qa_bot"

config["scope_name"]= "llm-scope"
config["secret_name"]="chain-key"
config["host"]="db_host"

# COMMAND ----------



# Databricks notebook source
# MAGIC %md # Vector Search in Databricks
# MAGIC
# MAGIC
# MAGIC **Pre-req**: This notebook assumes you have already created a Model Serving endpoint for the embedding model.  See `embedding_model_endpoint` below, and the companion notebook for creating endpoints.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Extracting Databricks documentation sitemap and pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC The first step in the RAG app is to build the Vector and retrieve relevant documents to our queestion.
# MAGIC
# MAGIC Let's see how we can do that!

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/config"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %md ## Let's remind ourselves the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC The following creates the source Delta table.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {(config['use-case'])}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

vector_search_endpoint_name = config['vector_endpoint']

# COMMAND ----------

# Run only once for setting up the lab, all the users will store indexes in the same endpoint
# vsc.create_endpoint(
#     name=vector_search_endpoint_name,
#     endpoint_type="STANDARD"
# )

# COMMAND ----------

endpoint = vsc.get_endpoint(
  name=vector_search_endpoint_name)
endpoint

# COMMAND ----------

# MAGIC %md ## Create vector index

# COMMAND ----------

# Vector index
vs_index_fullname = f"{config['catalog']}"+"."+f"{config['database_name']}"+"."+f"{config['vector_index']}"
embedding_model_endpoint = config["embedding_model_endpoint"]

# COMMAND ----------

index = vsc.create_delta_sync_index(
  endpoint_name=config['vector_endpoint'],
  source_table_name=f"{config['catalog']}.{config['database_name']}.{config['use-case']}",
  index_name= f"{config['catalog']}.{config['database_name']}.{config['vector_index']}",
  pipeline_type='TRIGGERED',
  primary_key="New_ID",
  embedding_source_column="text",
  embedding_model_endpoint_name=embedding_model_endpoint
)
index.describe()

# COMMAND ----------

# MAGIC %md ## Get a vector index  
# MAGIC
# MAGIC Use the get_index() method to retrieve the vector index object using the vector index name. You can also use the describe() method on the index object to see a summary of the index's configuration information.

# COMMAND ----------

index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)

index.describe()

# COMMAND ----------

# MAGIC %md ## Similarity search
# MAGIC
# MAGIC Query the Vector Index to find similar documents!

# COMMAND ----------

# Wait for the endpoint to be ready before runnign this,otherwise it will error
all_columns = spark.table(f"{config['catalog']}.{config['database_name']}.{config['use-case']}").columns

results = index.similarity_search(
  query_text="Am i covered in case of accident?",
  columns=all_columns,
  num_results=10)
results

# COMMAND ----------

# Search with a filter.
results = index.similarity_search(
  query_text="Damage in property",
  columns=all_columns,
  filters={"page": (5,6,7,8,9)},
  num_results=10)

results

# COMMAND ----------



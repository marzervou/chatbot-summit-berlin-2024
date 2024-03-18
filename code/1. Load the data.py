# Databricks notebook source
# MAGIC %md So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.

# COMMAND ----------

!pip install pypdf

# COMMAND ----------

# MAGIC %run "./utils/config"

# COMMAND ----------

# MAGIC %run "./utils/preprocess"

# COMMAND ----------

# DBTITLE 1,Step 1: Import Required Functions
import json
import pyspark.sql.functions as fn
import pyspark.sql.functions as fn
from langchain.text_splitter import TokenTextSplitter
# from utils.preprocess import preprocess_using_langchain

# COMMAND ----------

# DBTITLE 1,Step 2: Load the Raw Data to Table
df = preprocess_using_langchain(config)
display(df)

# COMMAND ----------

_ = (
  spark.createDataFrame(df)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable(config['use-case-raw'])
  )

# COMMAND ----------

# Retrieve Raw Inputs
raw_inputs = (
  spark
    .table(config['use-case-raw'])
  ) 

display(raw_inputs)

# COMMAND ----------

# DBTITLE 1,Step 3: Chunk Documents
# Chunking Configurations
chunk_size = 1000
chunk_overlap = 200

# Divide Inputs into Chunks
@fn.udf('array<string>')
def get_chunks(text):

  # instantiate tokenization utilities
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # split text into chunks
  return text_splitter.split_text(text)


# split text into chunks
chunked_inputs = (
  raw_inputs
    .withColumn('chunks', get_chunks('full_text')) # divide text into chunks
    .drop('full_text')
    .withColumn('num_chunks', fn.expr("size(chunks)"))
    .withColumn('chunk', fn.expr("explode(chunks)"))
    .drop('chunks')
    .withColumnRenamed('chunk','text')
  )

  # display transformed data
display(chunked_inputs)

# COMMAND ----------

# DBTITLE 1,Step 3: Save Data to Table
# DBTITLE 
# save data to table
_ = (
  chunked_inputs
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(config['use-case'])
  )

# count rows in table
print(spark.table(config['use-case']).count())

# COMMAND ----------

print(config['use-case'])

# COMMAND ----------



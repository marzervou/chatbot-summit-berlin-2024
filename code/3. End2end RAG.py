# Databricks notebook source
# MAGIC %md # End2end RAG in Databricks
# MAGIC
# MAGIC
# MAGIC **Pre-req**: This notebook assumes you have already created a Model Serving endpoint for the embedding model.  See `embedding_model_endpoint` below, and the companion notebook for creating endpoints.

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.0 langchain==0.0.344 databricks-vectorsearch==0.22  mlflow[databricks]
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC %pip install gradio
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/config"

# COMMAND ----------

# MAGIC %run "./utils/DatabricksApp"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Let's add the LLM model in the RAG app and chain all the app together
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">

# COMMAND ----------

# MAGIC %md ## Get a vector index  
# MAGIC
# MAGIC Use the get_index() method to retrieve the vector index object using the vector index name. You can also use the describe() method on the index object to see a summary of the index's configuration information.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
import os

# COMMAND ----------

vector_search_endpoint_name = config['vector_endpoint']
vs_index_fullname = f"{config['catalog']}"+"."+f"{config['database_name']}"+"."+f"{config['vector_index']}"
embedding_model_endpoint = config["embedding_model_endpoint"]

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)
index.describe()

# COMMAND ----------

# MAGIC %md ## Let's add the chain
# MAGIC
# MAGIC The first column retrieved is loaded into page_content and the rest into metadata.

# COMMAND ----------

# DBTITLE 1,Let's create a pat token to authenticate
# url used to send the request to your model from the serverless endpoint
os.environ["DATABRICKS_HOST"] = config["host"]
os.environ['DATABRICKS_TOKEN'] = config['PAT_TOKEN']
serving_endpoint_name = config['serving_endpoint_name']
os.environ["DATABRICKS_ENDPOINT"] = os.environ["DATABRICKS_HOST"] + "/serving-endpoints/" + serving_endpoint_name + "/invocations"

# COMMAND ----------

# DBTITLE 1,Let's test our embedding model
embedding_model = DatabricksEmbeddings(endpoint=config["embedding_model_endpoint"])
print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

# COMMAND ----------

# DBTITLE 1,Let's test our Llama2 model
# Test Databricks Foundation LLM model
from langchain.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint=config['llm_model'], max_tokens = 200)
print(f"Test chat model: {chat_model.predict('Supplemental Liability Plus Excess Insurance')}")

# COMMAND ----------

# DBTITLE 1,Let's put our vector store in a function to be able to retrieve the docs
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = config["host"]
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=config['vector_endpoint'],
        index_name=f"{config['catalog']}.{config['database_name']}.{config['vector_index']}"
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Am i covered in case of accident?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# DBTITLE 1,Assembling the complete RAG Chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

TEMPLATE = """You are a assistant built to answer policy related questions based on the context provided, the context is a document and use no other information.If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know.
Given the context:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vectorstore,
    chain_type_kwargs={"prompt": prompt})


# COMMAND ----------

# DBTITLE 1,Test chatbot
import langchain
langchain.debug = False 
question = {"query": "Am i coverd anywehre in the world?"}
answer = chain.run(question)

# COMMAND ----------

answer

# COMMAND ----------

# MAGIC %md ### Let's put the app in a serving endpoint

# COMMAND ----------

# DBTITLE 1,Register with mlflow
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = config['model_name']

with mlflow.start_run(run_name="chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )


# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

host=os.environ["DATABRICKS_HOST"]

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=config['model_name'],
            entity_version=1,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/llm-scope/chain-key}}",  # <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to  newer version , this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,Let's test the endpoint
question = "Am i coverd anywhere in the world?"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])

# COMMAND ----------

# MAGIC %md ### Let's build a quick App

# COMMAND ----------

build_app()

# COMMAND ----------



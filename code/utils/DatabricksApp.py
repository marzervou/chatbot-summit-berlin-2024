# Databricks notebook source
import itertools
import gradio as gr
import requests
import os
from gradio.themes.utils import sizes


def respond(message, history):

    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    local_token = os.getenv('DATABRICKS_TOKEN')
    local_endpoint = os.getenv('DATABRICKS_ENDPOINT')
    if local_token is None or local_endpoint is None:
        return "ERROR missing env variables"

    # Add your API token to the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {local_token}'
    }

    q = {"inputs": [message]}
    try:
        response = requests.post(
            local_endpoint, json=q, headers=headers, timeout=100)
        response_data = response.json()
        #print(response_data)
        response_data=response_data["predictions"][0]
        #print(response_data)

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"
        # + str(response.status_code) + " response:" + response.text

    # print(response.json())
    return response_data


def build_app():
    theme = gr.themes.Soft(
        text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
    )

    demo = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
        textbox=gr.Textbox(placeholder="Ask me a question",
                        container=False, scale=7),
        title="Databricks LLM RAG demo - Chat with llama2 Databricks model serving endpoint",
        description="This chatbot is a demo example for the dbdemos llm chatbot.",
        examples=[["what is limit of the misfueling cost covered in the policy?"],
                ["What happens if I lose my keys?"],
                ["what is the maximum Age of a Vehicle the insurance covers?"],],
        cache_examples=False,
        theme=theme,
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
    )
    
    demo.launch(share=True)


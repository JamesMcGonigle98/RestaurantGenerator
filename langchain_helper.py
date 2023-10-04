"""
Langchain Streamlit App

Creating a streamlit app that uses an LLM
"""

__date__ = "2023-10-04"
__author__ = "JamesMcGonigle"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import openai

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from secret_key import openai_key, serpapi_key

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from diffusers import DiffusionPipeline
import torch



# %% --------------------------------------------------------------------------
# Create Open AI Model
# -----------------------------------------------------------------------------
llm = OpenAI(openai_api_key= openai_key)


# %% --------------------------------------------------------------------------
# Add memory
# -----------------------------------------------------------------------------
# memory = ConversationBufferMemory()
# convo = ConversationChain(llm=llm)
# window = ConversationBufferWindowMemory(k=1)


# %% --------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------
# First Template (for name)
prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template = "I want to open a restaurant for {cuisine} food. Suggest a fancy name for it. Only one name please"

)

# Second Template (for items)
prompt_template_items = PromptTemplate(
    input_variables=['number_of_items','restaurant_name'],
    template = "Suggest {number_of_items} menu items for {restaurant_name}. Return it as a comma seperated list"

)

# %% --------------------------------------------------------------------------
# Creating a Function
# -----------------------------------------------------------------------------
def generate_restaurant_name_and_items(cuisine, number_of_items):

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
    chains = [name_chain, items_chain],
    input_variables=['cuisine','number_of_items'],
    output_variables=['restaurant_name','menu_items']
    )

    response = chain({'cuisine':cuisine,
                      'number_of_items':number_of_items})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian",5))




# %% --------------------------------------------------------------------------
# Creating Agent (Wikipedia)
# -----------------------------------------------------------------------------
# tools = load_tools(["wikipedia","llm-math"], llm = llm)

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )

# # %% --------------------------------------------------------------------------
# # Creating Agent (SerpAPI)
# # -----------------------------------------------------------------------------
# tools = load_tools(["serpapi","llm-math"], llm = llm)

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )





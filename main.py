"""
Streamlit App

Streamlit App for LangChain Interaction
"""

__date__ = "2023-10-04"
__author__ = "JamesMcGonigle"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import langchain_helper



# %% --------------------------------------------------------------------------
# Streamlit App Design
# -----------------------------------------------------------------------------
st.title("Restaurant Generator")

cuisine = st.sidebar.text_input("What type of food do you want to make?")
number_of_items = st.sidebar.number_input("How many items do you want on this menu?", step=1)

search = st.sidebar.button("Generate Restaurant")

if search:

    response = langchain_helper.generate_restaurant_name_and_items(cuisine, number_of_items)
    
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(",")
    st.write("**Menu Items**")

    for item in menu_items:
        st.write("-", item)


# %%

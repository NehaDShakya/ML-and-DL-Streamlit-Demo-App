import streamlit as st
from multiapp import MultiApp
from pages.ml import tp_01, tp_02  # import your app modules here

app = MultiApp()

st.markdown(
    """
# ML and DL Streamlit Demo App
This app is compilation of the TPs I did in my Intro to Machine Learning and Deep Learning Course.
"""
)

# Add all your application here
app.add_app("(ML) TP 01 :  Linear Regression Single Variable", tp_01.app)
app.add_app("(ML) TP 02 :  Linear Regression Multiple Variables", tp_02.app)

# The main app
app.run()

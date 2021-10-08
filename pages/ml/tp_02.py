# import required libraries
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from word2number import w2n


def app():
    # Add Page Title
    st.title("(ML) TP-02 : Linear Regression Multiple Variable")

    st.write(
        "In this TD we try and predict the price of house given its area, number of bedrooms and age. We also plot the linear regrssion."
    )

    # import the data sets
    df_home_prices = pd.read_csv("data/ml/tp_02/homeprices.csv")

    st.write("We have a dataset of house prices that we use to train our model:")
    # printing the prices dataset
    st.write(df_home_prices.head(10))

    # Data Cleaning
    st.write("We check if there is any missing values:")
    st.write(df_home_prices.isnull().sum())

    st.write(
        "Since we have missing values in the  bedroom column we fill them with the median:"
    )
    # Calculate medain bedrooms
    median_bedroom = math.floor(df_home_prices.bedrooms.median())
    st.write("Median bedroom number:", median_bedroom)

    # Fill missing bedroom data with median
    df_home_prices["bedrooms"] = df_home_prices["bedrooms"].fillna(median_bedroom)

    # printing the prices dataset
    st.write(df_home_prices.head(10))
    st.write(df_home_prices.isnull().sum())

    st.write(
        "Now we create and fit a linear regression model and check model coefficent and intercept: "
    )

    # Create and fit the linear regression model
    model_home_prices = LinearRegression()
    model_home_prices.fit(
        df_home_prices[["area", "bedrooms", "age"]], df_home_prices["price"]
    )

    # Check model coefficent and intercept
    st.write("Coefficient: ", model_home_prices.coef_[0])
    st.write("Intercept: ", model_home_prices.intercept_)

    st.header("Exercise")

    st.write(
        "In this exercise we predict the salary given the experience, test score (out of 10) and interview score (out of 10). We also check the coefficient and intercept of the model."
    )

    # Read training data csv files
    df_hiring = pd.read_csv("data/ml/tp_02/hiring.csv")
    st.write(df_hiring.head(10))

    # Data Cleaning
    st.write("We check if there is any missing values:")
    st.write(df_hiring.isnull().sum())

    st.markdown(
        """
        Since we have ;issing values we do the following:
        - Fill missing experience with 'zero'.
        - Fill missing test_score(out of 10) data with mean test_score.
        
        A linear model does not take text data so we convert experience data from string to integer (using word2number library).
        """
    )
    # Fill missing experience with "zero"
    df_hiring["experience"] = df_hiring["experience"].fillna("zero")

    # Calculate mean test_score
    mean_test_score = math.floor(df_hiring["test_score(out of 10)"].mean())
    st.write("Median test score:", mean_test_score)
    # Fill missing test_score(out of 10) data with mean test_score
    df_hiring["test_score(out of 10)"] = df_hiring["test_score(out of 10)"].fillna(
        mean_test_score
    )

    # Change experience data from word to number
    df_hiring["experience"] = df_hiring["experience"].apply(w2n.word_to_num)

    st.write(df_hiring.head(10))
    st.write(df_hiring.isnull().sum())

    # Create Linear regression object
    model_hiring = LinearRegression()

    # Train the model using training data
    model_hiring.fit(
        df_hiring[
            ["experience", "test_score(out of 10)", "interview_score(out of 10)"]
        ],
        df_hiring["salary($)"],
    )

    # Check model coefficent and intercept
    st.write("Coefficient: ", model_hiring.coef_[0])
    st.write("Intercept: ", model_hiring.intercept_)

# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


def app():
    # Add Page Title
    st.title("(ML) TP-01 : Linear Regression Single Variable")

    st.write(
        "In this TD we try and predict the price of house given its area. We also plot the linear regrssion."
    )

    # import the data sets
    df_prices = pd.read_csv("data/ml/tp_01/homeprices.csv")
    df_areas = pd.read_csv("data/ml/tp_01/areas.csv")

    st.write("We have a dataset of house prices that we use to train our model:")
    # printing the prices dataset
    st.write(df_prices.head())

    st.write(
        "We also have a dataset with area of houses for which we need to predict the price:"
    )
    # printing the areas dataset
    st.write(df_areas.head())

    st.write(
        "First we make a scatter plot for the area and price from the first dataset:"
    )
    # Creating scatter plot
    fig_prices_01 = plt.figure()
    plt.scatter(x=df_prices["area"], y=df_prices["price"], c="red", marker="+")
    plt.xlabel("Area (sq. feet)")
    plt.ylabel("Prices")
    # plt.plot(df_prices["area"], df_prices["price"], color="blue")

    # Display the plot
    st.write(fig_prices_01)

    st.write("Now we predict the prices for the houses in the second dataset:")

    # Create and fit the linear regression model
    model_prices = LinearRegression()
    model_prices.fit(df_prices[["area"]], df_prices["price"])

    fig_prices_02 = plt.figure()
    plt.scatter(x=df_prices["area"], y=df_prices["price"], c="red", marker="+")
    plt.xlabel("Area (sq. feet)")
    plt.ylabel("Prices")
    plt.plot(df_prices["area"], model_prices.predict(df_prices[["area"]]), color="blue")

    st.write(fig_prices_02)

    df_areas["price"] = model_prices.predict(df_areas[["area"]])
    st.write(df_areas.head())

    st.header("Exercise")

    st.write(
        "In this exercise we fit and graph the linear regression model for the per capita income dataset. We also check the slope and intercept of the model."
    )

    # Read training data csv files
    df_canada_pci = pd.read_csv("data/ml/tp_01/canada_per_capita_income.csv")
    st.write(df_canada_pci.head())

    st.write("First we make a scatter plot for the per capita incomein USD:")
    # Creating scatter plot
    fig_canada_pci_01 = plt.figure()
    plt.scatter(
        x=df_canada_pci["year"],
        y=df_canada_pci["per capita income (US$)"],
        c="red",
        marker="+",
    )
    plt.xlabel("Year")
    plt.ylabel("Per Capita Income (US$)")
    # plt.plot(df_canada_pci["year"], df_canada_pci["per capita income (US$)"], color = "blue")
    st.write(fig_canada_pci_01)

    # Create Linear regression object
    model_canada_pci = LinearRegression()

    # Train the model using training data
    model_canada_pci.fit(
        df_canada_pci[["year"]], df_canada_pci["per capita income (US$)"]
    )

    # Check model coefficent and intercept
    st.write("Coefficient: ", model_canada_pci.coef_[0])
    st.write("Intercept: ", model_canada_pci.intercept_)

    # Adding linear regression to scatter plot
    fig_canada_pci_02 = plt.figure()
    plt.scatter(
        x=df_canada_pci["year"],
        y=df_canada_pci["per capita income (US$)"],
        c="red",
        marker="+",
    )
    plt.xlabel("Year")
    plt.ylabel("Per Capita Income (US$)")
    plt.plot(
        df_canada_pci["year"],
        model_canada_pci.predict(df_canada_pci[["year"]]),
        color="blue",
    )

    st.write(fig_canada_pci_02)

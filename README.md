# Machine-Learning-Streamlit-App

This app is compilation of the TPs I did in my Intro to Machine Learning and Deep Learning Course.

## Demo

Launch the web app:

[[Streamlit App]](https://ml-and-dl-streamlit-demo-app.herokuapp.com/y)

## Reproducing this web app

To recreate this web app on your own computer, do the following.

### Create conda environment

Firstly, we will create a conda environment called *ml_dl_demo*

```
conda create -n ml_dl_demo python=3.8.11
```

Secondly, we will login to the *multipage* environement

```
conda activate ml_dl_demo
```

### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/NehaDShakya/ML-and-DL-Streamlit-Demo-App/master/requirements.txt

```

Pip install libraries

```
pip install -r requirements.txt
```

### Download and unzip this repo

Download [this repo](https://github.com/NehaDShakya/ML-and-DL-Streamlit-Demo-App/archive/refs/heads/master.zip) and unzip as your working directory.

### Launch the app

```
streamlit run app.py
```

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = st.file_uploader("Upload dataset")

if df is not None:
    df = pd.DataFrame(df)

    X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.3)

    trainset = tf.data.Dataset.from_tensor_slices((dict(X_train),dict(y_train))).batch(32)
    validationset = tf.data.Dataset.from_tensor_slices((dict(X_val),dict(y_val))).batch(32)

    indexes = st.multiselect("Index", df.columns)
    index_values = {}
    for index in indexes:
        index_values[index] = X_train[index].unique()

    numerical_cols_list = st.multiselect("Numerical Features", df.columns)
    #categorical_cols_list = st.multiselect("Categorical Features", df.columns)

    cols = [tf.feature_column.numeric_column(col) for col in numerical_cols_list]

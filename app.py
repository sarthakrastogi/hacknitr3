import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = st.file_uploader("Upload dataset")

if df is not None:
    df = pd.DataFrame(df)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    trainset = tf.data.Dataset.from_tensor_slices((dict(X_train),dict(y_train))).batch(32)
    validationset = tf.data.Dataset.from_tensor_slices((dict(X_val),dict(y_val))).batch(32)

    indexes = st.multiselect("Index", df.columns)
    index_values = {}
    for index in indexes:
        index_values[index] = X_train[index].unique()

    numerical_cols_list = st.multiselect("Numerical Features", df.columns)

    cols = [tf.feature_column.numeric_column(col) for col in numerical_cols_list]

    index_values_tf_embeddings = {}
    for index, u in index_values.items():
        col = tf.feature_column.categorical_column_with_vocabulary_list(index, u)
        index_values_tf_embeddings[index] = tf.feature_column.embedding_column(col, dimension=int(len(index_values[index])**(1/4)))
        tf.feature_columns.append(index_values_tf_embeddings[index])

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)



    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=1))


    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(trainset, validation_data=valset, epochs=10)

    embeddings = [model.get_weights()[i] for i in range(len(model.get_weights()))]

    embeddings_df = pd.DataFrame()
    for emb in embeddings:
        emb = pd.DataFrame(emb)
        embeddings_df.join(emb)

    embeddings_df.to_csv('embeddings_df.csv')

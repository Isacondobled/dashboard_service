from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px
import streamlit as st

X_raw, y_raw = load_iris(return_X_y=True, as_frame=True)
df_raw = X_raw
df_raw['species'] = y_raw

df_baking = df_raw.copy()
df_baking.columns = df_baking.columns.str.replace(' ', '_')
df_baking['species'] = df_baking['species'].map({
    0:'setosa',
    1:'versicolor',
    2:'virginica'
    })
df_baking['species'] = df_baking['species'].astype('category')
df = df_baking.copy()


st.title("Iris dashboard")
st.write("Iris dataset table")

species = st.selectbox('Filter to:', ['setosa','versicolor','virginica'])
st.write(df[df['species']==species])

my_plot = px.scatter_matrix(df, dimensions=['sepal_length_(cm)','sepal_width_(cm)', 'petal_length_(cm)','petal_width_(cm)'],
                            color='species')
st.plotly_chart(my_plot)

st.markdown('Se observa la separación de clases de manera clara en las variables sepal_length y sepal_width, esto es los vectores $x_1, x_2$')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Halaman Perbandingan",
)
st.title('Halaman Perbandingan')
st.subheader("Anda dapat melihat bagaimana perbandingan dari kedua model dengan lebih jelas pada halaman ini!")

if 'data_prediksi3' not in st.session_state or 'data_prediksiNW3' not in st.session_state:
    st.info("Anda belum menjalankan model!")
else:
    val_mape1 = st.session_state["val_mape1"].values[0]
    val_mape2 = st.session_state["val_mape2"].values[0]
    val_mape3 = st.session_state["val_mape3"].values[0]
    val_mapeNW1 = st.session_state["val_mapeNW1"].values[0]
    val_mapeNW2 = st.session_state["val_mapeNW2"].values[0]
    val_mapeNW3 = st.session_state["val_mapeNW3"].values[0]
    y_akt1 = st.session_state["y_akt1"]
    prednnstandar1 = st.session_state["prednnstandar1"]
    prednnmodif1 = st.session_state["prednnmodif1"]
    y_akt2 = st.session_state["y_akt2"]
    prednnstandar2 = st.session_state["prednnstandar2"]
    prednnmodif2 = st.session_state["prednnmodif2"]
    y_akt3 = st.session_state["y_akt3"]
    prednnstandar3 = st.session_state["prednnstandar3"]
    prednnmodif3 = st.session_state["prednnmodif3"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Uji BP Standar & Nguyen-Widrow**")
        st.write("**Uji 1**")
        st.write("**Uji 2**")
        st.write("**Uji 3**")

    with col2:
        st.write("**BP MAPE Testing (Model Standar)**")
        st.markdown(":green[" + f"{val_mape1[0]:.5f}" + "]")
        st.markdown(":green[" + f"{val_mape2[0]:.5f}" + "]")
        st.markdown(":green[" + f"{val_mape3[0]:.5f}" + "]")

    with col3:
        st.write("**BP MAPE Testing (Nguyen-Widrow)**")
        st.markdown(":green[" + f"{val_mapeNW1[0]:.5f}" + "]")
        st.markdown(":green[" + f"{val_mapeNW2[0]:.5f}" + "]")
        st.markdown(":green[" + f"{val_mapeNW3[0]:.5f}" + "]")

    kol1,kol2 = st.columns(2)
    with kol1:
        st.write("Grafik Prediksi Model Standar Uji 1")
        a = range(len(y_akt1))
        plt.scatter(a,prednnstandar1, color='blue')
        plt.scatter(a,y_akt1, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write("Grafik Prediksi Model Standar Uji 2")
        a = range(len(y_akt2))
        plt.scatter(a,prednnstandar2, color='blue')
        plt.scatter(a,y_akt2, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write("Grafik Prediksi Model Standar Uji 3")
        a = range(len(y_akt3))
        plt.scatter(a,prednnstandar3, color='blue')
        plt.scatter(a,y_akt3, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    with kol2:
        st.write("Grafik Prediksi Model NW Uji 1")
        a = range(len(y_akt1))
        plt.scatter(a,prednnmodif1, color='blue')
        plt.scatter(a,y_akt1, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write("Grafik Prediksi Model NW Uji 2")
        a = range(len(y_akt2))
        plt.scatter(a,prednnmodif2, color='blue')
        plt.scatter(a,y_akt2, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write("Grafik Prediksi Model NW Uji 3")
        a = range(len(y_akt3))
        plt.scatter(a,prednnmodif3, color='blue')
        plt.scatter(a,y_akt3, color='red')
        plt.xlabel("Jumlah Data Test")
        plt.ylabel("Hasil")
        plt.title("Perbandingan Data Prediksi Model Standar dan Data Aktual")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    data = {
        'Uji 1': [val_mape1[0], val_mapeNW1[0]],
        'Uji 2': [val_mape2[0], val_mapeNW2[0]],
        'Uji 3': [val_mape3[0], val_mapeNW3[0]]
    }
    df = pd.DataFrame(data, index=['Backpropagation', 'Backpropagation Nguyen-Widrow'])
    fig = go.Figure()
    colors = ['red', 'green']
    for i, (index, row) in enumerate(df.iterrows()):
        formatted_mape1 = f"{row['Uji 1']:.6f}"
        formatted_mape2 = f"{row['Uji 2']:.6f}"
        formatted_mape3 = f"{row['Uji 3']:.6f}"
        fig.add_trace(go.Bar(
            x=row.index,
            y=row.values,
            name=index,
            text=[formatted_mape1,formatted_mape2, formatted_mape3],
            marker_color=colors[i]
        ))
    fig.update_layout(
        title='Perbandingan MAPE antara Backpropagation dan Backpropagation Nguyen-Widrow',
        xaxis_title='Metode Uji',
        yaxis_title='MAPE (%)'
    )
    st.plotly_chart(fig, use_container_width=True)

    val_mape_avg = np.mean([val_mape1, val_mape2, val_mape3])
    val_mapeNW_avg = np.mean([val_mapeNW1, val_mapeNW2, val_mapeNW3])
    data = {
        'Method': ['Backpropagation Validation', 'Backpropagation Nguyen-Widrow Validation'],
        'Average MAPE': [val_mape_avg, val_mapeNW_avg],
        'Type': ['Backpropagation', 'Backpropagation Nguyen-Widrow']
    }
    df = pd.DataFrame(data)
    fig = go.Figure()
    for i, row in df.iterrows():
        if row['Type'] == 'Backpropagation':
            color = 'red'
        else:
            color = 'green'
        formatted_avg_mape = f"{row['Average MAPE']:.6f}"
        fig.add_trace(go.Bar(
            x=[row['Method']],
            y=[row['Average MAPE']],
            name=row['Type'],
            text=[formatted_avg_mape],
            textposition='auto',
            offsetgroup=str(row['Average MAPE']),
            hoverinfo='none',
            marker_color=color
        ))
    # Atur layout grafik
    fig.update_layout(
        title='Perbandingan rata-rata MAPE Backpropagation dan Backpropagation Nguyen-Widrow',
        xaxis_title='Metode',
        yaxis_title='Average MAPE (%)',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

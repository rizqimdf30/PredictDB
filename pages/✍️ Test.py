import streamlit as st
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, conversion, default_converter
from rpy2.robjects.conversion import localconverter

st.set_page_config(
    page_title="Halaman Tes",
)
st.title('Halaman Tes')
st.subheader("Anda dapat melakukan tes model terbaik pada halaman ini!")

if 'model3' not in st.session_state or 'modelNW3' not in st.session_state:
    st.info("Belum ada model untuk dievaluasi!")
else:
    model1 = st.session_state['model1']
    model2 = st.session_state['model2']
    model3 = st.session_state['model3']
    modelNW1 = st.session_state['modelNW1']
    modelNW2 = st.session_state['modelNW2']
    modelNW3 = st.session_state['modelNW3']
    X_train1 = st.session_state['xtrain1']
    X_train2 = st.session_state['xtrain2']
    X_train3 = st.session_state['xtrain3']
    X_test1 = st.session_state['xtest1']
    X_test2 = st.session_state['xtest2']
    X_test3 = st.session_state['xtest3']
    y_train1 = st.session_state['ytrain1']
    y_train2 = st.session_state['ytrain2']
    y_train3 = st.session_state['ytrain3']
    y_test1 = st.session_state['ytest1']
    y_test2 = st.session_state['ytest2']
    y_test3 = st.session_state['ytest3']
    Xa_asli1 = st.session_state['Xa_asli1']
    Xa_asli2 = st.session_state['Xa_asli2']
    Xa_asli3 = st.session_state['Xa_asli3']
    d = float(st.session_state['d'])
    minXi = float(st.session_state['minXi'])

    best_model = None
    ket = None
    best_val_mape = float('inf')
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    # Uji 1
    if 'val_mape1' in st.session_state:
        val_mape1 = st.session_state["val_mape1"].values[0]
        if val_mape1 < best_val_mape:
            best_val_mape = val_mape1
            best_model = model1
            ket = "Backpropagation Standar Model Uji 1"
            X_train = X_train1
            X_test = X_test1
            y_train = y_train1
            y_test = y_test1
            Xa_asli = Xa_asli1

    # Uji 2
    if 'val_mape2' in st.session_state:
        val_mape2 = st.session_state["val_mape2"].values[0]
        if val_mape2 < best_val_mape:
            best_val_mape = val_mape2
            best_model = model2
            ket = "Backpropagation Standar Model Uji 2"
            X_train = X_train2
            X_test = X_test2
            y_train = y_train2
            y_test = y_test2
            Xa_asli = Xa_asli2

    # Uji 3
    if 'val_mape3' in st.session_state:
        val_mape3 = st.session_state["val_mape3"].values[0]
        if val_mape3 < best_val_mape:
            best_val_mape = val_mape3
            best_model = model3
            ket = "Backpropagation Standar Model Uji 3"
            X_train = X_train3
            X_test = X_test3
            y_train = y_train3
            y_test = y_test3
            Xa_asli = Xa_asli3

    # Uji 1
    if 'val_mapeNW1' in st.session_state:
        val_mapeNW1 = st.session_state["val_mapeNW1"].values[0]
        if val_mapeNW1 < best_val_mape:
            best_val_mape = val_mapeNW1
            best_model = modelNW1
            ket = "Backpropagation Nguyen-Widrow Model Uji 1"
            X_train = X_train1
            X_test = X_test1
            y_train = y_train1
            y_test = y_test1
            Xa_asli = Xa_asli1

    # Uji 2
    if 'val_mapeNW2' in st.session_state:
        val_mapeNW2 = st.session_state["val_mapeNW2"].values[0]
        if val_mapeNW2 < best_val_mape:
            best_val_mape = val_mapeNW2
            best_model = modelNW2
            ket = "Backpropagation Nguyen-Widrow Model Uji 2"
            X_train = X_train2
            X_test = X_test2
            y_train = y_train2
            y_test = y_test2
            Xa_asli = Xa_asli2

    # Uji 3
    if 'val_mapeNW3' in st.session_state:
        val_mapeNW3 = st.session_state["val_mapeNW3"].values[0]
        if val_mapeNW3 < best_val_mape:
            best_val_mape = val_mapeNW3
            best_model = modelNW3
            ket = "Backpropagation Nguyen-Widrow Model Uji 3"
            X_train = X_train3
            X_test = X_test3
            y_train = y_train3
            y_test = y_test3
            Xa_asli = Xa_asli3

    if best_model:
        st.write(f"Model terbaik: **{ket}**")
        st.write(f"Val MAPE terbaik: :green[" + f"{best_val_mape[0]}" + "]")
    else:
        st.write("Belum ada data model untuk dievaluasi")

    form_evaluasi = st.form('evaluasi')
    with form_evaluasi:
        st.write("Evaluasi model:")
        kolom1, kolom2, kolom3, kolom4 = st.columns(4)
        with kolom1:
            Kolom1 = st.number_input('Ketinggian wilayah (rentang 15-420):', 0)
            colom1 = (Kolom1 - Xa_asli.iloc[:, 0].min()) / (Xa_asli.iloc[:, 0].max() - Xa_asli.iloc[:, 0].min())
        with kolom2:
            Kolom2 = st.number_input('Fasilitas kesehatan (rentang 1-6):', 0)
            colom2 = (Kolom2 - Xa_asli.iloc[:, 1].min()) / (Xa_asli.iloc[:, 1].max() - Xa_asli.iloc[:, 1].min())
        with kolom3:
            Kolom3 = st.number_input('Kepadatan Penduduk (rentang 429-7465):', 0)
            colom3 = (Kolom3 - Xa_asli.iloc[:, 2].min()) / (Xa_asli.iloc[:, 2].max() - Xa_asli.iloc[:, 2].min())
        with kolom4:
            Kolom4 = st.number_input('Jumlah kasus di tahun sebelumnya:', 0)
            Kolom4 = np.log(Kolom4+0.5)
            colom4 = (Kolom4 - Xa_asli.iloc[:, 3].min()) / (Xa_asli.iloc[:, 3].max() - Xa_asli.iloc[:, 3].min())
        submitted = st.form_submit_button("Submit")

    if submitted:
        if 15 <= Kolom1 <= 420 and 1 <= Kolom2 <= 6 and 429 <= Kolom3 <= 7465:
            with conversion.localconverter(default_converter):
                def predict_best_model(best_model, input_data):
                    if best_model:
                        input_data_df = pd.DataFrame(input_data, columns=["X1", "X2", "X3", "X4"])
                        with localconverter(default_converter + pandas2ri.converter):
                            input_data_r = pandas2ri.py2rpy(input_data_df)
                        prediction_r = robjects.r.predict(best_model, input_data_r)
                        prediction = pandas2ri.rpy2py(prediction_r)
                    else:
                        prediction = None
                    return prediction

                prediction = predict_best_model(best_model, [[colom1, colom2, colom3, colom4]])

                if prediction:
                    prediction_value = prediction[0][-1]
                    predictlog = prediction_value * d + minXi
                    real_value = np.maximum(np.round(np.exp(predictlog)-0.5),0)
                    st.subheader("Hasil prediksi kasus DBD di wilayah tersebut:")
                    st.subheader(f":green[" + f"{real_value:.0f}" + " kasus]")
                    st.subheader("Keterangan:")
                    st.write(f"Prediksi (Normalisasi): **{prediction_value}**")
                    st.write(f"Prediksi (Denormalisasi(log)): **{predictlog}**")
                    st.write(f"Normalisasi Ketinggian Wilayah: **{colom1}**")
                    st.write(f"Normalisasi Fasilitas Kesehatan: **{colom2}**")
                    st.write(f"Normalisasi Kepadatan Penduduk: **{colom3}**")
                    st.write(f"Normalisasi Jumlah Kasus Tahun Sebelumnya: **{colom4}**")
        else:
            st.error("Mohon pastikan input berada dalam rentang yang valid!")
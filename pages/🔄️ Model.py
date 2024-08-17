import streamlit as st
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter

st.set_page_config (
    page_title="Halaman Pemodelan",
)
st.title('Halaman Model')

def run_model():
    # Load script R
    script = """
    library(neuralnet)
    library(readxl)
    library(feather)

    setwd("C:/Users/ahmad/Predict_DB/newwebpredict")
    kasus<-read_xlsx("banyumas.xlsx",sheet="kasus")
    kasus<-data.frame(kasus)
    faskesa<-read_xlsx("banyumas.xlsx",sheet="faskes")
    faskesa<-data.frame(faskesa)
    tinggia<-read_xlsx("banyumas.xlsx",sheet="tinggi")
    tinggia<-data.frame(tinggia)
    padata<-read_xlsx("banyumas.xlsx",sheet="padat")
    padata<-data.frame(padata)
    kasusdbd<-c()
    for(i in 3:13)
    {
    kasusdbd<-append(kasusdbd,kasus[,i])
    }
    length(kasusdbd)
    kasusdbd<-log(kasusdbd+0.5)
    #
    faskes<-c()
    for(i in 3:13)
    {
    faskes<-append(faskes,faskesa[,i])
    }
    length(faskes)
    #
    padat<-c()
    for(i in 3:13)
    {
    padat<-append(padat,padata[,i])
    }
    length(padat)
    tinggi<-c()
    for(i in 3:13)
    {
    tinggi<-append(tinggi,tinggia[,i])
    }
    length(tinggi)

    kasusdbd_1<-kasusdbd[1:270]   #(ytmin1)
    padat<-padat[28:297]
    faskes<-faskes[28:297]
    kasusdbd<-kasusdbd[28:297]
    tinggi<-tinggi[28:297]

    Xa<-data.frame(tinggi,faskes,padat,kasusdbd_1,kasusdbd)
    minXi<-min(Xa[,5])
    Xa_asli<-Xa
    d<-max(Xa[,5])-min(Xa[,5]) # d= max{Xi}-min{Xi}
    Xa[,1]<-(Xa[,1]-min(Xa[,1]))/(max(Xa[,1])-min(Xa[,1]))
    Xa[,2]<-(Xa[,2]-min(Xa[,2]))/(max(Xa[,2])-min(Xa[,2]))
    Xa[,3]<-(Xa[,3]-min(Xa[,3]))/(max(Xa[,3])-min(Xa[,3]))
    Xa[,4]<-(Xa[,4]-min(Xa[,4]))/(max(Xa[,4])-min(Xa[,4]))
    Xa[,5]<-(Xa[,5]-min(Xa[,5]))/(max(Xa[,5])-min(Xa[,5]))

    # PERSIAPAN MEMILAH DATA TRAINING DAN TESTING #
    n<-length(Xa[,1])
    ntraining<-n*0.85
    X_train<-Xa[1:ntraining,]
    X_tes<-Xa[(ntraining+1):n,]

    # PERSIAPAN VARIABEL DALAM MODEL #
    y<-X_train[,5]
    y_train<-data.frame(y)
    X1<-X_train[,1]
    X2<-X_train[,2]
    X3<-X_train[,3]
    X4<-X_train[,4]
    Dat<-data.frame(y,X1,X2,X3,X4)

    # PERSIAPAN NGUYEN-WIDROW #
    n=4;p=3 ; theta=0.7*p^(1/n)
    beta_awal=runif(n*p,-0.5,0.5)
    # untuk setiap neuron
    bobot=c()
    for (k in 1:p)
    {
        bj_nw=c()
        awal=(k-1)*n+1
        akhir=k*n
        for( j in 1:n)
        {
        eta_k=norm(beta_awal[awal:akhir],type="2")
        bj_nw=append(bj_nw,theta*beta_awal[j]/eta_k) 
        }
        bobot=append(bobot,bj_nw)
    }
    bias=runif(1,-theta,theta)
    beta_awal=c(1,beta_awal)
    beta_k=c(bias,bobot)

    # MODEL STANDAR DAN MODEL MODIF #
    nnstandar <- neuralnet(y~X1+X2+X3+X4, data=Dat,hidden = 3,threshold = 0.01,
                        stepmax = 1e+05, startweights = runif(12), linear.output = TRUE,
                        algorithm = "rprop+", err.fct = "sse",act.fct = "logistic")
    nnmodif <- neuralnet(y~X1+X2+X3+X4, data=Dat,hidden = 3,threshold = 0.01,
                        stepmax = 1e+05, startweights = beta_k, linear.output = TRUE,
                        algorithm = "rprop+", err.fct = "sse",act.fct = "logistic")

    X_test <- data.frame(X_tes[,1:4])
    y_test <- data.frame(X_tes[,5])
    pr.nstandar <- compute(nnstandar, X_test)
    pr.nmodif <- compute(nnmodif, X_test)

    # DENORMALISASI DATA AKTUAL DAN PREDIKSI
    pr.nnstandar <- pr.nstandar$net.result * d + minXi
    pr.nnmodif <- pr.nmodif$net.result * d + minXi
    y_akt <- (y_test) * d + minXi

    # MENGEMBALIKAN KE BILANGAN SEBELUM LOG
    standarreal <- round(exp(pr.nnstandar)-0.5, digits = 0)
    modifreal <- round(exp(pr.nnmodif)-0.5, digits = 0)
    yreal <- exp(y_akt)-0.5

    # HITUNG MAPE UNTUK MODELSTANDAR DAN MODIF
    MAPE_std <- sum(abs(y_akt-pr.nnstandar)/y_akt)
    MAPE_modif <- sum(abs(y_akt-pr.nnmodif)/y_akt)

    # Konversi dataframe R menjadi dataframe Python
    write_feather(X_train, "X_train.feather")
    write_feather(X_test, "X_test.feather")
    write_feather(y_train, "y_train.feather")
    write_feather(y_test, "y_test.feather")
    write_feather(y_akt, "y_akt.feather")
    write_feather(yreal, "yreal.feather")
    write_feather(Xa_asli, "Xa_asli.feather")
    write.csv(d, "d.csv",row.names = FALSE)
    write.csv(minXi, "minXi.csv",row.names = FALSE)
    write.csv(MAPE_std, "MAPE_std.csv",row.names = FALSE)
    write.csv(MAPE_modif, "MAPE_modif.csv",row.names = FALSE)
    write.csv(pr.nnstandar, "prednnstandar.csv",row.names = FALSE)
    write.csv(pr.nnmodif, "prednnmodif.csv",row.names = FALSE)
    write.csv(standarreal, "standarreal.csv",row.names = FALSE)
    write.csv(modifreal, "modifreal.csv",row.names = FALSE)

    # Simpan model standar dalam format RDS
    saveRDS(nnstandar, file = "nnstandar.rds")

    # Simpan model modifikasi dalam format RDS
    saveRDS(nnmodif, file = "nnmodif.rds")
    """
    try:
        with conversion.localconverter(default_converter):
            robjects.r(script)
    except Exception as e:
        st.error(f'Terjadi kesalahan saat menjalankan model: {e}')

def uji1():
    st.header("Uji 1")
    run_model()        
    with conversion.localconverter(default_converter):
        # Muat model standar dari file .rds ke dalam Python
        nnstandar = robjects.r['readRDS']("nnstandar.rds")
        # Muat model modifikasi dari file .rds ke dalam Python
        nnmodif = robjects.r['readRDS']("nnmodif.rds")

    X_train = pd.read_feather("X_train.feather")
    X_test = pd.read_feather("X_test.feather")
    y_train = pd.read_feather("y_train.feather")
    y_test = pd.read_feather("y_test.feather")
    y_akt = pd.read_feather("y_akt.feather")
    Xa_asli = pd.read_feather("Xa_asli.feather")
    yreal = pd.read_feather("yreal.feather")
    d = pd.read_csv("d.csv", header=None)[0][1]
    minXi = pd.read_csv("minXi.csv", header=None)[0][1]
    MAPE_std = pd.read_csv("MAPE_std.csv")
    MAPE_modif = pd.read_csv("MAPE_modif.csv")
    prednnstandar = pd.read_csv("prednnstandar.csv")
    prednnmodif = pd.read_csv("prednnmodif.csv")
    standarreal = pd.read_csv("standarreal.csv")
    modifreal = pd.read_csv("modifreal.csv")
    MAPE_std_str = MAPE_std.to_string(index=False, header=False)
    MAPE_modif_str = MAPE_modif.to_string(index=False, header=False)
    
    colom1,colom2 = st.columns(2)
    with colom1:
        st.subheader("Backpropagation Model Standar")
        st.write("MAPE")
        st.write(MAPE_std_str)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Nilai Aktual(LOG)")
            st.write(y_akt)
            st.write("Nilai Aktual")
            st.write(yreal)
        with col2:
            st.write("Nilai Prediksi(LOG)")
            st.write(prednnstandar)
            st.write("Nilai Prediksi")
            st.write(standarreal)

    with colom2:
        st.subheader("Backpropagation Nguyen-Widrow")
        st.write("MAPE")
        st.write(MAPE_modif_str)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Nilai Aktual(LOG)")
            st.write(y_akt)
            st.write("Nilai Aktual")
            st.write(yreal)
        with col2:
            st.write("Nilai Prediksi(LOG)")
            st.write(prednnmodif)
            st.write("Nilai Prediksi")
            st.write(modifreal)

    if 'xtrain1' not in st.session_state:
        st.session_state['xtrain1'] = X_train
    else:
        st.session_state['xtrain1'] = X_train
    if 'xtest1' not in st.session_state:
        st.session_state['xtest1'] = X_test
    else:
        st.session_state['xtest1'] = X_test
    if 'ytrain1' not in st.session_state:
        st.session_state['ytrain1'] = y_train
    else:
        st.session_state['ytrain1'] = y_train
    if 'ytest1' not in st.session_state:
        st.session_state['ytest1'] = y_test
    else:
        st.session_state['ytest1'] = y_test
    if 'model1' not in st.session_state:
        st.session_state['model1'] = nnstandar
    else:
        st.session_state['model1'] = nnstandar
    if 'modelNW1' not in st.session_state:
        st.session_state['modelNW1'] = nnmodif
    else:
        st.session_state['modelNW1'] = nnmodif
    if 'data_prediksi1' not in st.session_state:
        st.session_state['data_prediksi1'] = standarreal
    else:
        st.session_state['data_prediksi1'] = standarreal
    if 'data_prediksiNW1' not in st.session_state:
        st.session_state['data_prediksiNW1'] = modifreal
    else:
        st.session_state['data_prediksiNW1'] = modifreal
    if 'data_aktual1' not in st.session_state:
        st.session_state['data_aktual1'] = yreal
    else:
        st.session_state['data_aktual1'] = yreal
    if 'y_akt1' not in st.session_state:
        st.session_state['y_akt1'] = y_akt
    else:
        st.session_state['y_akt1'] = y_akt
    if 'val_mape1' not in st.session_state:
        st.session_state['val_mape1'] = MAPE_std
    else:
        st.session_state['val_mape1'] = MAPE_std
    if 'val_mapeNW1' not in st.session_state:
        st.session_state['val_mapeNW1'] = MAPE_modif
    else:
        st.session_state['val_mapeNW1'] = MAPE_modif
    if 'd' not in st.session_state:
        st.session_state['d'] = d
    else:
        st.session_state['d'] = d
    if 'minXi' not in st.session_state:
        st.session_state['minXi'] = minXi
    else:
        st.session_state['minXi'] = minXi
    if 'prednnstandar1' not in st.session_state:
        st.session_state['prednnstandar1'] = prednnstandar
    else:
        st.session_state['prednnstandar1'] = prednnstandar
    if 'prednnmodif1' not in st.session_state:
        st.session_state['prednnmodif1'] = prednnmodif
    else:
        st.session_state['prednnmodif1'] = prednnmodif
    if 'Xa_asli1' not in st.session_state:
        st.session_state['Xa_asli1'] = Xa_asli
    else:
        st.session_state['Xa_asli1'] = Xa_asli
    return X_train, X_test, y_train, y_test, MAPE_std, MAPE_modif, nnstandar, nnmodif, standarreal, modifreal, yreal, prednnstandar, prednnmodif, y_akt, d, minXi, Xa_asli


def uji2():
    st.header("Uji 2")
    run_model()        
    with conversion.localconverter(default_converter):
        # Muat model standar dari file .rds ke dalam Python
        nnstandar = robjects.r['readRDS']("nnstandar.rds")
        # Muat model modifikasi dari file .rds ke dalam Python
        nnmodif = robjects.r['readRDS']("nnmodif.rds")

    X_train = pd.read_feather("X_train.feather")
    X_test = pd.read_feather("X_test.feather")
    y_train = pd.read_feather("y_train.feather")
    y_test = pd.read_feather("y_test.feather")
    y_akt = pd.read_feather("y_akt.feather")
    yreal = pd.read_feather("yreal.feather")
    Xa_asli = pd.read_feather("Xa_asli.feather")
    d = pd.read_csv("d.csv", header=None)[0][1]
    minXi = pd.read_csv("minXi.csv", header=None)[0][1]
    MAPE_std = pd.read_csv("MAPE_std.csv")
    MAPE_modif = pd.read_csv("MAPE_modif.csv")
    prednnstandar = pd.read_csv("prednnstandar.csv")
    prednnmodif = pd.read_csv("prednnmodif.csv")
    standarreal = pd.read_csv("standarreal.csv")
    modifreal = pd.read_csv("modifreal.csv")
    MAPE_std_str = MAPE_std.to_string(index=False, header=False)
    MAPE_modif_str = MAPE_modif.to_string(index=False, header=False)

    colom1,colom2 = st.columns(2)
    with colom1:
        st.subheader("Backpropagation Model Standar")
        st.write("MAPE")
        st.write(MAPE_std_str)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Nilai Aktual(LOG)")
            st.write(y_akt)
            st.write("Nilai Aktual")
            st.write(yreal)
        with col2:
            st.write("Nilai Prediksi(LOG)")
            st.write(prednnstandar)
            st.write("Nilai Prediksi")
            st.write(standarreal)

    with colom2:
        st.subheader("Backpropagation Nguyen-Widrow")
        st.write("MAPE")
        st.write(MAPE_modif_str)

        col1,col2 = st.columns(2)
        with col1:
            st.write("Nilai Aktual(LOG)")
            st.write(y_akt)
            st.write("Nilai Aktual")
            st.write(yreal)
        with col2:
            st.write("Nilai Prediksi(LOG)")
            st.write(prednnmodif)
            st.write("Nilai Prediksi")
            st.write(modifreal)

    if 'xtrain2' not in st.session_state:
        st.session_state['xtrain2'] = X_train
    else:
        st.session_state['xtrain2'] = X_train
    if 'xtest2' not in st.session_state:
        st.session_state['xtest2'] = X_test
    else:
        st.session_state['xtest2'] = X_test
    if 'ytrain2' not in st.session_state:
        st.session_state['ytrain2'] = y_train
    else:
        st.session_state['ytrain2'] = y_train
    if 'ytest2' not in st.session_state:
        st.session_state['ytest2'] = y_test
    else:
        st.session_state['ytest2'] = y_test
    if 'model2' not in st.session_state:
        st.session_state['model2'] = nnstandar
    else:
        st.session_state['model2'] = nnstandar
    if 'modelNW2' not in st.session_state:
        st.session_state['modelNW2'] = nnmodif
    else:
        st.session_state['modelNW2'] = nnmodif
    if 'data_prediksi2' not in st.session_state:
        st.session_state['data_prediksi2'] = standarreal
    else:
        st.session_state['data_prediksi2'] = standarreal
    if 'data_prediksiNW2' not in st.session_state:
        st.session_state['data_prediksiNW2'] = modifreal
    else:
        st.session_state['data_prediksiNW2'] = modifreal
    if 'data_aktual2' not in st.session_state:
        st.session_state['data_aktual2'] = yreal
    else:
        st.session_state['data_aktual2'] = yreal
    if 'y_akt2' not in st.session_state:
        st.session_state['y_akt2'] = y_akt
    else:
        st.session_state['y_akt2'] = y_akt
    if 'val_mape2' not in st.session_state:
        st.session_state['val_mape2'] = MAPE_std
    else:
        st.session_state['val_mape2'] = MAPE_std
    if 'val_mapeNW2' not in st.session_state:
        st.session_state['val_mapeNW2'] = MAPE_modif
    else:
        st.session_state['val_mapeNW2'] = MAPE_modif
    if 'd' not in st.session_state:
        st.session_state['d'] = d
    else:
        st.session_state['d'] = d
    if 'minXi' not in st.session_state:
        st.session_state['minXi'] = minXi
    else:
        st.session_state['minXi'] = minXi
    if 'prednnstandar2' not in st.session_state:
        st.session_state['prednnstandar2'] = prednnstandar
    else:
        st.session_state['prednnstandar2'] = prednnstandar
    if 'prednnmodif2' not in st.session_state:
        st.session_state['prednnmodif2'] = prednnmodif
    else:
        st.session_state['prednnmodif2'] = prednnmodif
    if 'Xa_asli2' not in st.session_state:
        st.session_state['Xa_asli2'] = Xa_asli
    else:
        st.session_state['Xa_asli2'] = Xa_asli
    return X_train, X_test, y_train, y_test, MAPE_std, MAPE_modif, nnstandar, nnmodif, standarreal, modifreal, yreal, prednnstandar, prednnmodif, y_akt, d, minXi, Xa_asli


def uji3():
    st.header("Uji 3")
    run_model()
    with conversion.localconverter(default_converter):
        # Muat model standar dari file .rds ke dalam Python
        nnstandar = robjects.r['readRDS']("nnstandar.rds")
        # Muat model modifikasi dari file .rds ke dalam Python
        nnmodif = robjects.r['readRDS']("nnmodif.rds")
        
    X_train = pd.read_feather("X_train.feather")
    X_test = pd.read_feather("X_test.feather")
    y_train = pd.read_feather("y_train.feather")
    y_test = pd.read_feather("y_test.feather")
    y_akt = pd.read_feather("y_akt.feather")
    yreal = pd.read_feather("yreal.feather")
    Xa_asli = pd.read_feather("Xa_asli.feather")
    d = pd.read_csv("d.csv", header=None)[0][1]
    minXi = pd.read_csv("minXi.csv", header=None)[0][1]
    MAPE_std = pd.read_csv("MAPE_std.csv")
    MAPE_modif = pd.read_csv("MAPE_modif.csv")
    prednnstandar = pd.read_csv("prednnstandar.csv")
    prednnmodif = pd.read_csv("prednnmodif.csv")
    standarreal = pd.read_csv("standarreal.csv")
    modifreal = pd.read_csv("modifreal.csv")
    MAPE_std_str = MAPE_std.to_string(index=False, header=False)
    MAPE_modif_str = MAPE_modif.to_string(index=False, header=False)
    
    with conversion.localconverter(default_converter):
        # Muat model standar dari file .rds ke dalam Python
        nnstandar = robjects.r['readRDS']("nnstandar.rds")

        # Muat model modifikasi dari file .rds ke dalam Python
        nnmodif = robjects.r['readRDS']("nnmodif.rds")

        colom1,colom2 = st.columns(2)
        with colom1:
            st.subheader("Backpropagation Model Standar")
            st.write("MAPE")
            st.write(MAPE_std_str)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt)
                st.write("Nilai Aktual")
                st.write(yreal)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnstandar)
                st.write("Nilai Prediksi")
                st.write(standarreal)

        with colom2:
            st.subheader("Backpropagation Nguyen-Widrow")
            st.write("MAPE")
            st.write(MAPE_modif_str)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt)
                st.write("Nilai Aktual")
                st.write(yreal)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnmodif)
                st.write("Nilai Prediksi")
                st.write(modifreal)
    
        if 'xtrain3' not in st.session_state:
            st.session_state['xtrain3'] = X_train
        else:
            st.session_state['xtrain3'] = X_train
        if 'xtest3' not in st.session_state:
            st.session_state['xtest3'] = X_test
        else:
            st.session_state['xtest3'] = X_test
        if 'ytrain3' not in st.session_state:
            st.session_state['ytrain3'] = y_train
        else:
            st.session_state['ytrain3'] = y_train
        if 'ytest3' not in st.session_state:
            st.session_state['ytest3'] = y_test
        else:
            st.session_state['ytest3'] = y_test
        if 'model3' not in st.session_state:
            st.session_state['model3'] = nnstandar
        else:
            st.session_state['model3'] = nnstandar
        if 'modelNW3' not in st.session_state:
            st.session_state['modelNW3'] = nnmodif
        else:
            st.session_state['modelNW3'] = nnmodif
        if 'data_prediksi3' not in st.session_state:
            st.session_state['data_prediksi3'] = standarreal
        else:
            st.session_state['data_prediksi3'] = standarreal
        if 'data_prediksiNW3' not in st.session_state:
            st.session_state['data_prediksiNW3'] = modifreal
        else:
            st.session_state['data_prediksiNW3'] = modifreal
        if 'data_aktual3' not in st.session_state:
            st.session_state['data_aktual3'] = yreal
        else:
            st.session_state['data_aktual3'] = yreal
        if 'y_akt3' not in st.session_state:
            st.session_state['y_akt3'] = y_akt
        else:
            st.session_state['y_akt3'] = y_akt
        if 'val_mape3' not in st.session_state:
            st.session_state['val_mape3'] = MAPE_std
        else:
            st.session_state['val_mape3'] = MAPE_std
        if 'val_mapeNW3' not in st.session_state:
            st.session_state['val_mapeNW3'] = MAPE_modif
        else:
            st.session_state['val_mapeNW3'] = MAPE_modif
        if 'd' not in st.session_state:
            st.session_state['d'] = d
        else:
            st.session_state['d'] = d
        if 'minXi' not in st.session_state:
            st.session_state['minXi'] = minXi
        else:
            st.session_state['minXi'] = minXi
        if 'prednnstandar3' not in st.session_state:
            st.session_state['prednnstandar3'] = prednnstandar
        else:
            st.session_state['prednnstandar3'] = prednnstandar
        if 'prednnmodif3' not in st.session_state:
            st.session_state['prednnmodif3'] = prednnmodif
        else:
            st.session_state['prednnmodif3'] = prednnmodif
        if 'Xa_asli3' not in st.session_state:
            st.session_state['Xa_asli3'] = Xa_asli
        else:
            st.session_state['Xa_asli3'] = Xa_asli

    return X_train, X_test, y_train, y_test, MAPE_std, MAPE_modif, nnstandar, nnmodif, standarreal, modifreal, yreal, prednnstandar, prednnmodif, y_akt, d, minXi, Xa_asli

kol1,kol2 = st.columns(2)
with kol1:
    st.image("assets/grafikDBD.png")
with kol2:
    st.image("assets/petabanyumas.jpg")

if 'model_dijalankan' not in st.session_state:
    st.session_state['model_dijalankan'] = False

if not st.session_state['model_dijalankan']:
    st.subheader("Anda dapat mulai melakukan pengujian model dengan menekan tombol di bawah!")
    st.info("Model belum dijalankan!")
    if st.button('Jalankan Model Neural Network'):
        uji1()
        uji2()
        uji3()
        st.session_state['model_dijalankan'] = True
        st.success('Model Neural Network telah dijalankan!')

else:
    st.header("Hasil Pengujian Model Sebelumnya!")
    with conversion.localconverter(default_converter):
        st.header("Uji 1")
        MAPE_std1 = st.session_state["val_mape1"]
        MAPE_modif1 = st.session_state["val_mapeNW1"]
        MAPE_std_str1 = MAPE_std1.to_string(index=False, header=False)
        MAPE_modif_str1 = MAPE_modif1.to_string(index=False, header=False)
        y_akt1 = st.session_state["y_akt1"]
        y_akt1 = pd.DataFrame(y_akt1)
        yreal1 = st.session_state["data_aktual1"]
        yreal1 = pd.DataFrame(yreal1)
        prednnstandar1 = st.session_state["prednnstandar1"]
        prednnstandar1 = pd.DataFrame(prednnstandar1)
        standarreal1 = st.session_state["data_prediksi1"]
        standarreal1 = pd.DataFrame(standarreal1)
        prednnmodif1 = st.session_state["prednnmodif1"]
        prednnmodif1 = pd.DataFrame(prednnmodif1)
        modifreal1 = st.session_state["data_prediksiNW1"]
        modifreal1 = pd.DataFrame(modifreal1)

        colom1,colom2 = st.columns(2)
        with colom1:
            st.subheader("Backpropagation Model Standar")
            st.write("MAPE")
            st.write(MAPE_std_str1)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt1)
                st.write("Nilai Aktual")
                st.write(yreal1)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnstandar1)
                st.write("Nilai Prediksi")
                st.write(standarreal1)

        with colom2:
            st.subheader("Backpropagation Nguyen-Widrow")
            st.write("MAPE")
            st.write(MAPE_modif_str1)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt1)
                st.write("Nilai Aktual")
                st.write(yreal1)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnmodif1)
                st.write("Nilai Prediksi")
                st.write(modifreal1)

        st.header("Uji 2")
        MAPE_std2 = st.session_state["val_mape2"]
        MAPE_modif2 = st.session_state["val_mapeNW2"]
        MAPE_std_str2 = MAPE_std2.to_string(index=False, header=False)
        MAPE_modif_str2 = MAPE_modif2.to_string(index=False, header=False)
        y_akt2 = st.session_state["y_akt2"]
        y_akt2 = pd.DataFrame(y_akt2)
        yreal2 = st.session_state["data_aktual2"]
        yreal2 = pd.DataFrame(yreal2)
        prednnstandar2 = st.session_state["prednnstandar2"]
        prednnstandar2 = pd.DataFrame(prednnstandar2)
        standarreal2 = st.session_state["data_prediksi2"]
        standarreal2 = pd.DataFrame(standarreal2)
        prednnmodif2 = st.session_state["prednnmodif2"]
        prednnmodif2 = pd.DataFrame(prednnmodif2)
        modifreal2 = st.session_state["data_prediksiNW2"]
        modifreal2 = pd.DataFrame(modifreal2)

        colom1,colom2 = st.columns(2)
        with colom1:
            st.subheader("Backpropagation Model Standar")
            st.write("MAPE")
            st.write(MAPE_std_str2)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt2)
                st.write("Nilai Aktual")
                st.write(yreal2)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnstandar2)
                st.write("Nilai Prediksi")
                st.write(standarreal2)

        with colom2:
            st.subheader("Backpropagation Nguyen-Widrow")
            st.write("MAPE")
            st.write(MAPE_modif_str2)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt2)
                st.write("Nilai Aktual")
                st.write(yreal2)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnmodif2)
                st.write("Nilai Prediksi")
                st.write(modifreal2)

        st.header("Uji 3")
        MAPE_std3 = st.session_state["val_mape3"]
        MAPE_modif3 = st.session_state["val_mapeNW3"]
        MAPE_std_str3 = MAPE_std3.to_string(index=False, header=False)
        MAPE_modif_str3 = MAPE_modif3.to_string(index=False, header=False)
        y_akt3 = st.session_state["y_akt3"]
        y_akt3 = pd.DataFrame(y_akt3)
        yreal3 = st.session_state["data_aktual3"]
        yreal3 = pd.DataFrame(yreal3)
        prednnstandar3 = st.session_state["prednnstandar3"]
        prednnstandar3 = pd.DataFrame(prednnstandar3)
        standarreal3 = st.session_state["data_prediksi3"]
        standarreal3 = pd.DataFrame(standarreal3)
        prednnmodif3 = st.session_state["prednnmodif3"]
        prednnmodif3 = pd.DataFrame(prednnmodif3)
        modifreal3 = st.session_state["data_prediksiNW3"]
        modifreal3 = pd.DataFrame(modifreal3)

        colom1,colom2 = st.columns(2)
        with colom1:
            st.subheader("Backpropagation Model Standar")
            st.write("MAPE")
            st.write(MAPE_std_str3)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt3)
                st.write("Nilai Aktual")
                st.write(yreal3)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnstandar3)
                st.write("Nilai Prediksi")
                st.write(standarreal3)

        with colom2:
            st.subheader("Backpropagation Nguyen-Widrow")
            st.write("MAPE")
            st.write(MAPE_modif_str3)

            col1,col2 = st.columns(2)
            with col1:
                st.write("Nilai Aktual(LOG)")
                st.write(y_akt3)
                st.write("Nilai Aktual")
                st.write(yreal3)
            with col2:
                st.write("Nilai Prediksi(LOG)")
                st.write(prednnmodif3)
                st.write("Nilai Prediksi")
                st.write(modifreal3)
    if st.button('Jalankan Kembali Model Neural Network'):
        st.session_state['data_aktual1'] = None
        st.session_state['y_akt1'] = None
        st.session_state['val_mape1'] = None
        st.session_state['val_mapeNW1'] = None
        st.session_state['prednnstandar1'] = None
        st.session_state['prednnmodif1'] = None
        st.session_state['data_prediksi1'] = None
        st.session_state['data_prediksiNW1'] = None
        st.session_state['data_aktual2'] = None
        st.session_state['y_akt2'] = None
        st.session_state['val_mape2'] = None
        st.session_state['val_mapeNW2'] = None
        st.session_state['prednnstandar2'] = None
        st.session_state['prednnmodif2'] = None
        st.session_state['data_prediksi2'] = None
        st.session_state['data_prediksiNW2'] = None
        st.session_state['data_aktual3'] = None
        st.session_state['y_akt3'] = None
        st.session_state['val_mape3'] = None
        st.session_state['val_mapeNW3'] = None
        st.session_state['prednnstandar3'] = None
        st.session_state['prednnmodif3'] = None
        st.session_state['data_prediksi3'] = None
        st.session_state['data_prediksiNW3'] = None
        uji1()
        uji2()
        uji3()
        st.success('Model Neural Network telah dijalankan!')
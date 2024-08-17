import streamlit as st 

st.set_page_config (
    page_title="Model Prediksi",
)

st.title("Aplikasi Perbandingan Model Prediksi")
st.subheader("Pemodelan Prediksi dengan menggunakan Implementasi :green[Nguyen-Widrow] pada :green[Backpropagation Neural Network]")

st.markdown(
    """
    Aplikasi web ini merupakan aplikasi untuk melihat pengaruh dari algoritma :green[Nguyen-Widrow] dalam melakukan prediksi pada kasus tertentu.
    Pemodelan dilakukan dengan menggunakan bahasa pemrograman R dan divisualisasikan dengan menggunakan Framework GUI Streamlit berbasis bahasa python.
    
    Anda dapat melakukan pengujian model pada halaman model di samping. Output dari pemodelan tersebut akan langsung membandingkan bagaimana hasil dari model
    Backpropagation Standar dan juga Backpropagation dengan :green[Nguyen-Widrow]
    """
)

st.info(
    """
    ### ❕Disclaimer
    Aplikasi ini merupakan hasil tugas akhir saya yang mana lebih :red[berfokus untuk membandingkan model] dari :green[Backpropagation Standar] dan :green[Backpropagation Nguyen-Widrow]

    **Rizqi Ahmad Fauzan**
    """
)
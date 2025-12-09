import joblib
import streamlit as st

model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar Publik")
st.write("Apliakasi ini dibuat menggunakan Teknologi NLP dengan memanfaatkan model machine learning logistic legression")
input = st.text_input("Masukan Komentar Anda")
if st.button("Submit"):
    if input.strip() == "":
        st.warning("komentar tidak boleh kosong")
    else: 
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "negatif",
            1: "positif"
        }
        st.subheader("Hasil Analisis")
        st.write("**komentar : **", label_map.get(prediksi, prediksi))
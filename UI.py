import streamlit as st

st.title("Hello World! 👋")

st.write("Das ist meine erste Streamlit App.")

# Interaktives Element
name = st.text_input("Wie heißt du?")

if name:
    st.write(f"Hallo, {name}! 🎉")

# Button
if st.button("Klick mich!"):
    st.balloons()
from main import RAG
import streamlit as st

st.title('Webbase Rag application')
url=st.text_input(label='give the url')
prompt=st.text_input(label='what is your answer')
button=st.button(label='onclick')

if url and prompt and button:
    rag=RAG(url)
    answer=rag.rag(prompt)
    st.success(answer['content'])
    st.balloons()

    

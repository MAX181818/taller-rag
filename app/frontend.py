import os, requests, streamlit as st
st.set_page_config(page_title='RAG Milvus', layout='wide')
st.title('Laboratorio RAG · Milvus + LLM local')
st.caption('Corpus de noticias y Bluesky; contraste siempre las fuentes originales.')
model=st.selectbox('LLM local', ['qwen2.5:1.5b','llama3.2:1b','llama3.2:3b','gemma2:2b'])
k=st.slider('Top-k',1,15,5)
question=st.text_area('Pregunta', '¿Qué se afirma en las fuentes sobre el tema investigado?')
if st.button('Consultar'):
    try:
        r=requests.post(os.getenv('API_URL','http://localhost:8000')+'/ask',json={'query':question,'model':model,'k':k},timeout=260)
        r.raise_for_status(); out=r.json()
        st.subheader('Respuesta');st.write(out['answer'])
        st.caption(f"Modelo: {out['model']} | Latencia: {out['latency_seconds']} s")
        st.subheader('Evidencias')
        for i, c in enumerate(out['evidence'],1):
            with st.expander(f"[{i}] {c['document_id']} · score={c['score']:.3f}"):
                st.write(c['text']);st.write(c.get('url',''))
    except Exception as e: st.error(str(e))

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone as PineconeClient

load_dotenv()
st.set_page_config(page_title="AI PDF Chatbot")
st.header("Chat with your Knowledge Base")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


@st.cache_resource
def get_context(query_text):
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))


    query_vector = embeddings.embed_query(query_text)


    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )


    return "\n".join([match['metadata']['text'] for match in results['matches']])



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                context = get_context(prompt)


                full_prompt = f"""
                You are a helpful assistant. Use the provided context to answer the question.
                If the answer isn't in the context, say you don't know based on the documents.

                Context: {context}

                Question: {prompt}

                Answer:"""


                response = llm.invoke(full_prompt)
                answer = response.content
                st.markdown(answer)

            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
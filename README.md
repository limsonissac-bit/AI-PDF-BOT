# AIchatbot
# AI PDF Chatbot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions based on the content. Built with **Google Gemini 3.1** and **Pinecone**.

##  Features
- **Semantic Search:** Uses Google `gemini-embedding-001` to convert text into 3072-dimensional vectors.
- **Vector Storage:** Pinecone serverless index for lightning-fast document retrieval.
- **LLM Integration:** Powered by Gemini 3.1 Flash for accurate, context-aware answers.
- **User Interface:** Clean, interactive chat UI built with Streamlit.

##  Tech Stack
- **Language:** Python 3.14
- **AI/LLM:** LangChain, Google Generative AI
- **Database:** Pinecone
- **Frontend:** Streamlit

##  Setup & Installation

### 1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/AI-PDF-Chatbot.git](https://github.com/YOUR_USERNAME/AI-PDF-Chatbot.git)
cd AIchatbot

### 2. Configure Environment Variables

### 3. Install dependencies 

pip install -r requirements.txt


### HOW TO USE:
1. Place your PDFs in a folder named data/ and run:


python ingest.py

2. Launch Chatbot using:

streamlit run main.py










# Mini RAG - Retrieval Augmented Generation (Top Rated Wines Dataset)

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using a dataset of top-rated wines. It leverages:

- Sentence Transformers to create semantic embeddings of wine descriptions,
- Qdrant as a vector store to perform similarity search,
- Llamafile (Phi-2 model) as a local LLM to generate contextual, helpful responses.

# Goal

To enable users to query the system for wine recommendations and get natural-language answers that combine semantic search with LLM reasoning.

# Dataset

I used a CSV file containing various wine attributes.

Key fields include:

- "name"
- "region"
- "variety"
- "rating"
- "notes"

# Tools Used

- "sentence-transformers": for embedding text 
- "qdrant-client": for storing and searching embeddings
- "openai": Python client to communicate with a local LLM via API (Llamafile)
- "llamafile": local inference of Phi-2, served over http://127.0.0.1:8080/

# How to Run

1. Clone the repo and install the dependencies:
   ***bash
   pip install requirements.txt

2. Start your Llamafile model (i used "phi-2.Q4_0.llamafile") via:
./phi-2.Q4_0.llamafile.exe

3. Open the notebook and run all cells to:

Load and preprocess the wine data
Create embeddings and populate Qdrant
Query wine recommendations

Made with ❤️ as part of my learning in applied machine learning.








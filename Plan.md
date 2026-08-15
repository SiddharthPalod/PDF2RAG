Load into Vector DB: The Vector Database indexes these embeddings for hyper-fast semantic search.

Phase 2: Retrieval & Generation (Runtime)

This phase happens when a user asks your application a question.

Embed the Query: The user inputs a question (e.g., "What does the diagram on page 4 show?"). You pass this exact query through the same model (all-MiniLM-L6-v2) to get a 384-dimensional query vector.

Vector Search: The Vector DB compares the query vector against all stored vectors (using Cosine Similarity) and retrieves the "Top K" (e.g., top 5) most relevant chunks.

Context Assembly: You combine the text of those retrieved chunks into a single prompt block.

LLM Generation: You send a prompt to an LLM (like GPT-4, Llama 3, or Claude) structured like: "Answer the user's question using ONLY the following context. Context: [Inserted Top K Chunks]. Question: [User Query]". The LLM then generates the final answer.
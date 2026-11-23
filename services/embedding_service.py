from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    def __init__(self):
        self.model=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    def embed_documents(self, all_chunks):
        embeddings = self.model.embed_documents(all_chunks)
        return embeddings
    def embed_query(self,embed_questions):
        return self.model.embed_query(embed_questions)
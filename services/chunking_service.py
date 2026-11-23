
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:

    def __init__(self):
        self.all_chunks = []
        self.all_metadata = []
        self.all_ids = []

        self.splitter = self.get_splitter()

    def get_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def split_text(self, all_documents):

        for document in all_documents:

            chunks = self.splitter.split_text(
                document["text"]
            )

            for index, chunk in enumerate(chunks):

                self.all_chunks.append(chunk)

                self.all_metadata.append({
                    "source": document["source"],
                    "chunk_index": index
                })

                self.all_ids.append(
                    f'{document["source"]}_chunk_{index}'
                )

        print("Total chunks:", len(self.all_chunks))

        return (
            self.all_chunks,
            self.all_metadata,
            self.all_ids
        )

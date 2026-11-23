
from pathlib import Path
from pypdf import PdfReader


class LoadPDFService:

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.pdf_dir = self.base_dir / "document"

        self.pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not self.pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in: {self.pdf_dir}"
            )

    def load_pdfs(self):

        all_documents = []

        for pdf_path in self.pdf_files:

            reader = PdfReader(str(pdf_path))

            pages = []

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

            document = "\n".join(pages)

            if not document.strip():

                print(f"Skipping empty PDF: {pdf_path.name}")
                continue

            all_documents.append({
                "source": pdf_path.name,
                "text": document
            })

            print(f"Loaded: {pdf_path.name}")

        if not all_documents:
            raise ValueError(
                "No readable text found in the PDFs."
            )

        print("Total PDFs loaded:", len(all_documents))

        return all_documents

import pymupdf
from pathlib import Path

def load_pdf(file_path):
    path = Path(file_path)

    #check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not path.suffix.lower() == ".pdf":
        raise ValueError("The file is not a PDF.")
    if not path.is_file():
        raise ValueError(f"The path is not a file: {file_path}")
    print(f"Loading PDF file: {file_path}")

    # Load the PDF
    doc = pymupdf.open(file_path)
    print("Pages: ", len(doc))

    #Extract file name

    file_name = path.name

    #Extract metadata

    metadata = doc.metadata

    pages = []

    for page_number, page in enumerate(doc, start = 1):
        text = page.get_text()

        text_data = {
            "page_number": page_number,
            "text": text
        }

        pages.append(text_data)

    #Structured data

    return {
        "file_name": file_name,
        "metadata": metadata,
        "pages": pages
    }
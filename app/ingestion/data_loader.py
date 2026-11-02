import fitz
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
    doc = fitz.open(file_path)
    print("Pages: ", len(doc))

    #Extract text from each page

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        print(f"Page {page_number}, {text}")  # Print first 100 characters of each page

    doc.close()
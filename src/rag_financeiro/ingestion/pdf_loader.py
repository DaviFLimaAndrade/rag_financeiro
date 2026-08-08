from docling.document_converter import DocumentConverter

def load_document(pdf_path: str):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document  # DoclingDocument
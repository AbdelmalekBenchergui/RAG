import tempfile
from unstructured.partition.pdf import partition_pdf


def parse_pdf_bytes(file_bytes: bytes, filename: str):
    text_parts = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file.flush()
        elements = partition_pdf(filename=tmp_file.name, strategy="auto")
        for element in elements:
            text = " ".join(str(element).split())
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts), {"filename": filename, "type": "pdf"}

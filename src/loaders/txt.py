def parse_txt_bytes(file_bytes: bytes, filename: str):
    text = file_bytes.decode("utf-8")
    return text, {"filename": filename, "type": "txt"}

from bs4 import BeautifulSoup


def parse_html_bytes(file_bytes: bytes, filename: str):
    soup = BeautifulSoup(file_bytes, "html.parser")
    for junk in soup(["script", "style", "meta", "nav", "footer", "header"]):
        junk.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text, {"filename": filename, "type": "html"}

import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

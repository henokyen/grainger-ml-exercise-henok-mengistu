import yaml
from bs4 import BeautifulSoup
import re

def load_config(path: str):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict["parameters"]

def clean_html(raw_html):
    if not raw_html:
        return ''
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

# --- Normalize text (whitespace, casing, etc.) ---
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def is_useful_text(text, min_length=3):
    if not text or text.strip() == "":
        return False
    stripped = re.sub(r'[^\w]', '', text)
    return not stripped.isdigit() and len(text.strip()) >= min_length

# --- Combine all cleaning steps ---
def clean_and_filter(text):
    text = clean_html(text)
    text = normalize_text(text)
    return text if is_useful_text(text) else ''

import yaml
from langchain_community.vectorstores import Chroma

def load_links_from_file(file_path):
    links = []
    with open(file_path, 'r') as file:
        for line in file:
            link = line.strip()  # Remove leading/trailing whitespaces and newline characters
            if link:  # Ensure the line is not empty
                links.append(link)
    return links


def load_config():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_Chromadb(embedding_function):
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

import yaml
from langchain_community.document_loaders import WebBaseLoader
import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

config = utils.load_config()

links = utils.load_links_from_file('doclinks.txt')
loader = WebBaseLoader(links)
docs = loader.load()

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name=config["Embedding_model_path"],     # Provide the pre-trained model's path
    model_kwargs=config["model_kwargs"], # Pass the model configuration options
    encode_kwargs=config["encode_kwargs"] # Pass the encoding options
)

Chroma.from_documents(docs, embeddings, persist_directory = "./chroma_db")


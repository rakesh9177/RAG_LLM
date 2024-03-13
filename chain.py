from langchain import HuggingFacePipeline
import utils
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

config = utils.load_config()
'''
tokenizer = AutoTokenizer.from_pretrained(config["llm_model_name"], padding="max_length", truncation=True, model_max_length=512)

question_answerer = pipeline(
    "text-generation", 
    model=config["llm_model_name"], 
    tokenizer=tokenizer,
    return_tensors='pt',
    max_new_tokens = 200
)

llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 2048},
)
'''
llm = HuggingFacePipeline.from_model_id(
    model_id=config["llm_model_id"],
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000},
)


embeddings = HuggingFaceEmbeddings(
    model_name=config["Embedding_model_path"],     # Provide the pre-trained model's path
    model_kwargs=config["model_kwargs"], # Pass the model configuration options
    encode_kwargs=config["encode_kwargs"] # Pass the encoding options
)

db = utils.get_Chromadb(embedding_function=embeddings)

retriever = db.as_retriever()

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever= db.as_retriever(),
    memory=memory
)

def answer_question(question):
    #result = qa.run({"query": question})
    result = conversation_chain.invoke(question)
    return result["answer"]

demo = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="LangSmith documentation Chatbot",
    description="Ask a question on documentation of LangSmith"
)
demo.launch()
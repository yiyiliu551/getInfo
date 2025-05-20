!pip install -q langchain transformers accelerate sentence-transformers InstructorEmbedding faiss-cpu
!pip install -U langchain-community
from google.colab import drive
drive.mount('/content/drive')
import os
import glob

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


fs_path = "/content/drive/MyDrive/eiression/data/react"
all_docs = []
for file_path in glob.glob(os.path.join(fs_path, "**/*.*"), recursive=True):
    if file_path.endswith((".md", ".txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(page_content=text, metadata={"source": file_path})
        all_docs.append(doc)

raw_docs = all_docs

# this use gpt
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(raw_docs)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
#use similarity to get the
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

!huggingface-cli login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

#use  llama2
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=base_model,
  tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=pipe)

!pip install -q gradio

import gradio as gr

def get_answer_with_prompt(query: str) -> str:

    relevant_docs = retriever.get_relevant_documents(query)

    prompt = f"""Based solely on the retrieved documents, answer the question.
It only needs to rephrase sentences based on the retrieved articles, without any additional complicated elements.
Question:
{query}

Answer:"""

    r_output = llm(prompt)
    # get result
    if "Answer:" in r_output:
        processed_answer = r_output.split("Answer:", 1)[1].strip()
    else:
        processed_answer = r_output.strip()

    citations = ""
    for idx, doc in enumerate(relevant_docs, start=1):
        # reffer in the doc
        source = doc.metadata.get("source", "Unknown Source")
        citations += "{}. Source: {}\n".format(idx, source)


    final_output = "Answer:\n" + processed_answer+" \n\nrefference:\n"+ citations
    return final_output
#this use gpt
iface = gr.Interface(
    fn=get_answer_with_prompt,
    inputs=gr.components.Textbox(lines=2, placeholder="Please enter your question..."),
    outputs=gr.components.Textbox(lines=10),
    title="Prompt-based Question Answering System",
    description="Enter a question, and the system will generate an answer along with citation information (sourced from your folder)."
)

iface.launch()






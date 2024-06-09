from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PDFPlumberLoader
import re


# Function to process in batches
def process_in_batches(data, batch_size, is_query=False):
    if is_query:
        batch_dict = embed_tokenizer(data, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = embed_model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0].cpu()
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy().flatten(), normalized_embeddings.numpy().flatten()
    else:
        embeddings_list = []
        normalized_embeddings_list = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_dict = embed_tokenizer(batch_data, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = embed_model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0].cpu()
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_list.append(embeddings)
                normalized_embeddings_list.append(normalized_embeddings)
        return torch.cat(embeddings_list, dim=0), torch.cat(normalized_embeddings_list, dim=0)

# Function to combine context into a single string
def combine_context(context_list):
    return "\n".join(context_list)

# Function for sentence transformer similarity
def sentence_transformer_similarity(query):
    model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    query_embedding = model.encode(query)
    passage_embeddings = model.encode(data)
    similarity = np.dot(passage_embeddings, query_embedding)
    return similarity

# Load models and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_model_path = '/content/drive/MyDrive/HuggingFaceModels/'
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
embed_model = AutoModel.from_pretrained(embed_model_path, trust_remote_code=True, token="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")
embed_model.to(device)

llm_model = AutoModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/HuggingFaceLLM",
    device_map="cuda",
    torch_dtype="auto",
    load_in_4bit=True,
    trust_remote_code=True,
)
llm_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/HuggingFaceLLM")

#data loading
loader = PDFPlumberLoader("/content/NVIDIAAn.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)
data = [splits[i].page_content for i in range(len(splits))]

embeddings, normalized_embeddings = process_in_batches(data, batch_size=2, is_query=False)

torch.cuda.empty_cache()
df1 = pd.DataFrame(embeddings)
df2 = pd.DataFrame(normalized_embeddings)
df1["text"] = data
df2["text"] = data

embeddings = np.vstack(embeddings).astype(np.float32)
normalized_embeddings = np.vstack(normalized_embeddings).astype(np.float32)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
res = faiss.StandardGpuResources()
normalized_dimension = normalized_embeddings.shape[1]
index_cos = faiss.GpuIndexFlatIP(res, normalized_dimension)
index.add(embeddings)
index_cos.add(normalized_embeddings)

def generate_queries(query, num_queries=4):
    prompt = """Example Flow:
Original query: "What is NVIDIA's primary business focus?"
Generated variations:
1.What core industry does NVIDIA predominantly operate in?
2.What is the main area of expertise for NVIDIA in the technology sector?
3.Which market segment is at the heart of NVIDIA's business operations?
4.What primary products or services define NVIDIA's business strategy?"""
    messages_euclidean = [{"role": "system", "content":f"As a search assistant, your task is to generate 4 different variations of the following search query to aid in comprehensive semantic retrieval. Each variation should focus on slightly different aspects or rephrasing of the original query to ensure diverse retrieval results. Please provide each variation on a new line without any additional text or numbering.\n{prompt}"},
                          {"role": "user", "content": f"User Question:{query} now give me generate 4 different variations of the following search query "},]
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
    )
    generation_args = {
        "max_new_tokens": 400,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    generated_queries = pipe(messages_euclidean, **generation_args)[0]['generated_text']
    torch.cuda.empty_cache()
    lines = generated_queries.split("\n")
    question_pattern = re.compile(r'^\d+\.\s*(.*)')
    questions = [line.strip() for line in lines if question_pattern.match(line.strip())]
    return questions


def search_and_generate(query):
    query_embedding, normalized_query_embeddings = process_in_batches([query], batch_size=1, is_query=True)

    # Euclidean distance search
    distance, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), k=4)
    context_list_euclidean = df2.iloc[indices[0]]["text"].tolist()
    context_euclidean = combine_context(context_list_euclidean)
    indices_euclidean = indices[0].tolist()
    prompt_rags = """Prompt:-Given the following 'Context' generate a detailed response to the 'User Question."""
    messages_euclidean = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": f"{prompt_rags}\n\n\n'Context':\n\n{context_euclidean}\n\n\n 'User Question':\n{query}"},
    ]
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
    )
    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output_euclidean = pipe(messages_euclidean, **generation_args)[0]['generated_text']
    torch.cuda.empty_cache()

    # Cosine similarity search
    cos_sim, cos_indices = index_cos.search(normalized_query_embeddings.reshape(1, -1).astype(np.float32), k=4)
    context_list_cosine = df2.iloc[cos_indices[0]]["text"].tolist()
    context_cosine = combine_context(context_list_cosine)
    indices_cosine = cos_indices[0].tolist()
    #If the context does not contain the necessary information to answer the question, acknowledge that and provide a response based on your general knowledge.
    messages_cosine = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": f"{prompt_rags}\n\n\n'Context':\n{context_cosine}\n\n\n\n'User Question':\n{query}"},
    ]
    output_cosine = pipe(messages_cosine, **generation_args)[0]['generated_text']
    torch.cuda.empty_cache()

    # Sentence transformer similarity
    similarity = sentence_transformer_similarity(query)
    top4_values, top4_indices = torch.topk(torch.tensor(similarity), 4)
    context_list_sentence = df2.iloc[top4_indices]["text"].tolist()
    context_sentence = combine_context(context_list_sentence)
    indices_sentence = top4_indices.tolist()
    messages_sentence = [
        {"role": "system", "content": "You are a helpful AI assistant. Given the following 'context', generate a detailed response to the 'User Question"},
        {"role": "user", "content": f"{prompt_rags}\n\n\nContext:\n{context_sentence}\n\n\n\n'User Question':\n{query}"},
    ]
    output_sentence_transformer = pipe(messages_sentence, **generation_args)[0]['generated_text']
    torch.cuda.empty_cache()

     # MultiQueryRetriever (cosine similarity only)
    generated_queries = generate_queries(query)
    all_contexts = []
    all_indices = []

    for generated_query in generated_queries:
        norm_query_embedding, norm_query_embeddings = process_in_batches([generated_query], batch_size=1, is_query=True)
        cos_sim, cos_indices = index_cos.search(norm_query_embeddings.reshape(1, -1).astype(np.float32), k=4)
        all_contexts.extend(df2.iloc[cos_indices[0]]["text"].tolist())
        all_indices.extend(cos_indices[0].tolist())
        torch.cuda.empty_cache()

    # Removing duplicates
    unique_indices = list(set(all_indices))
    unique_contexts = [df2.iloc[i]["text"] for i in unique_indices]

    combined_multi_query_context = combine_context(unique_contexts)
    messages_multi_query = [
        {"role": "system", "content": "You are a helpful AI assistant. Given the following 'context', generate a detailed response to the 'User Question"},
        {"role": "user", "content": f"{prompt_rags}\n\n\nContext:\n{combined_multi_query_context}\n\n\n\n'User Question':{query}"},

    ]
    output_multi_query = pipe(messages_multi_query, **generation_args)[0]['generated_text']
    torch.cuda.empty_cache()

    return (output_euclidean,
            f"Context:\n{context_euclidean}\n\nIndices:\n{indices_euclidean}",
            output_cosine,
            f"Context:\n{context_cosine}\n\nIndices:\n{indices_cosine}",
            output_sentence_transformer,
            f"Context:\n{context_sentence}\n\nIndices:\n{indices_sentence}",
            output_multi_query,
            f"quries :-\n{generated_queries}Context:\n{combined_multi_query_context}\n\nIndices:\n{unique_indices}")

# Gradio interface
iface = gr.Interface(
    fn=search_and_generate,
    inputs="text",
    outputs=[
        gr.Textbox(label="Output based on Euclidean distance"),
        gr.Textbox(label="Euclidean Context and Indices"),
        gr.Textbox(label="Output based on cosine similarity"),
        gr.Textbox(label="Cosine Context and Indices"),
        gr.Textbox(label="Output based on sentence transformer similarity"),
        gr.Textbox(label="Sentence Transformer Context and Indices"),
        gr.Textbox(label="Output based on MultiQueryRetriever (cosine similarity)"),
        gr.Textbox(label="MultiQueryRetriever Context and Indices"),
    ],
    title="Chatbot with Multiple Outputs",
    description="Enter a query and get four different responses based on different similarity measures, along with context and indices."
)

# Launch the interface
iface.launch(debug=True)
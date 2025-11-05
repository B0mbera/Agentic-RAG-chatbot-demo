"""
rag agent chatbot demo - autonom döntésekkel - kamu llm-mel
"""

import os
import sys
from typing import Dict, List, Any, TypedDict
from pathlib import Path

import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from llama_cpp import Llama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

#beallitasok

class Config:
    KNOWLEDGE_DIR = "./knowledge"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    VECTOR_DB_PATH = "./chroma_db_pdf"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_DOCS = 5
 #opcionalis   LLM_MODEL_PATH = "./models/llama-2-7b-chat.gguf"
    USE_LOCAL_LLM = False
    USE_MOCK_LLM = True
    TEMPERATURE = 0.7
    MAX_TOKENS = 512


config = Config()


class KamuLLM:
    def __init__(self, temperature=0.7, max_tokens=512):
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def __call__(self, prompt, **kwargs):
        if "Kontextus:" in prompt:
            lines = prompt.split('\n')
            context_lines = []
            question = ""
            
            in_context = False
            for line in lines:
                if "Kontextus:" in line:
                    in_context = True
                elif "Kerdes:" in line:
                    in_context = False
                    question = line.split(":", 1)[-1].strip() if ":" in line else ""
                elif in_context and line.strip():
                    context_lines.append(line.strip())
            
            if context_lines:
                context_preview = ' '.join(context_lines[:2])[:250]
                return f"A dokumentumok alapjan: {context_preview}... (Mock LLM - config.USE_MOCK_LLM = False a valodi LLM-hez)"
        
        return "Altalanos valasz. (Mock mod aktiv)"
    
    def invoke(self, prompt, **kwargs):
        return self.__call__(prompt, **kwargs)


def load_pdf_documents(knowledge_dir: str) -> List[Document]:
    if not os.path.exists(knowledge_dir):
        print(f"Error: Directory not found: {knowledge_dir}")
        return []
    
    pdf_files = list(Path(knowledge_dir).glob("**/*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files in: {knowledge_dir}")
        return []
    
    print(f"Loading {len(pdf_files)} PDF files...")
    
    all_documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {str(e)}")
    
    print(f"Loaded {len(all_documents)} pages total")
    return all_documents


def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_or_load_vectorstore(chunks: List[Document], embedding_model: str, db_path: str):
    if not chunks:
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embeddings loaded")
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return None
    
    try:
        if os.path.exists(db_path) and os.listdir(db_path):
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            print("Vector store loaded from disk")
        else:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
            print(f"Vector store created with {len(chunks)} chunks")
        
        return vectorstore
    except Exception as e:
        print(f"Vector store error: {str(e)}")
        return None


def initialize_llm(config: Config):
    if config.USE_MOCK_LLM:
        print("Using Mock LLM (set config.USE_MOCK_LLM = False for real LLM)")
        return KamuLLM(temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
 #   elif config.USE_LOCAL_LLM and os.path.exists(config.LLM_MODEL_PATH):
        try:
            return LlamaCpp(
                model_path=config.LLM_MODEL_PATH,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                verbose=False
            )
        except Exception as e:
            print(f"LLM loading error: {str(e)}, falling back to Mock")
            return KamuLLM(temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
    else:
        return KamuLLM(temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)


class AgentState(TypedDict):
    query: str
    needs_rag: bool
    retrieved_context: str
    response: str
    confidence: float

# kezdetleges validáló query döntéshozáshoz 
def analyze_query(state: AgentState) -> AgentState:
    """
    autonom döntés: kell-e RAG a válaszhoz
    """
    query = state["query"].lower()
    
    # kérdések amelyek RAG-ot jeleznek
    rag_indicators = ["mi", "ki", "milyen", "mikor", "hol", "hogyan", "mirol", 
                      "tartalmaz", "szerint", "dokumentum", "szol", "tartott"]
    # Kérdések, amelyek nem igényelnek dokumentumokat
    direct_indicators = ["mennyi az ido", "mi a datum", "hello", "szia", "koszonom","Mennyi 2+2?"]

    # Döntési logika
    if any(ind in query for ind in direct_indicators):
        state["needs_rag"] = False
    elif any(ind in query for ind in rag_indicators):
        state["needs_rag"] = True
    else:
        # minden más esetben használjon RAG-ot
        state["needs_rag"] = True
    
    return state


def retrieve_context(state: AgentState, vectorstore) -> AgentState:
    if vectorstore is None:
        state["retrieved_context"] = ""
        state["confidence"] = 0.0
        return state
    
    try:
        docs = vectorstore.similarity_search(state["query"], k=config.TOP_K_DOCS)
        
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:400]
                context_parts.append(f"[{i}] Source: {source}, Page: {page}\n{content}")
            
            state["retrieved_context"] = "\n\n".join(context_parts)
            state["confidence"] = 0.8
        else:
            state["retrieved_context"] = ""
            state["confidence"] = 0.3
    except Exception as e:
        state["retrieved_context"] = ""
        state["confidence"] = 0.0
    
    return state


def generate_response(state: AgentState, llm) -> AgentState:
    query = state["query"]
    
    if state["needs_rag"] and state["retrieved_context"]:
        prompt = f"""Valaszolj a kerdesre a dokumentumok alapjan magyarul.

Kontextus:
{state["retrieved_context"]}

Kerdes: {query}

Valasz:"""
    else:
        prompt = f"""Valaszolj roviden magyarul.

Kerdes: {query}

Valasz:"""
    
    try:
        state["response"] = llm.invoke(prompt)
    except Exception as e:
        state["response"] = f"Error: {str(e)}"
    
    return state


def should_use_rag(state: AgentState) -> str:
    return "retrieve" if state["needs_rag"] else "generate_direct"


def create_agentic_workflow(vectorstore, llm):
    """
    LangGraph workflow with autonomous routing
    """
    workflow = StateGraph(AgentState)
    
    def retrieve_with_vectorstore(state):
        return retrieve_context(state, vectorstore)
    
    def generate_with_llm(state):
        return generate_response(state, llm)
    
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_with_vectorstore)
    workflow.add_node("generate_with_rag", generate_with_llm)
    workflow.add_node("generate_direct", generate_with_llm)
    
    workflow.set_entry_point("analyze")
    
    workflow.add_conditional_edges(
        "analyze",
        should_use_rag,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct"
        }
    )
    
    workflow.add_edge("retrieve", "generate_with_rag")
    workflow.add_edge("generate_with_rag", END)
    workflow.add_edge("generate_direct", END)
    
    print("Agentic workflow created")
    return workflow.compile()


def query_agentic_rag(question: str, workflow_app) -> Dict[str, Any]:
    initial_state = {
        "query": question,
        "needs_rag": False,
        "retrieved_context": "",
        "response": "",
        "confidence": 0.0
    }
    
    result = workflow_app.invoke(initial_state)
    return result


def main():
    print("=== RAG Agent app teszt ===\n")
    
    documents = load_pdf_documents(config.KNOWLEDGE_DIR)
    if not documents:
        print("Error: Could not load documents")
        return
    
    chunks = split_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        print("Error: Chunking failed")
        return
    
    vectorstore = create_or_load_vectorstore(chunks, config.EMBEDDING_MODEL, config.VECTOR_DB_PATH)
    if vectorstore is None:
        print("Error: Vector store creation failed")
        return
    
    llm = initialize_llm(config)
    workflow_app = create_agentic_workflow(vectorstore, llm)
    
    print("\n=== Interactive Chat ===")
    print("Ask questions about the documents. Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'kilepes']:
                print("\nGoodbye!")
                break
            
            print("-" * 60)
            result = query_agentic_rag(question, workflow_app)
            
            print(f"RAG used: {'Yes' if result['needs_rag'] else 'No'}")
            
            if result['needs_rag'] and result['retrieved_context']:
                print(f"Retrieved docs: {len(result['retrieved_context'].split('['))}")
                print(f"Confidence: {result['confidence']:.2f}")
            
            print(f"\nAnswer:\n{result['response']}")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)

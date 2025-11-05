# Agentic RAG Chatbot

Autonóm döntéshozatalt használó RAG (Retrieval-Augmented Generation) chatbot LangGraph és LangChain segítségével.

## 🚀 Gyors Használat

### 1. Telepítés
```bash
pip install -r requirements.txt
```

### 2. PDF hozzáadása
Helyezd a PDF fájlokat a `knowledge/` mappába.

### 3. Futtatás
```bash
python agentic_rag_app.py
```

## 🤖 Működés

A chatbot **autonóm döntést hoz**, hogy mikor kell a PDF dokumentumokból információt keresnie:

```
Kérdés → [ELEMZÉS] → Kell RAG? 
                        ├─ IGEN → [Keresés PDF-ben] → Válasz kontextussal
                        └─ NEM  → Direkt válasz
```

### Agentic Viselkedés
- **Analyze node**: Elemzi a kérdést, dönt a RAG szükségességről
- **Conditional routing**: LangGraph automatikusan irányít a megfelelő ágra
- **State management**: Követi a query állapotát a teljes workflow-n keresztül

## 📁 Technológiák

- **LangGraph**: Agentic workflow, conditional edges
- **LangChain**: Document loading, text splitting
- **ChromaDB**: Vector store, similarity search
- **HuggingFace**: Multilingual embeddings (magyar támogatás)

## 💡 Példa Kérdések

**RAG-et használ:**
- "Mi található a dokumentumban?"
- "Milyen témákat tárgyal a PDF?"

**Direkt válasz:**
- "Mennyi 2+2?"
- "Szia, hogy vagy?"

---

**Állásinterjú projekt** - Demonstrálja az agentic AI, RAG technikát és task decomposition-t.

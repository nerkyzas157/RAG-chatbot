from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents for the question.
    
    Combines all retrieved document contents into a single context string.
    """
    
    question = state["question"]
    
    try:
        documents = retriever.invoke(question)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return {
            "documents": [],
            "context": "No relevant documents found.",
        }
    
    if not documents:
        print("No documents retrieved")
        return {
            "documents": [],
            "context": "No relevant documents found.",
        }
    
    # Combine all document contents for context
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Document {i+1} - Source: {source}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
        
    return {
        "documents": documents,
        "context": context,
    }

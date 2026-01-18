from typing import List, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """State for the RAG workflow graph."""
    
    # User input
    question: str
    
    # Retrieved documents
    documents: List[Document]
    context: str
    
    # Generation and evaluation
    generation: str
    feedback: Optional[str]
    
    # Loop control
    attempts: int
    max_attempts: int
    answer_ready: bool
    
    # Chat memory
    chat_history: List[tuple[str, str]]  # List of (human, ai) message pairs

from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from graph.graph import app
from graph.state import GraphState

load_dotenv(override=True)


class RAGChatbot:
    """RAG Chatbot with conversation memory."""
    
    def __init__(self, max_attempts: int = 5):
        """
        Initialize the chatbot.
        
        Args:
            max_attempts: Maximum evaluation/regeneration attempts per question.
        """
        self.max_attempts = max_attempts
    
    def ask(
        self,
        question: str,
        chat_history: List[Tuple[str, str]]
    ) -> str:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User's question.
            chat_history: Previous conversation history as (user, assistant) tuples.
            
        Returns:
            The generated answer.
        """
        if not question or not question.strip():
            return "Please provide a valid question."
        
        # Prepare initial state
        initial_state: GraphState = {
            "question": question.strip(),
            "documents": [],
            "context": "",
            "generation": "",
            "feedback": None,
            "attempts": 0,
            "max_attempts": self.max_attempts,
            "answer_ready": False,
            "chat_history": list(chat_history),
        }
        
        try:
            # Run the graph
            result = app.invoke(initial_state)
            
            # Extract the answer
            answer = result.get("generation", "I couldn't generate an answer. Please try again later.")
            return answer
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"Error in RAG pipeline: {e}")
            return error_msg


# Global chatbot instance
chatbot_instance = RAGChatbot(max_attempts=5)


def respond(
    message: str,
    history: list
) -> str:
    """
    Gradio chat interface callback.
    
    Args:
        message: Current user message.
        history: Conversation history.
        
    Returns:
        Assistant response.
    """
    # Convert message format to (user, assistant) tuples
    tuple_history: List[Tuple[str, str]] = []
    
    if history:
        # Process pairs of messages
        i = 0
        while i < len(history) - 1:
            user_msg = history[i]
            assistant_msg = history[i + 1]
            
            user_content = user_msg.get("content", "")
            assistant_content = assistant_msg.get("content", "")
            
            if user_content and assistant_content:
                tuple_history.append((user_content, assistant_content))
            i += 2
    
    # Keep only last 10 exchanges for memory management
    recent_history = tuple_history[-10:] if len(tuple_history) > 10 else tuple_history
    
    response = chatbot_instance.ask(message, recent_history)
    return response


def create_app() -> gr.ChatInterface:
    """Create and configure the Gradio application."""
    
    demo = gr.ChatInterface(
        fn=respond,
        title='"Mano Būstas" RAG Chatbot',
        description="Ask questions about provided services and information.",
    )
    
    return demo


def main() -> None:
    """Run the Gradio web application."""
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        pwa=False,
    )


if __name__ == "__main__":
    main()


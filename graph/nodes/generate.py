from typing import Any, Dict

from graph.chains.generator import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer based on the question, context, and optional feedback.
    
    Uses retrieved documents as context and incorporates evaluator feedback
    when available for improved answers on retry attempts.
    """    
    question = state["question"]
    context = state.get("context", "")
    feedback = state.get("feedback")
    chat_history = state.get("chat_history", [])
    
    # Format chat history for context
    history_text = ""
    if chat_history:
        history_lines = []
        for human_msg, ai_msg in chat_history[-10:]:  # Last 10 exchanges
            history_lines.append(f"Human: {human_msg}")
            history_lines.append(f"Assistant: {ai_msg}")
        history_text = "\n".join(history_lines)
    
    try:
        generation: str = generation_chain.invoke({
            "question": question,
            "context": context,
            "feedback": feedback or "No feedback yet.",
            "chat_history": history_text or "No previous conversation.",
        })
    except Exception as e:
        print(f"Error in generation: {e}")
        generation = f"I encountered an error while generating the answer: {str(e)}"
        
    return {"generation": generation}

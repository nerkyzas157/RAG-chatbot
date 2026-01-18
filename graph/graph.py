from typing import Literal

from langgraph.graph import END, StateGraph

from graph.nodes.evaluate import evaluate
from graph.nodes.generate import generate
from graph.nodes.retrieve import retrieve
from graph.state import GraphState


def should_continue(state: GraphState) -> Literal["generate", "end", "max_attempts"]:
    """
    Determine the next step after evaluation.
    
    Returns:
        - "end": Answer is satisfactory
        - "generate": Need to regenerate with feedback
        - "max_attempts": Maximum attempts reached
    """
    answer_ready = state.get("answer_ready", False)
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 5)
    
    if answer_ready:
        return "end"
    
    if attempts >= max_attempts:
        print("Max Evaluator-Optimizer attempts reached")
        return "max_attempts"
    
    return "generate"


def handle_max_attempts(state: GraphState) -> dict:
    """
    Handle the case when maximum attempts are reached.
    
    Modifies the generation to include a clarification request.
    """    
    current_generation = state.get("generation", "")
    clarification_message = (
        "\n\n---\n"
        "**Note:** I was unable to fully answer your question after multiple attempts. "
        "Could you please clarify or rephrase your question? "
        "More specific details would help me provide a better answer."
    )
    
    return {
        "generation": current_generation + clarification_message,
        "answer_ready": True,  # Mark as ready to exit the loop
    }


def build_graph() -> StateGraph:
    """Build and compile the RAG workflow graph."""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("max_attempts", handle_max_attempts)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    
    # Add conditional edges from evaluate
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "generate": "generate",             # Loop back to regenerate
            "end": END,                         # Answer is ready
            "max_attempts": "max_attempts",     # Handle max attempts
        }
    )
    
    # Max attempts leads to end
    workflow.add_edge("max_attempts", END)
    
    # Compile the graph
    return workflow.compile()


# Create the compiled app
app = build_graph()

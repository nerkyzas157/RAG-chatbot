from typing import Any, Dict

from graph.chains.evaluator import evaluation_chain
from graph.state import GraphState


def evaluate(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate the generated answer against the question.
    
    Returns updated state with:
    - answer_ready: True if the answer is satisfactory
    - feedback: Improvement suggestions if answer is not ready
    - attempts: Incremented attempt counter
    """    
    question = state["question"]
    generation = state["generation"]
    attempts = state.get("attempts", 0) + 1
    
    # Evaluate the answer
    result = evaluation_chain.invoke({
        "question": question,
        "answer": generation
    })
    
    answer_ready = result.binary_score
    feedback = result.feedback if not answer_ready else None
    
    return {
        "attempts": attempts,
        "answer_ready": answer_ready,
        "feedback": feedback,
    }
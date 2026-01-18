from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from graph.chains.prompts import EVALUATION_PROMPT

load_dotenv(override=True)


class GradeAnswer(BaseModel):
    """Evaluate answer quality against the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    feedback: str = Field(
        description="If score is no, explain what is missing or wrong. Empty if yes."
    )


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)


base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            EVALUATION_PROMPT
        )
    ]
)

evaluation_chain: RunnableSequence = base_prompt | structured_llm_grader
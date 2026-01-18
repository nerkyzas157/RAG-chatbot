from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.chains.prompts import GENERATION_PROMPT

load_dotenv(override=True)


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            GENERATION_PROMPT
        )
    ]
)

# Chain outputs a string directly
generation_chain: RunnableSequence = base_prompt | llm | StrOutputParser()
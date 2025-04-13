from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from dotenv import load_dotenv
from knowledge_base import pdf_knowledge_base

load_dotenv("configs/.env")


pdf_knowledge_base.load(recreate=False)

agent = Agent(
    model=AzureOpenAI(id="gpt-35-turbo"),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
)

agent.print_response("Ask me about something from the knowledge base")

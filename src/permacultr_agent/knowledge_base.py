from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.models.azure import AzureOpenAI
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.azure_openai import
from dotenv import load_dotenv

# TODO define embedder
load_dotenv("configs/.env")


pdf_knowledge_base = PDFKnowledgeBase(
    path="data/pdfs",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=AzureOpenAI(id="gpt-35-turbo"),
        search_type=SearchType.hybrid,
    ),
    reader=PDFReader(chunk=True),
)

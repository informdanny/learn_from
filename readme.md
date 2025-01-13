# Learn From
Learn from is a project that aims to help people learn from some of the internets best and brightest. The project is currently in its early stages and is not yet ready for use.

Tech stack will include:
- Python
- FastAPI (for agentic workflows)
- 

## Data Pipeline
The data pipeline is the first module to be built. It allows you to capture the youtube transcripts of any specific channel and store them in relational database. The MVP focuses on Youtube but could easily include Podcasts, Social Media, and internet search results.

## RAG Pipeline
The RAG pipeline will focus on expirimentation with various models and chunking & retrieval techniques.

### Ingesting Data
Data Ingestion will support various chunking techniques.
- Simply storing data to Openai's File Store and using with assistant API.
- Fixed Chunking (Token, Sentence, Paragraph)
- Recursive Chunking (Sentence, Paragraph, Page)
- Agentic Chunking. We might forgo this because of cost.

### Data Retrieval
Data Retrieval will support various retrieval techniques.
- Chunk to Context - Use chunks to retrieve context that will then be used to answer questions.
- Chunk -> Document to Context - Use chunks to retrieve entire documents that will then be used to answer questions.
- Reranking is a must. Likely we will use Cohere.

### Evaluation
Evaluation envolves testing the retrieval and chunking techniques. We will use a combination of human evaluation and automated evaluation to decide which techniques are best.
- Human Evaluation
- Automated Evaluation using LangGraph or Openai's native



# Create Frontend
The frontend will be a chat interface that allows users to ask questions and get answers.

The frontend will either be built using React and will be hosted on Vercel, or as low code (slack, bubble.io, etc).

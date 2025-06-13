# Dynamic Multi-Agent System

## Overview
This project is a **Dynamic Multi-Agent System** built using **Python** and **Streamlit**. It allows users to create, manage, and interact with customizable AI agents capable of performing tasks such as file processing, data analysis, and external searches. The system supports various file types (**CSV**, **Excel**, **PDF**, **DOCX**, **TXT**) and integrates tools for **Wikipedia**, **YouTube**, **Arxiv**, and web searches via **Tavily** and **Firecrawl**. It uses a **SQLite** database to store user and agent information and supports **Retrieval-Augmented Generation (RAG)** for document searches.

## Features
- **User Authentication**: Users log in with a username, and their data is stored in a SQLite database.
- **Agent Management**: Create, edit, and delete AI agents with customizable roles, goals, backstories, and toolsets.
- **File Processing**: Upload and process files (CSV, Excel, PDF, DOCX, TXT) for content extraction and analysis.
- **Data Analysis**: Perform basic data analysis (e.g., summary statistics) on CSV and Excel files.
- **RAG Search**: Search stored documents using embeddings and cosine similarity for relevant content retrieval.
- **External Tools**: Integrate with Wikipedia, YouTube, Arxiv, Tavily, and Firecrawl for information retrieval.
- **PDF Generation**: Write content to PDF files using the FPDF library.
- **Streamlit Interface**: A user-friendly web interface for agent management, file uploads, and chat interactions.

## Requirements
- Python 3.8+
- SQLite (included with Python)

### Required Python Packages (via `pip install -r requirements.txt`)
- `streamlit`
- `pandas`
- `PyPDF2`
- `python-docx`
- `langchain-openai`
- `numpy`
- `python-dotenv`
- `fpdf`
- `sqlite3` (built-in)
- Custom `agno` library (assumed to provide `Agent`, `OpenAIChat`, and tool integrations)

## Installation

```bash
# Clone the repository
git clone https://github.com/anweshaprakash/dynamicagent
cd dynamicagent

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Set up environment variables in a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run the application:

```bash
streamlit run app.py
```

## Usage

### Launch the Application
- Run `streamlit run app.py` to start the Streamlit interface.
- Access the app at [http://localhost:8501](http://localhost:8501) in your browser.

### User Login
- Enter a username to log in. A user ID is automatically generated and stored in the SQLite database.

### Agent Creation
- Create a new agent by specifying its role, goal, backstory, and tools (e.g., file reading, Wikipedia search).
- Save agents to the database for future use.

### Agent Interaction
- Select an existing agent or create a new one.
- Upload files for processing or analysis.
- Enter queries in the chat interface to interact with the selected agent.
- Download generated PDFs or view tool outputs (e.g., search results, data summaries).

### Agent Management
- View, edit, or delete saved agents from the agent setup page.

## Database Structure

The SQLite database (`dynamic_agents.db`) contains two tables:

### `users` Table:
| Column      | Type      | Description                  |
|-------------|-----------|------------------------------|
| user_id     | TEXT      | Primary key, unique user ID  |
| username    | TEXT      | Unique username              |
| created_at  | TIMESTAMP | Account creation time        |

### `agents` Table:
| Column      | Type      | Description                                      |
|-------------|-----------|--------------------------------------------------|
| agent_id    | TEXT      | Primary key, unique agent ID                     |
| user_id     | TEXT      | Foreign key referencing `users(user_id)`         |
| role        | TEXT      | Agent’s role                                     |
| goal        | TEXT      | Agent’s goal                                     |
| backstory   | TEXT      | Agent’s backstory                                |
| tools       | TEXT      | JSON-encoded list of tools                       |
| created_at  | TIMESTAMP | Timestamp of agent creation                      |

## Tools

Supported tools:
- `read_file`: Reads TXT, PDF, DOCX, CSV, or Excel files.
- `write_file`: Writes content to PDF.
- `read_pdf`: Extracts text from PDF.
- `rag_search`: Retrieval-Augmented Generation on stored documents.
- `data_analysis`: Basic analysis on CSV or Excel (summary, head).
- External: `wikipedia`, `youtube`, `arxiv`, `tavily`, `firecrawl`, `csv`.

## File Management

- Uploaded files are stored in the `temp_files` directory temporarily.
- Supported formats: **CSV**, **XLS/XLSX**, **PDF**, **DOCX**, **TXT**.
- File content is processed and saved in session state for agent access.

## Notes

- Ensure `.env` contains valid `OPENAI_API_KEY` and `TAVILY_API_KEY`.
- `agno` is a custom library expected to define agent/tool behaviors.
- No support for image file processing yet.
- Embedding model used: `text-embedding-3-small`.
- Timezone set to **IST (Asia/Kolkata)**.

## Limitations

- Image processing is unimplemented.
- Internet required for tools like Wikipedia, YouTube, etc.
- `agno` must be provided manually as it's not a standard package.

## Contributing

Contributions are welcome!  
Open issues or submit pull requests for bug reports and feature requests.

## License

This project is licensed under the **MIT License**.

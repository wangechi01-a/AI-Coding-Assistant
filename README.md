
# AI Coding Assistant

This project is an AI-powered coding assistant built using a Retrieval-Augmented Generation (RAG) pipeline. It ingests and indexes documentation or codebase information, allowing users to query for help on code-related topics. The assistant retrieves relevant information from indexed documents and uses a language model to generate responses, providing users with accurate and contextual assistance.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Ingestion](#data-ingestion)
- [Querying the Assistant](#querying-the-assistant)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [License](#license)

## Project Overview

The AI Coding Assistant leverages FAISS (Facebook AI Similarity Search) for efficient document storage and retrieval, and uses Hugging Face's `LaMini-T5-738M` as a language model for generating text-based responses. The assistant helps developers by retrieving relevant code information and generating answers to technical questions, providing real-time support.

## Features

- **Document Ingestion**: Easily ingest custom documentation or knowledge into a vector store for quick access.
- **Query Support**: Retrieve relevant information from ingested data based on user queries.
- **Powered by Hugging Face**: Leverages the `LaMini-T5-738M` language model for generating contextual answers.
- **Streamlit UI**: Provides an interactive user interface for seamless querying and data ingestion.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/ai-coding-assistant.git
   cd ai-coding-assistant
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file (see [Configuration](#configuration)).

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run chatbot_app.py
   ```

2. **Interact with the Assistant**:
   - **Query**: Type in your coding-related question, and the assistant will respond based on the ingested documents.
   - **Ingest Data**: Use the data ingestion interface to add new documents to the assistant’s knowledge base.

## Data Ingestion

To ingest data:

1. Enter data in JSON or comma-separated format in the "Data Ingestion" section of the UI.
2. Click "Ingest Data" to save the new data into the FAISS index, making it available for future queries.

## Querying the Assistant

The assistant retrieves relevant documents based on similarity search and generates answers using the language model. It supports natural language questions, making it easy to use for developers of any skill level.

## Dependencies

The primary dependencies for this project are:

- [Streamlit](https://streamlit.io/) - For the web-based UI.
- [FAISS](https://github.com/facebookresearch/faiss) - For vector similarity search.
- [Hugging Face Transformers](https://huggingface.co/transformers/) - For language model integration.
- [Pydantic](https://pydantic-docs.helpmanual.io/) - For data validation.

See `requirements.txt` for a complete list of dependencies.

## Configuration

Set the following environment variables in a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `PERSIST_DIRECTORY`: Directory path for storing FAISS indexes.
- `FAISS_SETTINGS`: Additional FAISS configuration options.


## License

This project is licensed under the MIT License.

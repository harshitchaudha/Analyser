# Analyser

## Overview
This project implements a chatbot using the Gemini Pro API and integrates PDF text analysis. Users can upload PDF documents, which are then processed to create a searchable vector store. The chatbot allows users to ask questions based on the content of the uploaded PDFs.

## Features

1. **PDF Upload and Analysis: Upload multiple PDF documents, extract text, and split it into manageable chunks..**
2. **Vector Store Creation: Convert text chunks into a vector store using embeddings for efficient similarity search.**
3. **Chat Interface: Interact with the chatbot to ask questions about the PDF content using the Gemini Pro API.**
   
## Installation

1. **Clone the Repository**
   ```bash
   git clone <https://github.com/harshitchaudha/Analyser.git>
   cd <repository-directory>

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
 
4. **Set Up Environment Variables**
    Create a .env file in the project root directory.
    Add your Gemini Pro API key to the .env file:
    ```makefile
    GEMINI_API_KEY=your_api_key_here

## Usage
 
1. **Run the Streamlit App**
    ```bash
    streamlit run gemini_chatbot.py

2. **Interact with the Chatbot**
   Open your web browser and go to http://localhost:8501.
   Use the sidebar to upload PDF files.
   Click "Analyze" to process the uploaded PDFs.
   Ask questions in the chat input to get answers based on the PDF content.

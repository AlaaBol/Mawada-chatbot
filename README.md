## Features

- **Arabic Text Support**: Handles Arabic PDF files and generates answers in Arabic (with optional English translation).
- **Semantic Search**: Uses FAISS for efficient retrieval of relevant document chunks.
- **Conversational Memory**: Maintains a history of the conversation to provide context-aware answers.
- **Expandable History**: Displays the conversation history in an expandable format for easy navigation.
- **Local RAG**: Runs entirely on your local machine, ensuring privacy and control over your data.

## Prerequisites

1. **Python 3.8 or higher**: Ensure Python is installed on your system.
2. **Ollama 3**: Install the Ollama 3 model locally. Follow the instructions below.

### Installing Ollama 3

To use the Ollama 3 model, you need to install it locally. Follow these steps:

1. **Install Ollama**:
   - Visit the [Ollama GitHub repository](https://github.com/ollama/ollama) for installation instructions.
   - Download and install the Ollama CLI tool.

2. **Download the Llama 3 Model**:
   - Run the following command in your terminal to download the Llama 3 model:
     ```bash
     ollama pull llama3
     ```

3. **Verify Installation**:
   - Test the installation by running:
     ```bash
     ollama run llama3
     ```
   - If the model runs successfully, you're ready to use it in this project.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/IlyasNgou/Arabic-RAG-Ollama.git
   cd Arabic-RAG-Ollama
   ```
2. Create a virtual environment
   ```bash
   python -m venv venv
   venv/bin/activate
   ```
3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Running the project
1. Start the Streamlit app:
   ```bash
    streamlit run app.py
   ```
2. Open your browser and navigate to the URL provided in the terminal

3. Use the app:
   * Upload one or more Arabic PDF files in the sidebar.
   * Click "Process Documents" to extract and index the text.
   * Enter your question in the input box and click "Answer" to generate a response.
   * View the conversation history in the expandable section below.

## Notes
   * Ensure that the Ollama 3 model is running locally before starting the Streamlit app.
   * The app is designed to handle Arabic text, but it can be extended to support other languages.
   * For large documents, processing and embedding may take some time.

     

Enjoy using the Arabic PDF Semantic Search + AI Answers app! If you have any questions or feedback, feel free to open an issue or contribute to the project.

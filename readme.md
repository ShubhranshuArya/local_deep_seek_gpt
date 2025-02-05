# 🔮 LocalDeepSeek - Local MultiModal Chat Assistant

Welcome to LocalDeepSeek, an advanced AI application that brings together the power of DeepSeek's language and vision models for local image generation and multimodal interactions. This project offers a user-friendly interface built with Streamlit, allowing you to generate images from text, chat with AI about images, and even perform RAG (Retrieval Augmented Generation) on documents.

## ✨ Features

- **MultiModal Chat**: Have intelligent conversations about images
- **Deep GPT Chat**: Enhanced chat interface with "thinking" UI
- **Document RAG**: Upload PDFs and chat about their contents
- **Text-to-Image Generation**: Create stunning images from textual descriptions
- **Local Execution**: All features run locally on your machine

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
```sh
git clone https://github.com/yourusername/localdeepseek.git

cd localdeepseek
```
2. Create and activate a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```
3. Install the required dependencies:

```sh
pip install -r requirements.txt
```
4. Find Ollama installation guide 👉 [here](https://ollama.ai/)


### Key Components:

1. **Frontend Layer**
   - Streamlit interface
   - Real-time updates
   - Interactive parameter controls
   - File upload handling

2. **Core Processing Layer**
   - Model management & caching
   - GPU/CPU optimization
   - Memory management
   - Batch processing

3. **Model Layer**
   - DeepSeek language model
   - Vision processing
   - MultiModal integration
   - RAG implementation

4. **Storage Layer**
   - Vector database (Qdrant)
   - PDF processing
   - Image caching
   - Session management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- DeepSeek AI for their excellent models
- Streamlit for the intuitive UI framework
- Qdrant for vector storage capabilities
- The open-source community for various tools and libraries

## ⚠️ Disclaimer

This project uses AI models locally. Please ensure you have sufficient computational resources and comply with all relevant model licenses and usage terms.

---

For more information, bug reports, or feature requests, please visit our GitHub repository. Don't forget to star ⭐ the project if you find it useful!


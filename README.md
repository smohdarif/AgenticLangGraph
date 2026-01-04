# 🤖 Agentic RAG System

An intelligent document Q&A system that combines PDF analysis with web search capabilities using LangChain agents, OpenRouter, and Tavily.

## ✨ Features

- **📄 PDF Processing**: Upload and analyze PDF documents with semantic search
- **🔍 Intelligent Routing**: Agent automatically decides which tool to use
- **🌐 Web Search**: Falls back to Tavily web search when PDF doesn't have answers
- **📚 Wikipedia Integration**: Access to Wikipedia for general knowledge
- **💬 Conversational Interface**: Clean Streamlit chat interface with history
- **🧠 Multiple LLM Support**: Choose from GPT-4, Claude, Llama, and more via OpenRouter

## 🎯 How It Works

1. **Upload PDF** → System processes and creates vector embeddings
2. **Ask Question** → Agent searches PDF first
3. **No Answer in PDF?** → Agent automatically uses Tavily web search
4. **Need General Knowledge?** → Agent can query Wikipedia
5. **Get Answer** → Receive comprehensive answers with source citations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenRouter API key ([Get it here](https://openrouter.ai))
- Tavily API key ([Get it here](https://tavily.com))

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run agentic_rag_app.py
```

4. Open your browser to `http://localhost:8501`

### Configuration

1. Enter your **OpenRouter API Key** in the sidebar
2. Enter your **Tavily API Key** in the sidebar
3. Select your preferred LLM model
4. Upload a PDF document
5. Start asking questions!

## 📝 Usage Example

```
You: "What are the main points discussed in the document?"
AI: 📄 From PDF: The document discusses three main points:
    1. Implementation of AI systems
    2. Cost optimization strategies
    3. Performance metrics

You: "What are the latest trends in AI?"
AI: 🌐 From Tavily Search: Based on current web search,
    the latest AI trends include...
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenRouter (supporting multiple providers)
- **Framework**: LangChain
- **Search**: Tavily API
- **Knowledge Base**: Wikipedia API
- **Vector Store**: FAISS
- **Embeddings**: OpenAI text-embedding-3-small
- **PDF Processing**: PyPDF

## 🔧 Configuration Options

### Available Models (via OpenRouter)

- `openai/gpt-4o` - Most capable OpenAI model
- `openai/gpt-4-turbo` - Fast and powerful
- `openai/gpt-3.5-turbo` - Cost-effective (default)
- `anthropic/claude-3.5-sonnet` - Excellent reasoning
- `meta-llama/llama-3.1-70b-instruct` - Open-source option

### Temperature Setting

- **0.0**: Deterministic, focused answers
- **0.7**: Balanced creativity (default)
- **1.0**: More creative and varied responses

## 🏗️ Architecture

```
┌─────────────┐
│   User UI   │
│ (Streamlit) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LangChain      │
│  Agent          │
│  (Orchestrator) │
└────┬────────────┘
     │
     ├──► 📄 PDF Search Tool (FAISS Vector Store)
     │
     ├──► 🌐 Tavily Search Tool (Web Search)
     │
     └──► 📚 Wikipedia Tool (Knowledge Base)
```

## 📦 Files Included

- `agentic_rag_app.py` - Main application (single file)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🔐 API Keys

### OpenRouter
1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up and get your API key
3. Add credits to your account (pay-as-you-go)

### Tavily
1. Visit [tavily.com](https://tavily.com)
2. Sign up for free tier (1,000 searches/month)
3. Get your API key from dashboard

## 💡 Tips

1. **Better PDF Results**: Use clear, well-formatted PDFs
2. **Specific Questions**: Ask specific questions for better answers
3. **Model Selection**: Use GPT-3.5-turbo for speed, GPT-4o for quality
4. **Cost Optimization**: Start with cheaper models and upgrade if needed

## 🐛 Troubleshooting

### "Error processing PDF"
- Ensure PDF is not encrypted or password-protected
- Try a different PDF file
- Check file size (very large PDFs may take time)

### "API Key Error"
- Verify your API keys are correct
- Check if you have credits in your OpenRouter account
- Ensure Tavily API key is active

### "No module named X"
- Run `pip install -r requirements.txt` again
- Use a virtual environment for clean installation

## 🎓 Use Cases

- **Research**: Analyze research papers and get contextual information
- **Legal**: Review contracts and get relevant case law
- **Education**: Study materials with supplementary web information
- **Business**: Analyze reports with market context
- **Technical**: Review documentation with latest updates

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests!

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review API documentation
- Open an issue on GitHub

---

**Built with ❤️ using LangChain, OpenRouter, and Streamlit**

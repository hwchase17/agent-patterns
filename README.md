# LangGraph Agent with Streamlit Interface

A comprehensive example of building and deploying a LangGraph agent with a Streamlit web interface that supports streaming responses, feedback collection, and dynamic configuration updates.

## 🚀 Features

- **LangGraph Agent**: Chat agent with tool calling capabilities using `create_react_agent()`
- **Streamlit Interface**: Interactive web UI with real-time streaming responses
- **Feedback System**: Collect user feedback and dynamically update agent configuration
- **Memory Support**: Persistent conversation history using thread-based memory
- **Tool Integration**: Includes weather tool for demonstration
- **Configuration Management**: Runtime system prompt updates based on user feedback
- **Rerun Functionality**: Re-execute previous inputs with updated configuration

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (or Anthropic API key)
- LangSmith API key (optional, for tracing)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agent-patterns
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
   ```

## 🚀 Quick Start

### Step 1: Start the LangGraph Development Server

The LangGraph development server provides the API endpoints and LangGraph Studio UI:

```bash
langgraph dev
```

This will start the server at `http://localhost:2024`. You should see output similar to:
```
Starting LangGraph API server...
- API: http://localhost:2024
- LangGraph Studio: http://localhost:2024/studio
```

### Step 2: Launch the Streamlit Application

In a new terminal window, start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`.

## 💡 Usage

### Basic Chat Interaction

1. **Connect to LangGraph Server**: Click "Connect to LangGraph Server" in the sidebar
2. **Start Chatting**: Enter your message in the text input and click "Send"
3. **View Streaming Response**: Watch the agent's response stream in real-time
4. **Review History**: All conversations are preserved in the chat history

### Advanced Features

#### System Prompt Configuration
- Use the sidebar to view and modify the system prompt
- Changes take effect immediately for new conversations
- The default prompt enables tool usage and helpful responses

#### Feedback and Rerun System
1. **Provide Feedback**: After each agent response, use the feedback form to provide input
2. **Update Configuration**: Click "Update Configuration" to incorporate feedback into the system prompt
3. **Rerun Previous Input**: Use "Rerun Previous Input" to re-execute the last query with updated settings

#### Tool Usage
The agent includes a weather tool for demonstration. Try asking:
- "What's the weather like in San Francisco?"
- "Tell me about the current weather in Tokyo"
- "Is it raining in London right now?"

## 🏗️ Project Structure

```
agent-patterns/
├── agent/
│   └── agent.py          # LangGraph agent implementation
├── streamlit_app.py      # Streamlit web interface
├── langgraph.json        # LangGraph configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your environment variables (create from .env.example)
└── README.md           # This file
```

## 🔧 Configuration

### LangGraph Configuration (`langgraph.json`)

```json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./agent/agent.py:graph"
    },
    "env": ".env"
}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT models |
| `ANTHROPIC_API_KEY` | Alternative | Anthropic API key (if using Claude) |
| `LANGSMITH_API_KEY` | No | LangSmith API key for tracing and monitoring |

## 🛠️ Development

### Agent Development

The agent is defined in `agent/agent.py` and uses:
- **Model**: OpenAI GPT-4o-mini (configurable)
- **Tools**: Weather tool for demonstration
- **Memory**: InMemorySaver for conversation persistence
- **Prompt**: Dynamic system prompt that accepts configuration updates

### Streamlit App Features

The Streamlit application (`streamlit_app.py`) provides:
- Real-time streaming using `RemoteGraph.stream()`
- Thread-based conversation management
- Dynamic configuration updates
- Feedback collection and processing
- Session state management

## 🐛 Troubleshooting

### Common Issues

#### 1. "Failed to connect to LangGraph server"
**Solution**: Ensure the LangGraph development server is running:
```bash
langgraph dev
```
Check that the server is accessible at `http://localhost:2024`.

#### 2. "No module named 'langgraph'"
**Solution**: Install the required dependencies:
```bash
pip install -r requirements.txt
```

#### 3. "OpenAI API key not found"
**Solution**: 
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to the `.env` file
3. Restart both the LangGraph server and Streamlit app

#### 4. "Thread creation failed"
**Solution**: 
- Restart the LangGraph development server
- Clear browser cache and refresh the Streamlit app
- Check that your API keys are valid

#### 5. Streaming responses not working
**Solution**:
- Verify the LangGraph server is running and accessible
- Check browser console for JavaScript errors
- Try refreshing the Streamlit app

### Debug Mode

For additional debugging information, you can:
1. Check LangGraph server logs in the terminal where you ran `langgraph dev`
2. Use LangGraph Studio at `http://localhost:2024/studio` to test the agent directly
3. Enable LangSmith tracing by setting `LANGSMITH_API_KEY` in your `.env` file

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

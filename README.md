# LangGraph Agent with Token Management

A sophisticated LangGraph-based agent that automatically manages conversation length through intelligent summarization while maintaining full tool-calling capabilities. When the conversation approaches the token limit, the agent automatically summarizes the message history to keep interactions within bounds.

## 🚀 Features

- **Automatic Token Management**: Monitors conversation length and summarizes when approaching token limits
- **Multi-Tool Support**: Includes weather, calculator, knowledge search, and random facts tools
- **Multi-LLM Support**: Compatible with OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) models
- **Conversation Persistence**: Maintains conversation state with memory checkpointing
- **Configurable Limits**: Customizable token limits and summarization thresholds
- **LangGraph Studio Ready**: Includes proper `langgraph.json` configuration

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT models) or Anthropic API key (for Claude models)

## 🛠️ Installation

### Option 1: Using pip and requirements.txt

```bash
# Clone the repository
git clone <repository-url>
cd agent-patterns

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using pip and pyproject.toml

```bash
# Clone the repository
git clone <repository-url>
cd agent-patterns

# Install in development mode
pip install -e .
```

## ⚙️ Configuration

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   # Required: Add at least one API key
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional: Configure model and token limit
   MODEL_NAME=gpt-4
   MAX_TOKEN_LIMIT=4000
   ```

3. **Get API Keys:**
   - **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com/)

## 🎯 Usage

### Quick Start

Run the demonstration script to see the agent in action:

```bash
python main.py
```

This will run several demonstration scenarios:
- Basic conversation
- Individual tool demonstrations
- Multi-tool usage
- Token limit and summarization demo

### Using the Agent in Your Code

```python
import asyncio
from dotenv import load_dotenv
from src.agent_patterns.graph import graph
from src.agent_patterns.state import AgentState
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

async def chat_with_agent():
    # Initialize conversation state
    initial_state = AgentState(
        messages=[],
        current_token_count=0,
        summarization_count=0
    )
    
    # Configure conversation thread
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    # Send a message
    user_message = "What's the weather like in New York?"
    initial_state["messages"] = [HumanMessage(content=user_message)]
    
    # Get agent response
    result = await graph.ainvoke(initial_state, config)
    
    # Print the response
    print(result["messages"][-1].content)

# Run the conversation
asyncio.run(chat_with_agent())
```

### LangGraph Studio

This project is configured for LangGraph Studio. The `langgraph.json` file defines the graph entry point:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent_patterns/graph.py:graph"
  },
  "env": ".env"
}
```

## 🔧 Available Tools

The agent comes with four built-in tools:

1. **Weather Tool** (`get_weather`): Get weather information for any location
2. **Calculator** (`calculate`): Perform mathematical calculations safely
3. **Knowledge Search** (`search_knowledge`): Search for information on various topics
4. **Random Facts** (`get_random_fact`): Get interesting random facts

## 🏗️ Project Structure

```
agent-patterns/
├── src/agent_patterns/          # Main package
│   ├── __init__.py             # Package initialization
│   ├── graph.py                # LangGraph workflow definition
│   ├── state.py                # Agent state management
│   ├── tools.py                # Tool implementations
│   └── utils.py                # Utility functions
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── langgraph.json              # LangGraph configuration
├── main.py                     # Demonstration script
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Pip dependencies
└── README.md                   # This file
```

## 🧠 How Token Management Works

1. **Token Counting**: Uses `tiktoken` library for accurate token measurement
2. **Threshold Monitoring**: Checks token count at 80% of the configured limit
3. **Smart Summarization**: When threshold is reached:
   - Keeps the most recent message intact
   - Summarizes all previous messages into a concise system message
   - Maintains conversation context while reducing token count
4. **Configurable Limits**: Default limit is 4000 tokens, adjustable via `MAX_TOKEN_LIMIT`

## 🔄 Workflow Architecture

The agent follows this workflow:

1. **Token Check** → Monitor conversation length
2. **Agent Processing** → Generate LLM response with tool binding
3. **Tool Execution** → Execute tools if requested by the agent
4. **Loop Back** → Return to token checking for next iteration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (if available)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions, please open an issue on the repository.

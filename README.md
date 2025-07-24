# React Agent with Message Summarization

A LangGraph-powered React agent that automatically summarizes conversations when they exceed 100,000 tokens, maintaining conversation context while managing memory efficiently.

## Features

- **React Agent Architecture**: Built using LangGraph's StateGraph with custom state management
- **Automatic Message Summarization**: Triggers when conversations exceed 100,000 tokens
- **Tool Integration**: Includes search and calculator tools for demonstration
- **Memory Management**: Removes old messages while preserving recent context (last 5 messages)
- **Conversation Continuity**: Injects summaries as system messages to maintain context
- **TypeScript Support**: Fully typed implementation with proper error handling
- **LangGraph Platform Compatible**: Ready for local development and cloud deployment

## Architecture

The agent uses a custom StateGraph workflow with three main nodes:

1. **Conversation Node**: Handles React agent logic with tool binding and summary injection
2. **Tools Node**: Executes tool calls (search, calculator, etc.)
3. **Summarization Node**: Creates/extends summaries and removes old messages

### State Management

The agent extends LangGraph's `MessagesAnnotation` with a custom `summary` field:

```typescript
const ReactAgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  summary: Annotation<string>({
    reducer: (_, action) => action,
    default: () => "",
  }),
});
```

### Token Management

Uses `trimMessages` utility with ChatOpenAI token counter to monitor conversation length:

```typescript
const trimmedMessages = await trimMessages(messages, {
  strategy: "last",
  tokenCounter: tokenCounter,
  maxTokens: 100000, // 100,000 token limit
  startOn: "human",
  endOn: ["human", "tool"],
  includeSystem: true,
});
```

## Prerequisites

- Node.js 18+ 
- npm or yarn
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd react-agent-with-summarization
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   ```

## Usage

### Running the Example

The project includes a comprehensive example demonstrating all features:

```bash
npm run example
```

This will run three example scenarios:
1. **Normal Conversation**: Basic agent functionality without summarization
2. **Long Conversation**: Demonstrates automatic summarization when token limits are exceeded
3. **Conversation Continuation**: Shows how summaries maintain context across interactions

### Using the Agent Programmatically

```typescript
import { app } from "./agent";
import { HumanMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

// Create a thread configuration for conversation persistence
const config = {
  configurable: {
    thread_id: uuidv4(),
  },
};

// Invoke the agent
const result = await app.invoke(
  { 
    messages: [
      new HumanMessage({ 
        id: uuidv4(), 
        content: "Hello! Can you help me with some calculations?" 
      })
    ] 
  },
  config
);

// Access the response
const lastMessage = result.messages[result.messages.length - 1];
console.log("Agent:", lastMessage.content);

// Check if summarization occurred
if (result.summary) {
  console.log("Summary:", result.summary);
}
```

### Available Tools

The agent comes with two example tools:

1. **Search Tool**: Simulates web search functionality
2. **Calculator Tool**: Performs mathematical calculations

You can extend the agent by adding more tools to the `tools` array in `agent.ts`.

## Development

### Available Scripts

- `npm run dev`: Run the agent in development mode
- `npm run build`: Compile TypeScript to JavaScript
- `npm run start`: Run the compiled JavaScript version
- `npm run example`: Run the comprehensive example

### Local Development Server

The project includes a `langgraph.json` configuration file for LangGraph Platform compatibility:

```bash
# Start the LangGraph development server (if you have LangGraph CLI installed)
langgraph dev
```

### Project Structure

```
├── agent.ts           # Main React agent implementation
├── example.ts         # Comprehensive usage examples
├── package.json       # Dependencies and scripts
├── langgraph.json     # LangGraph Platform configuration
├── tsconfig.json      # TypeScript configuration
└── README.md          # This file
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization

You can customize the agent behavior by modifying:

- **Token Limit**: Change `maxTokens` in the `shouldSummarize` function
- **Messages to Keep**: Modify `messagesToKeep` in the `summarizeConversation` function
- **Tools**: Add or modify tools in the `tools` array
- **Model**: Change the OpenAI model in the `ChatOpenAI` initialization

## How It Works

1. **Message Processing**: Each user message is processed by the conversation node
2. **Token Counting**: The system continuously monitors token count using `trimMessages`
3. **Tool Execution**: If the agent needs to use tools, it routes to the tools node
4. **Summarization Trigger**: When messages exceed 100,000 tokens, it routes to summarization
5. **Summary Creation**: The LLM creates or extends the conversation summary
6. **Message Cleanup**: Old messages are removed using `RemoveMessage`, keeping recent context
7. **Context Injection**: Summaries are injected as system messages in future conversations

## License

MIT

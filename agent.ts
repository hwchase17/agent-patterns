import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { 
  SystemMessage, 
  HumanMessage, 
  AIMessage, 
  RemoveMessage,
  trimMessages 
} from "@langchain/core/messages";
import { 
  MessagesAnnotation, 
  StateGraph, 
  START, 
  END, 
  Annotation 
} from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { v4 as uuidv4 } from "uuid";

// Define the extended state annotation that includes both messages and summary
const ReactAgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  summary: Annotation<string>({
    reducer: (_, action) => action,
    default: () => "",
  }),
});

type State = typeof ReactAgentState.State;

// Initialize the OpenAI model for both conversation and token counting
const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.7,
});

// Token counter for determining when to summarize (using same model for consistency)
const tokenCounter = new ChatOpenAI({
  model: "gpt-4o",
});

// Define example tools for the React agent
const searchTool = tool(
  (input: { query: string }) => {
    // Simulate a search tool - in practice, this would call a real search API
    return `Search results for "${input.query}": Here are some relevant results about ${input.query}. This is a simulated search response.`;
  },
  {
    name: "search",
    description: "Search for information on the web. Use this when you need to find current information or answer questions that require up-to-date data.",
    schema: z.object({
      query: z.string().describe("The search query to execute"),
    }),
  }
);

const calculatorTool = tool(
  (input: { expression: string }) => {
    try {
      // Simple calculator - in practice, use a proper math evaluation library
      const result = eval(input.expression);
      return `The result of ${input.expression} is ${result}`;
    } catch (error) {
      return `Error calculating ${input.expression}: ${error}`;
    }
  },
  {
    name: "calculator",
    description: "Perform mathematical calculations. Use this for arithmetic operations, equations, and mathematical expressions.",
    schema: z.object({
      expression: z.string().describe("The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"),
    }),
  }
);

const tools = [searchTool, calculatorTool];
const toolNode = new ToolNode<State>(tools);

// Function to check if messages exceed token limit and need summarization
async function shouldSummarize(messages: any[]): Promise<boolean> {
  try {
    const trimmedMessages = await trimMessages(messages, {
      strategy: "last",
      tokenCounter: tokenCounter,
      maxTokens: 100000, // 100,000 token limit as specified
      startOn: "human",
      endOn: ["human", "tool"],
      includeSystem: true,
    });
    
    // If trimMessages had to remove messages, we've exceeded the limit
    return trimmedMessages.length < messages.length;
  } catch (error) {
    console.warn("Error checking token count, defaulting to no summarization:", error);
    return false;
  }
}

// Main conversation node that handles React agent logic
async function conversationNode(state: State): Promise<Partial<State>> {
  const { summary, messages } = state;
  
  // Prepare messages for the model
  let conversationMessages = [...messages];
  
  // If we have a summary, inject it as a system message at the beginning
  if (summary && summary.trim()) {
    const summaryMessage = new SystemMessage({
      id: uuidv4(),
      content: `Previous conversation summary: ${summary}`
    });
    conversationMessages = [summaryMessage, ...conversationMessages];
  }
  
  // Bind tools to the model for React agent functionality
  const modelWithTools = model.bindTools(tools);
  
  // Get response from the model
  const response = await modelWithTools.invoke(conversationMessages);
  
  return { messages: [response] };
}

// Function to determine next step: continue conversation, use tools, or summarize
async function shouldContinue(state: State): Promise<"tools" | "summarize" | typeof END> {
  const lastMessage = state.messages[state.messages.length - 1];
  
  // Check if we need to use tools
  if (lastMessage && (lastMessage as AIMessage).tool_calls?.length > 0) {
    return "tools";
  }
  
  // Check if we need to summarize due to token limit
  const needsSummarization = await shouldSummarize(state.messages);
  if (needsSummarization) {
    return "summarize";
  }
  
  // Otherwise, end the conversation
  return END;
}

// Summarization node that creates/extends conversation summary and removes old messages
async function summarizeConversation(state: State): Promise<Partial<State>> {
  const { summary, messages } = state;
  
  // Create summarization prompt
  let summaryPrompt: string;
  if (summary && summary.trim()) {
    summaryPrompt = `This is the current summary of the conversation: ${summary}\n\nExtend and update this summary by incorporating the new messages above. Keep the summary comprehensive but concise.`;
  } else {
    summaryPrompt = "Create a comprehensive summary of the conversation above, capturing the key topics, decisions, and important information discussed.";
  }
  
  // Add summarization prompt to messages
  const messagesForSummary = [
    ...messages,
    new HumanMessage({
      id: uuidv4(),
      content: summaryPrompt,
    }),
  ];
  
  // Generate new summary
  const summaryResponse = await model.invoke(messagesForSummary);
  const newSummary = typeof summaryResponse.content === "string" 
    ? summaryResponse.content 
    : "Summary generation failed";
  
  // Keep only the last 5 messages to maintain recent context
  const messagesToKeep = 5;
  const deleteMessages = messages
    .slice(0, -messagesToKeep)
    .map((m) => new RemoveMessage({ id: m.id }));
  
  return {
    summary: newSummary,
    messages: deleteMessages,
  };
}

// Build the StateGraph workflow
const workflow = new StateGraph(ReactAgentState)
  .addNode("conversation", conversationNode)
  .addNode("tools", toolNode)
  .addNode("summarize", summarizeConversation)
  .addEdge(START, "conversation")
  .addConditionalEdges("conversation", shouldContinue)
  .addEdge("tools", "conversation")
  .addEdge("summarize", END);

// Initialize memory saver for persistence
const memory = new MemorySaver();

// Compile the graph with checkpointer for persistence
export const app = workflow.compile({ 
  checkpointer: memory,
});

// Export the graph for visualization and debugging
export const graph = app.getGraph();

// Export types and utilities for external use
export type { State };
export { ReactAgentState, tools, model };


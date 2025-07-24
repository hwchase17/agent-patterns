import { app } from "./agent";
import { HumanMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

// Example demonstrating the React agent with automatic summarization
async function runExample() {
  console.log("🚀 Starting React Agent with Message Summarization Example\n");

  // Create a thread configuration for conversation persistence
  const config = {
    configurable: {
      thread_id: uuidv4(),
    },
  };

  try {
    // Example 1: Normal conversation that doesn't trigger summarization
    console.log("📝 Example 1: Normal conversation");
    console.log("=".repeat(50));

    const normalMessages = [
      "Hello! Can you help me with some calculations?",
      "What's 15 * 23?",
      "Great! Now can you search for information about TypeScript?",
      "Thanks! That's very helpful."
    ];

    for (const [index, message] of normalMessages.entries()) {
      console.log(`\n👤 User: ${message}`);
      
      const result = await app.invoke(
        { messages: [new HumanMessage({ id: uuidv4(), content: message })] },
        config
      );
      
      const lastMessage = result.messages[result.messages.length - 1];
      console.log(`🤖 Agent: ${lastMessage.content}`);
      
      if (result.summary) {
        console.log(`📋 Summary: ${result.summary}`);
      }
      
      console.log(`💬 Total messages: ${result.messages.length}`);
    }

    // Example 2: Simulate a long conversation that triggers summarization
    console.log("\n\n📚 Example 2: Long conversation triggering summarization");
    console.log("=".repeat(50));

    // Create a new thread for the long conversation example
    const longConversationConfig = {
      configurable: {
        thread_id: uuidv4(),
      },
    };

    // Simulate a very long conversation by adding many messages
    const longConversationTopics = [
      "Let's discuss the history of artificial intelligence. Can you tell me about the early pioneers?",
      "That's fascinating! Now tell me about machine learning algorithms and how they evolved.",
      "Excellent explanation! Can you search for recent developments in neural networks?",
      "Now let's talk about natural language processing. How has it changed over the years?",
      "Great insights! Can you calculate the compound growth rate if AI research funding grew from $1B to $50B over 10 years?",
      "Interesting! Now tell me about the ethical implications of AI development.",
      "Can you search for information about AI safety research?",
      "Let's discuss large language models. How do they work?",
      "Can you calculate how many parameters GPT-4 might have compared to GPT-3?",
      "Tell me about the transformer architecture and its impact on AI.",
      "Can you search for the latest research on AI alignment?",
      "What are the current challenges in making AI systems more interpretable?",
      "Let's talk about AI applications in healthcare. What are the most promising areas?",
      "Can you calculate the potential cost savings if AI reduces diagnostic errors by 30% in a $100B healthcare system?",
      "Now tell me about AI in autonomous vehicles. What are the technical challenges?",
      "Can you search for recent breakthroughs in computer vision?",
      "Let's discuss AI's impact on employment. What are economists saying?",
      "What about AI in education? How might it transform learning?",
      "Can you calculate how many jobs might be created vs displaced if AI adoption grows 20% annually?",
      "Finally, what's your perspective on the future of human-AI collaboration?"
    ];

    // Add a very long initial message to help trigger token limits faster
    const longInitialMessage = `I want to have a comprehensive discussion about artificial intelligence, covering its history, current state, and future implications. This is going to be a detailed conversation where we explore multiple aspects including technical developments, ethical considerations, economic impacts, and societal changes. I'm particularly interested in understanding how AI has evolved from its early days with pioneers like Alan Turing and John McCarthy, through the various AI winters and springs, to the current era of large language models and deep learning. We should also discuss the technical aspects like neural network architectures, training methodologies, and the computational requirements. Additionally, I want to explore the practical applications across different industries such as healthcare, finance, transportation, and education. The ethical dimensions are equally important, including questions about bias, fairness, transparency, and the long-term implications of artificial general intelligence. Economic considerations include job displacement, new job creation, productivity gains, and the distribution of AI benefits across society. Let's start with the historical perspective and work our way through these topics systematically.`;

    console.log(`\n👤 User: ${longInitialMessage.substring(0, 100)}...`);
    
    let result = await app.invoke(
      { messages: [new HumanMessage({ id: uuidv4(), content: longInitialMessage })] },
      longConversationConfig
    );

    console.log(`🤖 Agent: ${result.messages[result.messages.length - 1].content.substring(0, 200)}...`);
    console.log(`💬 Total messages: ${result.messages.length}`);
    if (result.summary) {
      console.log(`📋 Summary: ${result.summary.substring(0, 150)}...`);
    }

    // Continue the conversation with multiple topics
    for (const [index, message] of longConversationTopics.entries()) {
      console.log(`\n--- Message ${index + 2} ---`);
      console.log(`👤 User: ${message.substring(0, 80)}...`);
      
      result = await app.invoke(
        { messages: [new HumanMessage({ id: uuidv4(), content: message })] },
        longConversationConfig
      );
      
      const lastMessage = result.messages[result.messages.length - 1];
      console.log(`🤖 Agent: ${lastMessage.content.substring(0, 150)}...`);
      console.log(`💬 Total messages: ${result.messages.length}`);
      
      if (result.summary) {
        console.log(`📋 Summary: ${result.summary.substring(0, 200)}...`);
        console.log("🎯 SUMMARIZATION TRIGGERED! Old messages have been removed and summarized.");
      }

      // Add a small delay to make the output more readable
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Stop after a few iterations to demonstrate the concept
      if (index >= 8) {
        console.log("\n⏸️  Stopping example after demonstrating summarization...");
        break;
      }
    }

    // Example 3: Demonstrate conversation continuation with summary context
    console.log("\n\n🔄 Example 3: Continuing conversation with summary context");
    console.log("=".repeat(50));

    const continuationMessage = "Based on our previous discussion, can you summarize the key points we covered about AI and then tell me what you think is the most important challenge we need to address?";
    
    console.log(`\n👤 User: ${continuationMessage}`);
    
    result = await app.invoke(
      { messages: [new HumanMessage({ id: uuidv4(), content: continuationMessage })] },
      longConversationConfig
    );
    
    const finalMessage = result.messages[result.messages.length - 1];
    console.log(`🤖 Agent: ${finalMessage.content}`);
    console.log(`💬 Final message count: ${result.messages.length}`);
    
    if (result.summary) {
      console.log(`📋 Final summary: ${result.summary}`);
    }

    console.log("\n✅ Example completed successfully!");
    console.log("\n🔍 Key Features Demonstrated:");
    console.log("• Normal conversation flow with tool usage");
    console.log("• Automatic token counting and summarization triggering");
    console.log("• Message removal while preserving recent context");
    console.log("• Summary creation and extension");
    console.log("• Conversation continuity through summary injection");
    console.log("• Persistent conversation state across multiple interactions");

  } catch (error) {
    console.error("❌ Error running example:", error);
    
    // Provide helpful debugging information
    if (error.message.includes("OPENAI_API_KEY")) {
      console.log("\n💡 Make sure to set your OPENAI_API_KEY environment variable:");
      console.log("export OPENAI_API_KEY='your-api-key-here'");
    }
    
    if (error.message.includes("module")) {
      console.log("\n💡 Make sure to install dependencies:");
      console.log("npm install");
    }
  }
}

// Run the example if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runExample().catch(console.error);

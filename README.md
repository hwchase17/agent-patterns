# Form-Filling Agent

A LangGraph-based intelligent form-filling agent that guides users through multi-section forms with human-in-the-loop interactions, validation, and progress tracking.

## Features

- **Multi-Section Forms**: Process forms with multiple sections sequentially
- **Human-in-the-Loop**: Interactive input collection with user prompts
- **Input Validation**: Comprehensive validation with error handling and re-prompting
- **Progress Tracking**: Track completion status across form sections
- **Session Persistence**: Resume interrupted form sessions
- **Customizable Schema**: Easy configuration for different form types
- **Local Development**: Built-in development server support

## Architecture

The agent is built using LangGraph's StateGraph with the following components:

- **StateGraph**: Manages form workflow and state transitions
- **Human-in-the-Loop**: Uses `interrupt()` for user input collection
- **Validation Logic**: Type checking and custom validation rules
- **Checkpointer**: Persistent state storage for session resumption
- **Form Schema**: Configurable section and field definitions

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   # Create .env file for API keys if using external LLM providers
   echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
   ```

### Running the Agent

#### Option 1: Using the Example Script

```bash
python example_usage.py
```

This will start an interactive session where you can:
- Fill out a new form
- Resume an existing form session

#### Option 2: Using LangGraph Development Server

```bash
# Install LangGraph CLI if not already installed
pip install langgraph-cli

# Start the development server
langgraph dev
```

This starts a local web interface at `http://localhost:8123` for testing and debugging.

## Form Schema

The agent comes with a pre-configured form schema with three sections:

### Personal Information
- **first_name** (required, string): User's first name
- **last_name** (required, string): User's last name  
- **age** (required, integer): Age between 1-120
- **date_of_birth** (optional, string): Date in YYYY-MM-DD format

### Contact Details
- **email** (required, string): Valid email address
- **phone** (required, string): Phone number
- **address** (optional, string): Street address
- **city** (required, string): City name

### Preferences
- **newsletter** (required, boolean): Newsletter subscription preference
- **communication_method** (required, string): Preferred communication method
- **interests** (optional, list): List of interests

## Usage Examples

### Basic Usage

```python
import asyncio
from agent import create_form_agent

async def fill_form():
    agent = create_form_agent()
    config = {"configurable": {"thread_id": "my_session"}}
    
    result = await agent.ainvoke({}, config=config)
    print(f"Form completed! Session: {result['session_id']}")

asyncio.run(fill_form())
```

### Resuming a Session

```python
import asyncio
from agent import create_form_agent

async def resume_form():
    agent = create_form_agent()
    # Use the same thread_id to resume
    config = {"configurable": {"thread_id": "my_session"}}
    
    result = await agent.ainvoke({}, config=config)
    print("Session resumed and completed!")

asyncio.run(resume_form())
```

## Customization

### Adding New Form Sections

To add new form sections, modify the `FORM_SECTIONS` dictionary in `agent.py`:

```python
FORM_SECTIONS = {
    "personal_info": {
        "name": "Personal Information",
        "fields": {
            "first_name": {"type": "str", "required": True, "prompt": "Enter your first name"},
            # ... existing fields
        }
    },
    # Add your new section
    "employment": {
        "name": "Employment Information",
        "fields": {
            "company": {"type": "str", "required": True, "prompt": "Enter your company name"},
            "position": {"type": "str", "required": True, "prompt": "Enter your job title"},
            "salary": {"type": "int", "required": False, "prompt": "Enter your salary (optional)", "min": 0}
        }
    }
}
```

### Adding Custom Validation

Extend the `validate_field()` function to add custom validation rules:

```python
def validate_field(field_name: str, value: Any, field_config: Dict[str, Any]) -> Optional[str]:
    # ... existing validation logic
    
    # Add custom validation
    if field_name == "company" and len(value) < 2:
        return "Company name must be at least 2 characters"
    
    if field_name == "salary" and value < 0:
        return "Salary must be a positive number"
    
    return None
```

### Modifying State Schema

Update the `FormState` TypedDict to include new section data:

```python
class FormState(TypedDict):
    sections_completed: Annotated[List[str], operator.add]
    current_section: str
    
    # Existing sections
    personal_info: NotRequired[Dict[str, Any]]
    contact_details: NotRequired[Dict[str, Any]]
    preferences: NotRequired[Dict[str, Any]]
    
    # Add new section
    employment: NotRequired[Dict[str, Any]]
    
    validation_errors: NotRequired[List[str]]
    form_complete: bool
    session_id: str
```

## Development

### Project Structure

```
.
├── agent.py              # Main agent implementation
├── example_usage.py      # Example usage script
├── langgraph.json       # LangGraph configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Running Tests

To test the agent implementation:

```bash
# Test syntax
python -m py_compile agent.py
python -m py_compile example_usage.py

# Test import
python -c "from agent import create_form_agent; print('Import successful')"
```

### Debugging

Use the LangGraph development server for visual debugging:

```bash
langgraph dev
```

This provides:
- Visual graph representation
- Step-by-step execution tracking
- State inspection at each node
- Interactive debugging interface

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Session Not Resuming**: Check that you're using the same `thread_id` for session resumption
3. **Validation Errors**: Review the form schema and ensure input matches expected types
4. **Development Server Issues**: Ensure `langgraph.json` is in the same directory as `agent.py`

### Getting Help

- Check the LangGraph documentation: https://langchain-ai.github.io/langgraph/
- Review the example usage script for implementation patterns
- Use the development server's debugging features for step-by-step analysis

## License

This project is open source and available under the MIT License.
# agent-patterns

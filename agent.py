"""
Form-Filling Agent using LangGraph

This agent processes forms section by section, collecting user input through
human-in-the-loop interrupts and validating data before proceeding to the next section.
"""

from typing import TypedDict, Annotated, Literal, Optional, Dict, Any, List
from typing_extensions import NotRequired
import operator
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig


# Form State Schema
class FormState(TypedDict):
    """State schema for the form-filling agent."""
    # Form sections and their completion status
    sections_completed: Annotated[List[str], operator.add]
    current_section: str
    
    # Form data for each section
    personal_info: NotRequired[Dict[str, Any]]
    contact_details: NotRequired[Dict[str, Any]]
    preferences: NotRequired[Dict[str, Any]]
    
    # Validation and progress tracking
    validation_errors: NotRequired[List[str]]
    form_complete: bool
    session_id: str


# Form section definitions
FORM_SECTIONS = {
    "personal_info": {
        "name": "Personal Information",
        "fields": {
            "first_name": {"type": "str", "required": True, "prompt": "Enter your first name"},
            "last_name": {"type": "str", "required": True, "prompt": "Enter your last name"},
            "age": {"type": "int", "required": True, "prompt": "Enter your age", "min": 1, "max": 120},
            "date_of_birth": {"type": "str", "required": False, "prompt": "Enter your date of birth (YYYY-MM-DD)"}
        }
    },
    "contact_details": {
        "name": "Contact Details",
        "fields": {
            "email": {"type": "str", "required": True, "prompt": "Enter your email address"},
            "phone": {"type": "str", "required": True, "prompt": "Enter your phone number"},
            "address": {"type": "str", "required": False, "prompt": "Enter your address"},
            "city": {"type": "str", "required": True, "prompt": "Enter your city"}
        }
    },
    "preferences": {
        "name": "Preferences",
        "fields": {
            "newsletter": {"type": "bool", "required": True, "prompt": "Subscribe to newsletter? (yes/no)"},
            "communication_method": {"type": "str", "required": True, "prompt": "Preferred communication method (email/phone/mail)"},
            "interests": {"type": "list", "required": False, "prompt": "Enter your interests (comma-separated)"}
        }
    }
}


def validate_field(field_name: str, value: Any, field_config: Dict[str, Any]) -> Optional[str]:
    """Validate a single field value according to its configuration."""
    if field_config.get("required", False) and (value is None or value == ""):
        return f"{field_name} is required"
    
    if value is None or value == "":
        return None  # Optional field, no validation needed
    
    field_type = field_config.get("type", "str")
    
    try:
        if field_type == "int":
            int_val = int(value)
            if "min" in field_config and int_val < field_config["min"]:
                return f"{field_name} must be at least {field_config['min']}"
            if "max" in field_config and int_val > field_config["max"]:
                return f"{field_name} must be at most {field_config['max']}"
        elif field_type == "bool":
            if str(value).lower() not in ["yes", "no", "true", "false", "1", "0"]:
                return f"{field_name} must be yes/no or true/false"
        elif field_type == "str" and "@" in field_config.get("prompt", "").lower():
            # Simple email validation
            if "@" not in str(value) or "." not in str(value):
                return f"{field_name} must be a valid email address"
    except ValueError:
        return f"{field_name} must be of type {field_type}"
    
    return None


def initialize_form(state: FormState) -> FormState:
    """Initialize the form with a new session."""
    return {
        "session_id": str(uuid.uuid4()),
        "current_section": "personal_info",
        "form_complete": False,
        "sections_completed": []
    }


def collect_section_data(state: FormState) -> FormState:
    """Collect data for the current form section through human-in-the-loop."""
    current_section = state["current_section"]
    section_config = FORM_SECTIONS[current_section]
    
    print(f"\n=== {section_config['name']} ===")
    print("Please provide the following information:")
    
    section_data = {}
    validation_errors = []
    
    # Collect data for each field in the section
    for field_name, field_config in section_config["fields"].items():
        while True:
            # Interrupt to collect user input for this field
            user_input = interrupt({
                "section": current_section,
                "field": field_name,
                "prompt": field_config["prompt"],
                "required": field_config.get("required", False),
                "type": field_config.get("type", "str")
            })
            
            # Validate the input
            error = validate_field(field_name, user_input, field_config)
            if error:
                validation_errors.append(error)
                # Ask for input again with error message
                continue
            else:
                # Process the input based on type
                if field_config.get("type") == "int" and user_input:
                    section_data[field_name] = int(user_input)
                elif field_config.get("type") == "bool" and user_input:
                    section_data[field_name] = str(user_input).lower() in ["yes", "true", "1"]
                elif field_config.get("type") == "list" and user_input:
                    section_data[field_name] = [item.strip() for item in str(user_input).split(",")]
                else:
                    section_data[field_name] = user_input
                break
    
    # Store the collected data in the appropriate state field
    update = {current_section: section_data}
    if validation_errors:
        update["validation_errors"] = validation_errors
    
    return update


def validate_section(state: FormState) -> Command[Literal["collect_section_data", "determine_next_section"]]:
    """Validate the current section data and determine next action."""
    current_section = state["current_section"]
    section_data = state.get(current_section, {})
    section_config = FORM_SECTIONS[current_section]
    
    validation_errors = []
    
    # Validate all required fields are present and valid
    for field_name, field_config in section_config["fields"].items():
        field_value = section_data.get(field_name)
        error = validate_field(field_name, field_value, field_config)
        if error:
            validation_errors.append(error)
    
    if validation_errors:
        print(f"Validation errors in {section_config['name']}:")
        for error in validation_errors:
            print(f"  - {error}")
        return Command(
            update={"validation_errors": validation_errors},
            goto="collect_section_data"
        )
    else:
        print(f"✓ {section_config['name']} completed successfully")
        return Command(
            update={"sections_completed": [current_section]},
            goto="determine_next_section"
        )


def determine_next_section(state: FormState) -> Command[Literal["collect_section_data", "finalize_form"]]:
    """Determine the next section to process or finalize the form."""
    completed_sections = state.get("sections_completed", [])
    
    # Find the next uncompleted section
    for section_name in FORM_SECTIONS.keys():
        if section_name not in completed_sections:
            print(f"Moving to next section: {FORM_SECTIONS[section_name]['name']}")
            return Command(
                update={"current_section": section_name},
                goto="collect_section_data"
            )
    
    # All sections completed
    print("All sections completed! Finalizing form...")
    return Command(goto="finalize_form")


def finalize_form(state: FormState) -> FormState:
    """Finalize the form and provide a summary."""
    print("\n=== Form Completion Summary ===")
    print(f"Session ID: {state['session_id']}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display collected data
    for section_name in FORM_SECTIONS.keys():
        if section_name in state:
            section_config = FORM_SECTIONS[section_name]
            print(f"\n{section_config['name']}:")
            for field_name, value in state[section_name].items():
                print(f"  {field_name}: {value}")
    
    return {"form_complete": True}


# Build the form-filling agent graph
def create_form_agent():
    """Create and compile the form-filling agent graph."""
    builder = StateGraph(FormState)
    
    # Add nodes
    builder.add_node("initialize_form", initialize_form)
    builder.add_node("collect_section_data", collect_section_data)
    builder.add_node("validate_section", validate_section)
    builder.add_node("determine_next_section", determine_next_section)
    builder.add_node("finalize_form", finalize_form)
    
    # Add edges
    builder.add_edge(START, "initialize_form")
    builder.add_edge("initialize_form", "collect_section_data")
    builder.add_edge("collect_section_data", "validate_section")
    builder.add_edge("determine_next_section", END)
    builder.add_edge("finalize_form", END)
    
    # Compile with checkpointer for persistence
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Create the agent instance
app = create_form_agent()


#!/usr/bin/env python3
"""
Interactive example for the Form Filling Agent.

This example provides an interactive command-line interface that allows users to:
1. Choose form configurations
2. Input data interactively
3. See real-time progress updates
4. Handle errors and retry scenarios
5. Explore different completion strategies
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to the path so we can import form_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_agent.config import ConfigurationManager
from form_agent.agent import FormFillingAgent, FormStatus, SectionStatus


def setup_logging():
    """Set up logging for the interactive example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)


def choose_configuration() -> Path:
    """Let user choose which form configuration to use."""
    config_dir = Path(__file__).parent.parent / "configurations"
    config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
    
    print("
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file.name}")
    
    while True:
        try:
            choice = int(input(f"
            if 1 <= choice <= len(config_files):
                return config_files[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(config_files)}")
        except ValueError:
            print("Please enter a valid number")


def collect_user_data(agent: FormFillingAgent) -> Dict[str, Any]:
    """Interactively collect data from the user."""
    print(f"
    print(f"Description: {agent.form_config.description}")
    print("
    print("(Press Enter to skip optional fields, or type 'quit' to finish data entry)
    
    user_data = {}
    
    # Collect data for common field types
    common_fields = [
        ("first_name", "First Name"),
        ("last_name", "Last Name"),
        ("email", "Email Address"),
        ("phone", "Phone Number"),
        ("address", "Street Address"),
        ("city", "City"),
        ("state", "State"),
        ("zip_code", "ZIP Code"),
        ("position", "Position/Job Title"),
        ("experience_years", "Years of Experience"),
        ("education_level", "Education Level"),
        ("university", "University/School"),
        ("graduation_year", "Graduation Year"),
        ("skills", "Skills (comma-separated)"),
    ]
    
    for field_id, field_name in common_fields:
        value = input(f"{field_name}: ").strip()
        if value.lower() == 'quit':
            break
        if value:
            user_data[field_id] = value
    
    return user_data


def display_progress(agent: FormFillingAgent):
    """Display current form completion progress."""
    print(f"
    print("FORM COMPLETION PROGRESS")
    print(f"{'='*50}")
    
    form_status = agent.get_form_status()
    print(f"Overall Status: {form_status.value.upper()}")
    
    for section in agent.sections:
        status = agent.get_section_status(section.id)
        progress = section.get_completion_percentage()
        print(f"  {section.name}: {status.value.upper()} ({progress:.1f}%)")
        
        # Show incomplete required fields
        incomplete_fields = [f for f in section.fields if f.required and not f.is_completed]
        if incomplete_fields:
            print(f"    Missing: {', '.join(f.name for f in incomplete_fields)}")


def interactive_completion_loop(agent: FormFillingAgent, initial_data: Dict[str, Any]):
    """Run an interactive completion loop with user feedback."""
    print("
    
    # Initial completion attempt
    results = agent.run_completion_cycle(available_data=initial_data, max_iterations=3)
    display_progress(agent)
    
    print(f"
    print(f"  Status: {results['final_status'].value}")
    print(f"  Progress: {results['completion_percentage']:.1f}%")
    print(f"  Iterations: {results['iterations_completed']}")
    
    # If not complete, offer retry options
    while results['final_status'] != FormStatus.COMPLETED:
        print("
        print("1. Provide additional data")
        print("2. Retry with current data")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            print("
            additional_data = collect_user_data(agent)
            if additional_data:
                results = agent.retry_failed_sections(additional_data=additional_data)
                display_progress(agent)
        elif choice == "2":
            results = agent.retry_failed_sections()
            display_progress(agent)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
    
    if results['final_status'] == FormStatus.COMPLETED:
        print("
    else:
        print("


def main():
    """Main interactive example function."""
    logger = setup_logging()
    
    print("Welcome to the Interactive Form Filling Agent!")
    print("This tool will help you complete forms step by step.")
    
    try:
        # Step 1: Choose configuration
        config_path = choose_configuration()
        
        # Step 2: Load configuration
        config_manager = ConfigurationManager(logger=logger)
        form_config = config_manager.load_config(str(config_path))
        
        # Step 3: Initialize agent
        agent = FormFillingAgent(form_config, logger=logger)
        
        # Step 4: Collect initial data
        user_data = collect_user_data(agent)
        
        # Step 5: Run interactive completion
        interactive_completion_loop(agent, user_data)
        
        print("
        
    except KeyboardInterrupt:
        print("
        return 0
    except Exception as e:
        logger.error(f"Error during interactive example: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



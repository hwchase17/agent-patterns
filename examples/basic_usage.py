#!/usr/bin/env python3
"""
Basic usage example for the Form Filling Agent.

This example demonstrates how to:
1. Load a form configuration
2. Initialize the form filling agent
3. Provide data for form completion
4. Run the completion process
5. Check results and status
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path so we can import form_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_agent.config import ConfigurationManager
from form_agent.agent import FormFillingAgent


def setup_logging():
    """Set up basic logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """Main example function."""
    logger = setup_logging()
    logger.info("Starting basic form filling agent example")
    
    try:
        # Step 1: Load form configuration
        logger.info("Loading form configuration...")
        config_manager = ConfigurationManager(logger=logger)
        config_path = Path(__file__).parent.parent / "configurations" / "sample_form.json"
        form_config = config_manager.load_config(str(config_path))
        logger.info(f"Loaded form: {form_config.name}")
        
        # Step 2: Initialize the form filling agent
        logger.info("Initializing form filling agent...")
        agent = FormFillingAgent(form_config, logger=logger)
        logger.info(f"Agent initialized with {len(agent.sections)} sections")
        
        # Step 3: Provide sample data for form completion
        logger.info("Providing sample data...")
        sample_data = {
            "first_name": "John",
            "last_name": "Doe", 
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "address": "123 Main St",
            "city": "Anytown",
            "state": "CA",
            "zip_code": "12345"
        }
        
        # Step 4: Run the completion process
        logger.info("Starting form completion process...")
        results = agent.run_completion_cycle(available_data=sample_data, max_iterations=10)
        
        # Step 5: Check results and status
        logger.info("Form completion finished!")
        logger.info(f"Final status: {results['final_status']}")
        logger.info(f"Iterations completed: {results['iterations_completed']}")
        logger.info(f"Sections processed: {results['sections_processed']}")
        logger.info(f"Overall progress: {results['completion_percentage']:.1f}%")
        
        # Display section-by-section results
        logger.info("
        for section_id, status in results['section_statuses'].items():
            section = agent.get_section_by_id(section_id)
            logger.info(f"  {section.name}: {status}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during example execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


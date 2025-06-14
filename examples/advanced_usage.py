#!/usr/bin/env python3
"""
Advanced usage example for the Form Filling Agent.

This example demonstrates:
1. Loading complex form configurations
2. Handling partial data scenarios
3. Section dependency management
4. Error handling and retry mechanisms
5. State management and progress tracking
6. Custom validation and field completion strategies
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to the path so we can import form_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_agent.config import ConfigurationManager
from form_agent.agent import FormFillingAgent, FormStatus, SectionStatus


def setup_logging():
    """Set up detailed logging for the advanced example."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demonstrate_complex_form_loading(logger):
    """Demonstrate loading and inspecting a complex form configuration."""
    logger.info("=== Complex Form Loading Demo ===")
    
    config_manager = ConfigurationManager(logger=logger)
    config_path = Path(__file__).parent.parent / "configurations" / "complex_form.json"
    form_config = config_manager.load_config(str(config_path))
    
    logger.info(f"Loaded complex form: {form_config.name}")
    logger.info(f"Form description: {form_config.description}")
    logger.info(f"Number of sections: {len(form_config.sections)}")
    
    # Display section information
    for section in form_config.sections:
        logger.info(f"  Section: {section.name}")
        logger.info(f"    Fields: {len(section.fields)}")
        logger.info(f"    Dependencies: {section.dependencies}")
        logger.info(f"    Required: {section.required}")
    
    return form_config


def demonstrate_partial_data_completion(agent: FormFillingAgent, logger):
    """Demonstrate form completion with partial data."""
    logger.info("=== Partial Data Completion Demo ===")
    
    # Provide only partial data - some fields will be missing
    partial_data = {
        "first_name": "Jane",
        "last_name": "Smith",
        "email": "jane.smith@example.com",
        "phone": "555-987-6543",
        # Missing address information
        "position": "Software Engineer",
        "experience_years": "5",
        # Missing education and skills information
    }
    
    logger.info("Running completion with partial data...")
    results = agent.run_completion_cycle(available_data=partial_data, max_iterations=5)
    
    logger.info(f"Completion results with partial data:")
    logger.info(f"  Final status: {results['final_status']}")
    logger.info(f"  Progress: {results['completion_percentage']:.1f}%")
    logger.info(f"  Sections processed: {results['sections_processed']}")
    
    # Show which sections completed and which didn't
    incomplete_sections = agent.get_incomplete_sections()
    failed_sections = agent.get_failed_sections()
    
    if incomplete_sections:
        logger.info(f"  Incomplete sections: {[s.name for s in incomplete_sections]}")
    if failed_sections:
        logger.info(f"  Failed sections: {[s.name for s in failed_sections]}")
    
    return results


def demonstrate_retry_mechanism(agent: FormFillingAgent, logger):
    """Demonstrate retry mechanism with additional data."""
    logger.info("=== Retry Mechanism Demo ===")
    
    # Provide additional data for retry
    additional_data = {
        "address": "456 Oak Avenue",
        "city": "Springfield",
        "state": "IL",
        "zip_code": "62701",
        "education_level": "Bachelor's Degree",
        "university": "University of Illinois",
        "graduation_year": "2018",
        "skills": "Python, JavaScript, SQL, React"
    }
    
    logger.info("Retrying failed sections with additional data...")
    retry_results = agent.retry_failed_sections(additional_data=additional_data)
    
    logger.info(f"Retry results:")
    logger.info(f"  Sections retried: {retry_results['sections_retried']}")
    logger.info(f"  Sections completed: {retry_results['sections_completed']}")
    logger.info(f"  New completion percentage: {retry_results['completion_percentage']:.1f}%")
    
    return retry_results


def demonstrate_state_inspection(agent: FormFillingAgent, logger):
    """Demonstrate detailed state inspection and progress tracking."""
    logger.info("=== State Inspection Demo ===")
    
    # Get overall form status
    form_status = agent.get_form_status()
    logger.info(f"Overall form status: {form_status}")
    
    # Inspect each section
    for section in agent.sections:
        section_status = agent.get_section_status(section.id)
        progress = section.get_completion_percentage()
        
        logger.info(f"Section '{section.name}':")
        logger.info(f"  Status: {section_status}")
        logger.info(f"  Progress: {progress:.1f}%")
        logger.info(f"  Required fields: {len(section.get_required_fields())}")
        logger.info(f"  Completed fields: {len([f for f in section.fields if f.is_completed])}")
        
        # Show incomplete fields
        incomplete_fields = [f for f in section.fields if f.required and not f.is_completed]
        if incomplete_fields:
            logger.info(f"  Incomplete required fields: {[f.name for f in incomplete_fields]}")
    
    # Show completion history
    history = agent.get_completion_history()
    if history:
        logger.info(f"Completion history ({len(history)} entries):")
        for entry in history[-3:]:  # Show last 3 entries
            logger.info(f"  {entry['timestamp']}: {entry['completion_percentage']:.1f}% - {entry['sections_completed']} sections")


def main():
    """Main advanced example function."""
    logger = setup_logging()
    logger.info("Starting advanced form filling agent example")
    
    try:
        # Step 1: Load complex form configuration
        form_config = demonstrate_complex_form_loading(logger)
        
        # Step 2: Initialize agent
        agent = FormFillingAgent(form_config, logger=logger)
        
        # Step 3: Demonstrate partial data completion
        partial_results = demonstrate_partial_data_completion(agent, logger)
        
        # Step 4: Demonstrate retry mechanism
        if partial_results['final_status'] != FormStatus.COMPLETED:
            retry_results = demonstrate_retry_mechanism(agent, logger)
        
        # Step 5: Demonstrate detailed state inspection
        demonstrate_state_inspection(agent, logger)
        
        logger.info("Advanced example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during advanced example execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


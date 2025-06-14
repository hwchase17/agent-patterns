"""
Core form filling agent that handles multiple form sections iteratively.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
import logging


class FormStatus(Enum):
    """Status of form completion."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionStatus(Enum):
    """Status of individual section completion."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class FormFillingAgent:
    """
    Core agent class that handles iterative form filling across multiple sections.
    
    The agent processes forms section by section, following specific instructions
    for each section and tracking completion status throughout the process.
    """
    
    def __init__(self, form_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the form filling agent.
        
        Args:
            form_config: Configuration dictionary containing form structure and sections
            logger: Optional logger instance for tracking operations
        """
        self.form_config = form_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize form state
        self.form_status = FormStatus.NOT_STARTED
        self.current_section_index = 0
        self.section_statuses: Dict[str, SectionStatus] = {}
        self.form_data: Dict[str, Any] = {}
        self.completion_history: List[Dict[str, Any]] = []
        
        # Initialize section statuses
        if 'sections' in form_config:
            for section in form_config['sections']:
                section_id = section.get('id', f"section_{len(self.section_statuses)}")
                self.section_statuses[section_id] = SectionStatus.PENDING
    
    def get_current_section(self) -> Optional[Dict[str, Any]]:
        """Get the current section being processed."""
        sections = self.form_config.get('sections', [])
        if 0 <= self.current_section_index < len(sections):
            return sections[self.current_section_index]
        return None
    
    def get_form_status(self) -> FormStatus:
        """Get the overall form completion status."""
        return self.form_status
    
    def get_section_status(self, section_id: str) -> SectionStatus:
        """Get the status of a specific section."""
        return self.section_statuses.get(section_id, SectionStatus.PENDING)
    
    def get_completion_progress(self) -> Dict[str, Any]:
        """Get detailed progress information about form completion."""
        total_sections = len(self.form_config.get('sections', []))
        completed_sections = sum(1 for status in self.section_statuses.values() 
                               if status == SectionStatus.COMPLETED)
        
        return {
            'form_status': self.form_status.value,
            'total_sections': total_sections,
            'completed_sections': completed_sections,
            'current_section_index': self.current_section_index,
            'progress_percentage': (completed_sections / total_sections * 100) if total_sections > 0 else 0,
            'section_statuses': {k: v.value for k, v in self.section_statuses.items()},
            'form_data': self.form_data.copy()
        }
    
    def reset_form(self):
        """Reset the form to initial state."""
        self.form_status = FormStatus.NOT_STARTED
        self.current_section_index = 0
        self.form_data.clear()
        self.completion_history.clear()
        
        # Reset all section statuses
        for section_id in self.section_statuses:
            self.section_statuses[section_id] = SectionStatus.PENDING
        
        self.logger.info("Form has been reset to initial state")


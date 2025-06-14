"""
Core form filling agent that handles multiple form sections iteratively.
"""
from typing import List, Dict, Any, Optional
from enum import Enum, auto
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


class FieldType(Enum):
    """Types of form fields that can be handled."""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    NUMBER = "number"
    DATE = "date"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTAREA = "textarea"
    FILE = "file"


class FormSection:
    """
    Represents an individual form section with its own instructions and field requirements.
    
    Each section contains:
    - Unique identifier and display name
    - Section-specific instructions for the agent
    - List of fields that need to be completed
    - Validation rules and requirements
    - Dependencies on other sections
    """
    
    def __init__(
        self,
        section_id: str,
        name: str,
        instructions: str,
        fields: List[Dict[str, Any]],
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        required: bool = True,
        order: int = 0
    ):
        """
        Initialize a form section.
        
        Args:
            section_id: Unique identifier for the section
            name: Display name of the section
            instructions: Detailed instructions for completing this section
            fields: List of field definitions for this section
            description: Optional description of the section's purpose
            dependencies: List of section IDs that must be completed before this one
            required: Whether this section is required for form completion
            order: Order in which this section should be processed
        """
        self.section_id = section_id
        self.name = name
        self.instructions = instructions
        self.fields = fields
        self.description = description
        self.dependencies = dependencies or []
        self.required = required
        self.order = order
        self.status = SectionStatus.PENDING
        self.field_data: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.completion_notes: List[str] = []
    
    def get_field_by_id(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific field definition by its ID."""
        for field in self.fields:
            if field.get('id') == field_id:
                return field
        return None
    
    def get_required_fields(self) -> List[Dict[str, Any]]:
        """Get all required fields in this section."""
        return [field for field in self.fields if field.get('required', False)]
    
    def get_optional_fields(self) -> List[Dict[str, Any]]:
        """Get all optional fields in this section."""
        return [field for field in self.fields if not field.get('required', False)]
    
    def is_dependencies_met(self, completed_sections: List[str]) -> bool:
        """Check if all dependencies for this section are met."""
        return all(dep in completed_sections for dep in self.dependencies)
    
    def get_completion_percentage(self) -> float:
        """Calculate the completion percentage of this section."""
        if not self.fields:
            return 100.0
        
        completed_fields = sum(1 for field in self.fields 
                             if field.get('id') in self.field_data 
                             and self.field_data[field.get('id')] is not None)
        
        return (completed_fields / len(self.fields)) * 100.0
    
    def validate_section(self) -> bool:
        """
        Validate the current section data.
        
        Returns:
            True if section is valid, False otherwise
        """
        self.validation_errors.clear()
        
        # Check required fields
        required_fields = self.get_required_fields()
        for field in required_fields:
            field_id = field.get('id')
            if field_id not in self.field_data or self.field_data[field_id] is None:
                self.validation_errors.append(f"Required field '{field.get('name', field_id)}' is missing")
        
        # Validate field types and constraints
        for field in self.fields:
            field_id = field.get('id')
            if field_id in self.field_data:
                value = self.field_data[field_id]
                if not self._validate_field_value(field, value):
                    field_name = field.get('name', field_id)
                    self.validation_errors.append(f"Invalid value for field '{field_name}'")
        
        return len(self.validation_errors) == 0
    
    def _validate_field_value(self, field: Dict[str, Any], value: Any) -> bool:
        """Validate a single field value against its constraints."""
        if value is None:
            return not field.get('required', False)
        
        field_type = field.get('type', FieldType.TEXT.value)
        
        # Basic type validation
        if field_type == FieldType.EMAIL.value:
            return '@' in str(value) and '.' in str(value)
        elif field_type == FieldType.NUMBER.value:
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif field_type == FieldType.SELECT.value:
            options = field.get('options', [])
            return value in options if options else True
        
        # Length constraints
        min_length = field.get('min_length')
        max_length = field.get('max_length')
        
        if min_length and len(str(value)) < min_length:
            return False
        if max_length and len(str(value)) > max_length:
            return False
        
        return True
    
    def set_field_value(self, field_id: str, value: Any) -> bool:
        """
        Set a field value and validate it.
        
        Args:
            field_id: ID of the field to set
            value: Value to set for the field
            
        Returns:
            True if value was set successfully, False if validation failed
        """
        field = self.get_field_by_id(field_id)
        if not field:
            return False
        
        if self._validate_field_value(field, value):
            self.field_data[field_id] = value
            return True
        
        return False
    
    def get_field_value(self, field_id: str) -> Any:
        """Get the current value of a field."""
        return self.field_data.get(field_id)
    
    def clear_field_data(self):
        """Clear all field data for this section."""
        self.field_data.clear()
        self.validation_errors.clear()
        self.completion_notes.clear()
        self.status = SectionStatus.PENDING
    
    def mark_completed(self, notes: Optional[str] = None):
        """Mark this section as completed."""
        if self.validate_section():
            self.status = SectionStatus.COMPLETED
            if notes:
                self.completion_notes.append(notes)
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            'section_id': self.section_id,
            'name': self.name,
            'description': self.description,
            'instructions': self.instructions,
            'fields': self.fields,
            'dependencies': self.dependencies,
            'required': self.required,
            'order': self.order,
            'status': self.status.value,
            'field_data': self.field_data.copy(),
            'validation_errors': self.validation_errors.copy(),
            'completion_notes': self.completion_notes.copy(),
            'completion_percentage': self.get_completion_percentage()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormSection':
        """Create a FormSection instance from dictionary data."""
        section = cls(
            section_id=data['section_id'],
            name=data['name'],
            instructions=data['instructions'],
            fields=data['fields'],
            description=data.get('description'),
            dependencies=data.get('dependencies', []),
            required=data.get('required', True),
            order=data.get('order', 0)
        )
        
        # Restore state if present
        if 'status' in data:
            section.status = SectionStatus(data['status'])
        if 'field_data' in data:
            section.field_data = data['field_data'].copy()
        if 'validation_errors' in data:
            section.validation_errors = data['validation_errors'].copy()
        if 'completion_notes' in data:
            section.completion_notes = data['completion_notes'].copy()
        
        return section


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



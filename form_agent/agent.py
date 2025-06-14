"""
Core form filling agent that handles multiple form sections iteratively.
"""
from typing import List, Dict, Any, Optional
from enum import Enum, auto
from .field import FormField
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
        self.sections: List[FormSection] = []
        self.current_section_index = 0
        self.section_statuses: Dict[str, SectionStatus] = {}
        self.form_data: Dict[str, Any] = {}
        self.completion_history: List[Dict[str, Any]] = []
        
        # Initialize section statuses
        if 'sections' in form_config:
            for i, section_data in enumerate(form_config['sections']):
                section_id = section_data.get('id', f"section_{i}")
                self.section_statuses[section_id] = SectionStatus.PENDING
                
                # Create FormSection objects
                section = FormSection.from_dict(section_data) if isinstance(section_data, dict) else section_data
                self.sections.append(section)
        
        # Sort sections by order
        self.sections.sort(key=lambda s: s.order)
    
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
    
    def can_process_section(self, section: FormSection) -> bool:
        """
        Check if a section can be processed based on its dependencies.
        
        Args:
            section: The section to check
            
        Returns:
            True if section can be processed, False otherwise
        """
        if not section.dependencies:
            return True
        
        completed_sections = [
            s.section_id for s in self.sections 
            if self.section_statuses.get(s.section_id) == SectionStatus.COMPLETED
        ]
        
        return section.is_dependencies_met(completed_sections)
    
    def get_next_processable_section(self) -> Optional[FormSection]:
        """
        Get the next section that can be processed based on dependencies and current status.
        
        Returns:
            Next processable section or None if no sections are available
        """
        for section in self.sections:
            section_status = self.section_statuses.get(section.section_id, SectionStatus.PENDING)
            
            # Skip completed or failed sections
            if section_status in [SectionStatus.COMPLETED, SectionStatus.FAILED]:
                continue
            
            # Check if section can be processed
            if self.can_process_section(section):
                return section
        
        return None
    
    def attempt_field_completion(self, field: Dict[str, Any], section: FormSection, available_data: Dict[str, Any] = None) -> bool:
        """
        Attempt to complete a field using available data or intelligent guessing.
        
        Args:
            field: Field definition dictionary
            section: The section containing this field
            available_data: Optional dictionary of available data to use for field completion
            
        Returns:
            True if field was successfully completed, False otherwise
        """
        field_id = field.get('id')
        field_name = field.get('name', field_id)
        
        if not field_id:
            self.logger.warning(f"Field missing ID in section {section.section_id}")
            return False
        
        # Check if field is already completed
        if field_id in section.field_data and section.field_data[field_id] is not None:
            self.logger.debug(f"Field {field_name} already has value: {section.field_data[field_id]}")
            return True
        
        # Try to use available data
        if available_data:
            # Direct field ID match
            if field_id in available_data:
                value = available_data[field_id]
                if section.set_field_value(field_id, value):
                    self.logger.info(f"Set field {field_name} from available data: {value}")
                    return True
            
            # Try field name match
            if field_name.lower() in available_data:
                value = available_data[field_name.lower()]
                if section.set_field_value(field_id, value):
                    self.logger.info(f"Set field {field_name} from available data by name: {value}")
                    return True
            
            # Try intelligent matching based on field type and common patterns
            field_type = field.get('type', 'text')
            value = self._intelligent_field_matching(field, available_data)
            if value is not None and section.set_field_value(field_id, value):
                self.logger.info(f"Set field {field_name} through intelligent matching: {value}")
                return True
        
        # Use default value if available
        default_value = field.get('default_value')
        if default_value is not None:
            if section.set_field_value(field_id, default_value):
                self.logger.info(f"Set field {field_name} to default value: {default_value}")
                return True
        
        # For optional fields, we can consider them "completed" even if empty
        if not field.get('required', False):
            self.logger.debug(f"Optional field {field_name} left empty")
            return True
        
        self.logger.debug(f"Could not complete required field {field_name}")
        return False
    
    def _intelligent_field_matching(self, field: Dict[str, Any], available_data: Dict[str, Any]) -> Any:
        """
        Attempt intelligent matching of field values based on field type and common patterns.
        
        Args:
            field: Field definition
            available_data: Available data dictionary
            
        Returns:
            Matched value or None if no match found
        """
        field_type = field.get('type', 'text')
        field_name = field.get('name', '').lower()
        field_id = field.get('id', '').lower()
        
        # Common field name patterns
        patterns = {
            'email': ['email', 'email_address', 'e_mail', 'mail'],
            'phone': ['phone', 'phone_number', 'telephone', 'mobile', 'cell'],
            'name': ['name', 'full_name', 'first_name', 'last_name', 'fname', 'lname'],
            'address': ['address', 'street', 'street_address', 'addr'],
            'city': ['city', 'town'],
            'state': ['state', 'province', 'region'],
            'zip': ['zip', 'zipcode', 'postal_code', 'postcode'],
            'country': ['country', 'nation'],
            'date': ['date', 'birth_date', 'birthday', 'dob'],
            'age': ['age', 'years_old'],
            'company': ['company', 'organization', 'employer', 'business'],
            'title': ['title', 'job_title', 'position', 'role']
        }
        
        # Try to match based on field type
        if field_type == 'email':
            for key, value in available_data.items():
                if any(pattern in key.lower() for pattern in patterns['email']):
                    return value
                if '@' in str(value) and '.' in str(value):
                    return value
        
        elif field_type == 'phone':
            for key, value in available_data.items():
                if any(pattern in key.lower() for pattern in patterns['phone']):
                    return value
                # Check if value looks like a phone number
                value_str = str(value).replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
                if value_str.isdigit() and len(value_str) >= 10:
                    return value
        
        # Try to match based on field name patterns
        for pattern_type, pattern_list in patterns.items():
            if any(pattern in field_name or pattern in field_id for pattern in pattern_list):
                for key, value in available_data.items():
                    if any(pattern in key.lower() for pattern in pattern_list):
                        return value
        
        return None
    
    def process_current_section(self, available_data: Dict[str, Any] = None) -> bool:
        """
        Process the current section by attempting to fill its fields.
        
        Args:
            available_data: Optional dictionary of available data to use for field completion
            
        Returns:
            True if section was successfully processed, False otherwise
        """
        section = self.get_next_processable_section()
        if not section:
            self.logger.info("No processable sections available")
            return False
        
        self.logger.info(f"Processing section: {section.name} (ID: {section.section_id})")
        
        # Update section status
        self.section_statuses[section.section_id] = SectionStatus.IN_PROGRESS
        section.status = SectionStatus.IN_PROGRESS
        
        # Process each field in the section
        completed_fields = 0
        total_fields = len(section.fields)
        
        for field in section.fields:
            try:
                if self.attempt_field_completion(field, section, available_data):
                    completed_fields += 1
                else:
                    field_name = field.get('name', field.get('id', 'unknown'))
                    if field.get('required', False):
                        self.logger.warning(f"Failed to complete required field: {field_name}")
            except Exception as e:
                field_name = field.get('name', field.get('id', 'unknown'))
                self.logger.error(f"Error processing field {field_name}: {str(e)}")
        
        # Check if section is completed
        if section.validate_section():
            section.mark_completed(f"Completed {completed_fields}/{total_fields} fields")
            self.section_statuses[section.section_id] = SectionStatus.COMPLETED
            self.logger.info(f"Section {section.name} completed successfully")
            
            # Update form data
            self.form_data.update(section.field_data)
            
            # Add to completion history
            self.completion_history.append({
                'section_id': section.section_id,
                'section_name': section.name,
                'completed_at': self._get_timestamp(),
                'fields_completed': completed_fields,
                'total_fields': total_fields,
                'completion_percentage': section.get_completion_percentage()
            })
            
            return True
        else:
            # Section not fully completed
            required_fields = section.get_required_fields()
            missing_required = [
                field.get('name', field.get('id')) 
                for field in required_fields 
                if field.get('id') not in section.field_data or section.field_data[field.get('id')] is None
            ]
            
            if missing_required:
                self.logger.warning(f"Section {section.name} missing required fields: {', '.join(missing_required)}")
                section.status = SectionStatus.FAILED
                self.section_statuses[section.section_id] = SectionStatus.FAILED
            else:
                # Has validation errors but no missing required fields
                self.logger.info(f"Section {section.name} has validation errors: {section.validation_errors}")
                section.status = SectionStatus.FAILED
                self.section_statuses[section.section_id] = SectionStatus.FAILED
            
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def run_completion_cycle(self, available_data: Dict[str, Any] = None, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the main iterative completion cycle that processes sections sequentially.
        
        Args:
            available_data: Optional dictionary of available data to use for field completion
            max_iterations: Maximum number of iterations to prevent infinite loops
            
        Returns:
            Dictionary containing completion results and status
        """
        self.logger.info("Starting form completion cycle")
        self.form_status = FormStatus.IN_PROGRESS
        
        iteration = 0
        sections_processed_this_cycle = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"Completion cycle iteration {iteration}")
            
            # Try to process the next available section
            if self.process_current_section(available_data):
                sections_processed_this_cycle += 1
            else:
                # No section was processed, check if we're done or stuck
                next_section = self.get_next_processable_section()
                if not next_section:
                    # No more sections to process
                    break
                else:
                    # We have sections but couldn't process them
                    self.logger.warning(f"Could not process section {next_section.name}, may be missing required data")
                    break
        
        # Determine final form status
        completed_sections = sum(1 for status in self.section_statuses.values() 
                               if status == SectionStatus.COMPLETED)
        total_sections = len(self.sections)
        required_sections = sum(1 for section in self.sections if section.required)
        completed_required = sum(1 for section in self.sections 
                               if section.required and self.section_statuses.get(section.section_id) == SectionStatus.COMPLETED)
        
        if completed_required == required_sections:
            self.form_status = FormStatus.COMPLETED
            self.logger.info("Form completion cycle finished successfully")
        elif sections_processed_this_cycle == 0:
            self.form_status = FormStatus.FAILED
            self.logger.warning("Form completion cycle failed - no progress made")
        else:
            self.form_status = FormStatus.IN_PROGRESS
            self.logger.info("Form completion cycle made partial progress")
        
        # Return completion results
        return {
            'form_status': self.form_status.value,
            'iterations': iteration,
            'sections_processed': sections_processed_this_cycle,
            'completed_sections': completed_sections,
            'total_sections': total_sections,
            'required_sections': required_sections,
            'completed_required_sections': completed_required,
            'progress_percentage': (completed_sections / total_sections * 100) if total_sections > 0 else 0,
            'section_statuses': {k: v.value for k, v in self.section_statuses.items()},
            'completion_history': self.completion_history.copy(),
            'form_data': self.form_data.copy(),
            'validation_errors': self._get_all_validation_errors()
        }
    
    def _get_all_validation_errors(self) -> Dict[str, List[str]]:
        """Get all validation errors from all sections."""
        errors = {}
        for section in self.sections:
            if section.validation_errors:
                errors[section.section_id] = section.validation_errors.copy()
        return errors
    
    def get_incomplete_sections(self) -> List[FormSection]:
        """Get list of sections that are not yet completed."""
        return [
            section for section in self.sections
            if self.section_statuses.get(section.section_id) != SectionStatus.COMPLETED
        ]
    
    def get_failed_sections(self) -> List[FormSection]:
        """Get list of sections that failed to complete."""
        return [
            section for section in self.sections
            if self.section_statuses.get(section.section_id) == SectionStatus.FAILED
        ]
    
    def retry_failed_sections(self, available_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retry processing failed sections with potentially new data.
        
        Args:
            available_data: Optional dictionary of available data to use for field completion
            
        Returns:
            Dictionary containing retry results
        """
        failed_sections = self.get_failed_sections()
        if not failed_sections:
            return {'message': 'No failed sections to retry', 'sections_retried': 0}
        
        self.logger.info(f"Retrying {len(failed_sections)} failed sections")
        
        # Reset failed sections to pending
        for section in failed_sections:
            section.status = SectionStatus.PENDING
            self.section_statuses[section.section_id] = SectionStatus.PENDING
            section.validation_errors.clear()
        
        # Run completion cycle again
        return self.run_completion_cycle(available_data)




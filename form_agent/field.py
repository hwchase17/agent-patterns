"""
FormField class for representing individual form fields with validation and completion logic.
"""
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import re


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


class FormField:
    """
    Represents an individual form field with validation and completion logic.
    
    Each field contains:
    - Field metadata (ID, name, type, description)
    - Validation rules and constraints
    - Current value and completion status
    - Field-specific validation logic
    """
    
    def __init__(
        self,
        field_id: str,
        name: str,
        field_type: Union[FieldType, str],
        required: bool = False,
        description: Optional[str] = None,
        placeholder: Optional[str] = None,
        default_value: Optional[Any] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        options: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize a form field.
        
        Args:
            field_id: Unique identifier for the field
            name: Display name of the field
            field_type: Type of the field (FieldType enum or string)
            required: Whether this field is required
            description: Optional description of the field's purpose
            placeholder: Placeholder text for the field
            default_value: Default value for the field
            validation_rules: Dictionary of validation rules (min_length, max_length, pattern, etc.)
            options: List of options for select/radio/checkbox fields
            **kwargs: Additional field-specific configuration
        """
        self.field_id = field_id
        self.name = name
        self.field_type = FieldType(field_type) if isinstance(field_type, str) else field_type
        self.required = required
        self.description = description
        self.placeholder = placeholder
        self.default_value = default_value
        self.validation_rules = validation_rules or {}
        self.options = options or []
        self.extra_config = kwargs
        
        # Field state
        self.value: Optional[Any] = default_value
        self.is_completed = False
        self.validation_errors: List[str] = []
        self.completion_notes: List[str] = []
    
    def set_value(self, value: Any, validate: bool = True) -> bool:
        """
        Set the field value with optional validation.
        
        Args:
            value: Value to set for the field
            validate: Whether to validate the value before setting
            
        Returns:
            True if value was set successfully, False if validation failed
        """
        if validate and not self.validate_value(value):
            return False
        
        self.value = value
        self.is_completed = self._check_completion()
        return True
    
    def get_value(self) -> Any:
        """Get the current field value."""
        return self.value
    
    def clear_value(self):
        """Clear the field value and reset completion status."""
        self.value = self.default_value
        self.is_completed = False
        self.validation_errors.clear()
        self.completion_notes.clear()
    
    def validate_value(self, value: Optional[Any] = None) -> bool:
        """
        Validate the field value against all validation rules.
        
        Args:
            value: Value to validate (uses current value if None)
            
        Returns:
            True if value is valid, False otherwise
        """
        if value is None:
            value = self.value
        
        self.validation_errors.clear()
        
        # Check required field
        if self.required and (value is None or value == ""):
            self.validation_errors.append(f"Field '{self.name}' is required")
            return False
        
        # Skip validation if field is not required and empty
        if not self.required and (value is None or value == ""):
            return True
        
        # Type-specific validation
        if not self._validate_field_type(value):
            return False
        
        # Validation rules
        if not self._validate_rules(value):
            return False
        
        return len(self.validation_errors) == 0
    
    def _validate_field_type(self, value: Any) -> bool:
        """Validate value against field type constraints."""
        try:
            if self.field_type == FieldType.EMAIL:
                email_str = str(value)
                if '@' not in email_str or '.' not in email_str:
                    self.validation_errors.append(f"'{self.name}' must be a valid email address")
                    return False
            
            elif self.field_type == FieldType.PHONE:
                phone_str = str(value).replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
                if not phone_str.isdigit() or len(phone_str) < 10:
                    self.validation_errors.append(f"'{self.name}' must be a valid phone number")
                    return False
            
            elif self.field_type == FieldType.NUMBER:
                try:
                    float(value)
                except (ValueError, TypeError):
                    self.validation_errors.append(f"'{self.name}' must be a valid number")
                    return False
            
            elif self.field_type == FieldType.DATE:
                if isinstance(value, str):
                    try:
                        datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        try:
                            datetime.strptime(value, '%m/%d/%Y')
                        except ValueError:
                            self.validation_errors.append(f"'{self.name}' must be a valid date (YYYY-MM-DD or MM/DD/YYYY)")
                            return False
            
            elif self.field_type in [FieldType.SELECT, FieldType.RADIO]:
                if self.options and value not in self.options:
                    self.validation_errors.append(f"'{self.name}' must be one of: {', '.join(map(str, self.options))}")
                    return False
            
            elif self.field_type == FieldType.CHECKBOX:
                if not isinstance(value, (bool, list)):
                    self.validation_errors.append(f"'{self.name}' must be a boolean or list of values")
                    return False
                if isinstance(value, list) and self.options:
                    invalid_options = [v for v in value if v not in self.options]
                    if invalid_options:
                        self.validation_errors.append(f"'{self.name}' contains invalid options: {', '.join(map(str, invalid_options))}")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Validation error for '{self.name}': {str(e)}")
            return False
    
    def _validate_rules(self, value: Any) -> bool:
        """Validate value against custom validation rules."""
        value_str = str(value) if value is not None else ""
        
        # Length constraints
        min_length = self.validation_rules.get('min_length')
        max_length = self.validation_rules.get('max_length')
        
        if min_length and len(value_str) < min_length:
            self.validation_errors.append(f"'{self.name}' must be at least {min_length} characters long")
            return False
        
        if max_length and len(value_str) > max_length:
            self.validation_errors.append(f"'{self.name}' must be no more than {max_length} characters long")
            return False
        
        # Numeric constraints
        if self.field_type == FieldType.NUMBER:
            try:
                num_value = float(value)
                min_value = self.validation_rules.get('min_value')
                max_value = self.validation_rules.get('max_value')
                
                if min_value is not None and num_value < min_value:
                    self.validation_errors.append(f"'{self.name}' must be at least {min_value}")
                    return False
                
                if max_value is not None and num_value > max_value:
                    self.validation_errors.append(f"'{self.name}' must be no more than {max_value}")
                    return False
                    
            except (ValueError, TypeError):
                pass  # Type validation already handled this
        
        # Pattern matching
        pattern = self.validation_rules.get('pattern')
        if pattern:
            if not re.match(pattern, value_str):
                pattern_desc = self.validation_rules.get('pattern_description', 'the required format')
                self.validation_errors.append(f"'{self.name}' must match {pattern_desc}")
                return False
        
        return True
    
    def _check_completion(self) -> bool:
        """Check if the field is considered completed."""
        if self.required:
            return self.value is not None and self.value != "" and self.validate_value()
        else:
            # Optional fields are completed if they have a valid value or are intentionally left empty
            return self.value is None or self.value == "" or self.validate_value()
    
    def get_validation_errors(self) -> List[str]:
        """Get current validation errors."""
        return self.validation_errors.copy()
    
    def add_completion_note(self, note: str):
        """Add a note about field completion."""
        self.completion_notes.append(note)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert field to dictionary representation."""
        return {
            'field_id': self.field_id,
            'name': self.name,
            'field_type': self.field_type.value,
            'required': self.required,
            'description': self.description,
            'placeholder': self.placeholder,
            'default_value': self.default_value,
            'validation_rules': self.validation_rules.copy(),
            'options': self.options.copy(),
            'extra_config': self.extra_config.copy(),
            'value': self.value,
            'is_completed': self.is_completed,
            'validation_errors': self.validation_errors.copy(),
            'completion_notes': self.completion_notes.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormField':
        """Create a FormField instance from dictionary data."""
        field = cls(
            field_id=data['field_id'],
            name=data['name'],
            field_type=data['field_type'],
            required=data.get('required', False),
            description=data.get('description'),
            placeholder=data.get('placeholder'),
            default_value=data.get('default_value'),
            validation_rules=data.get('validation_rules', {}),
            options=data.get('options', []),
            **data.get('extra_config', {})
        )
        
        # Restore state if present
        if 'value' in data:
            field.value = data['value']
        if 'is_completed' in data:
            field.is_completed = data['is_completed']
        if 'validation_errors' in data:
            field.validation_errors = data['validation_errors'].copy()
        if 'completion_notes' in data:
            field.completion_notes = data['completion_notes'].copy()
        
        return field


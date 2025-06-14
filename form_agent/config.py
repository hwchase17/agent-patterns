"""
Configuration system for form filling agent.

This module provides a comprehensive configuration system that allows users to define
form structures, section instructions, and field mappings through JSON or YAML files.
"""
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    YML = "yml"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class FieldConfig:
    """Configuration for a form field."""
    id: str
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None
    placeholder: Optional[str] = None
    default_value: Optional[Any] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    options: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert field configuration to dictionary."""
        result = {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'required': self.required
        }
        
        if self.description:
            result['description'] = self.description
        if self.placeholder:
            result['placeholder'] = self.placeholder
        if self.default_value is not None:
            result['default_value'] = self.default_value
        if self.validation_rules:
            result['validation_rules'] = self.validation_rules
        if self.options:
            result['options'] = self.options
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldConfig':
        """Create field configuration from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            required=data.get('required', True),
            description=data.get('description'),
            placeholder=data.get('placeholder'),
            default_value=data.get('default_value'),
            validation_rules=data.get('validation_rules', {}),
            options=data.get('options')
        )


@dataclass
class SectionConfig:
    """Configuration for a form section."""
    id: str
    name: str
    instructions: str
    fields: List[FieldConfig]
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    required: bool = True
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section configuration to dictionary."""
        result = {
            'id': self.id,
            'name': self.name,
            'instructions': self.instructions,
            'fields': [field_config.to_dict() for field_config in self.fields],
            'required': self.required,
            'order': self.order
        }
        
        if self.description:
            result['description'] = self.description
        if self.dependencies:
            result['dependencies'] = self.dependencies
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectionConfig':
        """Create section configuration from dictionary."""
        fields = [FieldConfig.from_dict(field_data) for field_data in data.get('fields', [])]
        
        return cls(
            id=data['id'],
            name=data['name'],
            instructions=data['instructions'],
            fields=fields,
            description=data.get('description'),
            dependencies=data.get('dependencies', []),
            required=data.get('required', True),
            order=data.get('order', 0)
        )


@dataclass
class FormConfig:
    """Main form configuration containing all sections and metadata."""
    id: str
    name: str
    description: str
    sections: List[SectionConfig]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert form configuration to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'sections': [section.to_dict() for section in self.sections],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormConfig':
        """Create form configuration from dictionary."""
        sections = [SectionConfig.from_dict(section_data) for section_data in data.get('sections', [])]
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            sections=sections,
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {})
        )
    
    def get_section_by_id(self, section_id: str) -> Optional[SectionConfig]:
        """Get a section by its ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
    
    def get_field_by_id(self, field_id: str) -> Optional[FieldConfig]:
        """Get a field by its ID across all sections."""
        for section in self.sections:
            for field_config in section.fields:
                if field_config.id == field_id:
                    return field_config
        return None
    
    def validate_dependencies(self) -> List[str]:
        """Validate section dependencies and return any errors."""
        errors = []
        section_ids = {section.id for section in self.sections}
        
        for section in self.sections:
            for dependency in section.dependencies:
                if dependency not in section_ids:
                    errors.append(f"Section '{section.id}' depends on non-existent section '{dependency}'")
        
        return errors
    
    def get_processing_order(self) -> List[SectionConfig]:
        """Get sections in processing order, respecting dependencies."""
        # Sort by order first, then handle dependencies
        sorted_sections = sorted(self.sections, key=lambda s: s.order)
        
        # Simple topological sort for dependencies
        processed = set()
        result = []
        
        def can_process(section: SectionConfig) -> bool:
            return all(dep in processed for dep in section.dependencies)
        
        while len(result) < len(sorted_sections):
            added_any = False
            for section in sorted_sections:
                if section.id not in processed and can_process(section):
                    result.append(section)
                    processed.add(section.id)
                    added_any = True
            
            if not added_any:
                # Circular dependency or other issue
                remaining = [s for s in sorted_sections if s.id not in processed]
                raise ConfigValidationError(f"Cannot resolve dependencies for sections: {[s.id for s in remaining]}")
        
        return result


class ConfigurationManager:
    """Manages loading, validation, and manipulation of form configurations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._supported_field_types = {
            'text', 'email', 'phone', 'number', 'date', 
            'select', 'checkbox', 'radio', 'textarea', 'file'
        }
    
    def load_config(self, config_path: Union[str, Path]) -> FormConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            FormConfig object
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If configuration file doesn't exist
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine format from extension
        format_type = self._detect_format(config_path)
        
        # Load configuration data
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    config_data = json.load(f)
                else:  # YAML
                    config_data = yaml.safe_load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigValidationError(f"Failed to parse configuration file: {e}")
        
        # Validate and create configuration
        self.validate_config_data(config_data)
        form_config = FormConfig.from_dict(config_data)
        
        # Additional validation
        dependency_errors = form_config.validate_dependencies()
        if dependency_errors:
            raise ConfigValidationError(f"Dependency validation failed: {'; '.join(dependency_errors)}")
        
        self.logger.info(f"Successfully loaded configuration from {config_path}")
        return form_config
    
    def save_config(self, form_config: FormConfig, config_path: Union[str, Path], 
                   format_type: Optional[ConfigFormat] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            form_config: FormConfig object to save
            config_path: Path where to save the configuration
            format_type: Format to use (auto-detected from extension if not provided)
        """
        config_path = Path(config_path)
        
        if format_type is None:
            format_type = self._detect_format(config_path)
        
        config_data = form_config.to_dict()
        
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format_type == ConfigFormat.JSON:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:  # YAML
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def validate_config_data(self, config_data: Dict[str, Any]) -> None:
        """
        Validate configuration data structure.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required top-level fields
        required_fields = ['id', 'name', 'description', 'sections']
        for field in required_fields:
            if field not in config_data:
                raise ConfigValidationError(f"Missing required field: {field}")
        
        # Validate sections
        sections = config_data.get('sections', [])
        if not isinstance(sections, list) or not sections:
            raise ConfigValidationError("Configuration must have at least one section")
        
        section_ids = set()
        for i, section in enumerate(sections):
            self._validate_section(section, i)
            
            # Check for duplicate section IDs
            section_id = section['id']
            if section_id in section_ids:
                raise ConfigValidationError(f"Duplicate section ID: {section_id}")
            section_ids.add(section_id)
    
    def _validate_section(self, section: Dict[str, Any], index: int) -> None:
        """Validate a single section configuration."""
        # Check required section fields
        required_fields = ['id', 'name', 'instructions', 'fields']
        for field in required_fields:
            if field not in section:
                raise ConfigValidationError(f"Section {index}: Missing required field '{field}'")
        
        # Validate fields
        fields = section.get('fields', [])
        if not isinstance(fields, list) or not fields:
            raise ConfigValidationError(f"Section '{section['id']}': Must have at least one field")
        
        field_ids = set()
        for j, field_data in enumerate(fields):
            self._validate_field(field_data, section['id'], j)
            
            # Check for duplicate field IDs within section
            field_id = field_data['id']
            if field_id in field_ids:
                raise ConfigValidationError(f"Section '{section['id']}': Duplicate field ID '{field_id}'")
            field_ids.add(field_id)
    
    def _validate_field(self, field_data: Dict[str, Any], section_id: str, index: int) -> None:
        """Validate a single field configuration."""
        # Check required field fields
        required_fields = ['id', 'name', 'type']
        for field in required_fields:
            if field not in field_data:
                raise ConfigValidationError(f"Section '{section_id}', field {index}: Missing required field '{field}'")
        
        # Validate field type
        field_type = field_data['type']
        if field_type not in self._supported_field_types:
            raise ConfigValidationError(
                f"Section '{section_id}', field '{field_data['id']}': "
                f"Unsupported field type '{field_type}'. "
                f"Supported types: {', '.join(sorted(self._supported_field_types))}"
            )
        
        # Validate select/radio field options
        if field_type in ['select', 'radio'] and 'options' not in field_data:
            raise ConfigValidationError(
                f"Section '{section_id}', field '{field_data['id']}': "
                f"Field type '{field_type}' requires 'options' list"
            )
    
    def _detect_format(self, config_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        suffix = config_path.suffix.lower()
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            # Default to JSON
            return ConfigFormat.JSON
    
    def create_sample_config(self) -> FormConfig:
        """Create a sample configuration for demonstration purposes."""
        return FormConfig(
            id="sample_form",
            name="Sample Registration Form",
            description="A sample multi-section registration form",
            sections=[
                SectionConfig(
                    id="personal_info",
                    name="Personal Information",
                    instructions="Fill out your basic personal information",
                    order=1,
                    fields=[
                        FieldConfig(id="first_name", name="First Name", type="text", required=True),
                        FieldConfig(id="last_name", name="Last Name", type="text", required=True),
                        FieldConfig(id="email", name="Email Address", type="email", required=True),
                        FieldConfig(id="phone", name="Phone Number", type="phone", required=False)
                    ]
                ),
                SectionConfig(
                    id="address_info",
                    name="Address Information",
                    instructions="Provide your current address details",
                    order=2,
                    dependencies=["personal_info"],
                    fields=[
                        FieldConfig(id="street", name="Street Address", type="text", required=True),
                        FieldConfig(id="city", name="City", type="text", required=True),
                        FieldConfig(id="state", name="State", type="select", required=True, 
                                  options=["CA", "NY", "TX", "FL", "Other"]),
                        FieldConfig(id="zip_code", name="ZIP Code", type="text", required=True)
                    ]
                )
            ]
        )


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import sys
import re


@dataclass
class ResourceConfig:
    """Configuration for a single resource."""
    name: str
    kind: Optional[str] = None
    source: Optional[str] = None
    resource: Optional[str] = None
    resource_args: Dict[str, Any] = field(default_factory=dict)
    get: bool = False
    file_name: Optional[str] = None
    target: Optional[str] = None  # Path within the extracted archive to use as the source
    cmd: Optional[str] = None  # Command to generate the documentation
    subject: str = field(default="", init=False)  # Will be set after initialization
    
    def __post_init__(self):
        """Validate the resource configuration."""
        # Name is required and can't be empty
        if not self.name or not self.name.strip():
            raise ValueError(f"Resource name cannot be empty")
        
        # Validate name format - no special characters that would cause path issues
        if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', self.name):
            print(f"Warning: Resource name '{self.name}' contains special characters that might cause path issues")
        
        # Either source, resource, or cmd must be specified
        if self.source is None and self.resource is None and self.cmd is None:
            raise ValueError(f"Resource '{self.name}' must specify either 'source', 'resource', or 'cmd'")
            
        # Source, resource, and cmd should be mutually exclusive
        if sum(x is not None for x in [self.source, self.resource, self.cmd]) > 1:
            raise ValueError(f"Resource '{self.name}' can only have one of 'source', 'resource', or 'cmd' specified")
            
        # Check if source path exists when specified
        if self.source is not None and not Path(self.source).exists():
            print(f"Warning: Source path for '{self.name}' does not exist: {self.source}")
            
        # Check kind values - don't enforce a specific set since custom types are allowed
        if self.kind and not isinstance(self.kind, str):
            raise ValueError(f"Resource '{self.name}' has invalid kind: {self.kind}")
            
        # Check get flag consistency - only for resources with resource URL
        if self.get and self.resource is None:
            raise ValueError(f"Resource '{self.name}' has 'get=True' but no 'resource' specified")
            
        # If target is specified, ensure it's a string
        if self.target and not isinstance(self.target, str):
            raise ValueError(f"Resource '{self.name}' has invalid target: {self.target}")
            
        # If cmd is specified, ensure it's a string
        if self.cmd and not isinstance(self.cmd, str):
            raise ValueError(f"Resource '{self.name}' has invalid cmd: {self.cmd}")
            
        # If resource and get are specified, check resource_args used in resource
        if self.resource is not None and self.get:
            # Check if resource contains format specifiers
            if '{' in self.resource and '}' in self.resource:
                # Extract format specifiers from resource string
                format_vars = []
                i = 0
                while i < len(self.resource):
                    if self.resource[i] == '{':
                        start = i
                        i += 1
                        while i < len(self.resource) and self.resource[i] != '}':
                            i += 1
                        if i < len(self.resource) and self.resource[i] == '}':
                            format_vars.append(self.resource[start+1:i])
                    i += 1
                
                # Check if all format specifiers are in resource_args
                for var in format_vars:
                    if var not in self.resource_args:
                        print(f"Warning: Resource '{self.name}' uses '{var}' in URL but it's not in resource_args")
            
            # If file_name is specified and contains format specifiers, check those too
            if self.file_name and '{' in self.file_name and '}' in self.file_name:
                # Extract format specifiers from file_name string
                format_vars = []
                i = 0
                while i < len(self.file_name):
                    if self.file_name[i] == '{':
                        start = i
                        i += 1
                        while i < len(self.file_name) and self.file_name[i] != '}':
                            i += 1
                        if i < len(self.file_name) and self.file_name[i] == '}':
                            format_vars.append(self.file_name[start+1:i])
                    i += 1
                
                # Check if all format specifiers are in resource_args
                for var in format_vars:
                    if var not in self.resource_args:
                        print(f"Warning: Resource '{self.name}' uses '{var}' in file_name but it's not in resource_args")
                        
            # If target is specified and contains format specifiers, check those too
            if self.target and '{' in self.target and '}' in self.target:
                # Extract format specifiers from target string
                format_vars = []
                i = 0
                while i < len(self.target):
                    if self.target[i] == '{':
                        start = i
                        i += 1
                        while i < len(self.target) and self.target[i] != '}':
                            i += 1
                        if i < len(self.target) and self.target[i] == '}':
                            format_vars.append(self.target[start+1:i])
                    i += 1
                
                # Check if all format specifiers are in resource_args
                for var in format_vars:
                    if var not in self.resource_args:
                        print(f"Warning: Resource '{self.name}' uses '{var}' in target but it's not in resource_args")


def validate_resources(yaml_path: str) -> Dict[str, List[ResourceConfig]]:
    """Validate a resources YAML file and return the validated config."""
    
    # Read the YAML file
    try:
        with open(yaml_path, 'r') as f:
            raw_resources = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)
    
    if not isinstance(raw_resources, dict):
        print(f"Error: Resources file must be a dictionary")
        sys.exit(1)
        
    # Validate subject names
    invalid_subjects = []
    for subject in raw_resources.keys():
        if not re.match(r'^[a-zA-Z0-9_\-]+$', subject):
            invalid_subjects.append(subject)
    
    if invalid_subjects:
        print(f"Warning: The following subjects contain invalid characters for directory names: {', '.join(invalid_subjects)}")
        print("Subject names should only contain letters, numbers, underscores, and hyphens.")
    
    # Check for resource name collisions within subjects
    all_resource_paths = {}
    
    # Validate each resource
    validated_resources = {}
    errors = []
    
    for subject, resources in raw_resources.items():
        if not isinstance(resources, list):
            errors.append(f"Error: Resources for '{subject}' must be a list")
            continue
            
        validated_resources[subject] = []
        subject_resources = set()
        
        for i, resource_dict in enumerate(resources):
            try:
                # Convert to dataclass for validation
                resource = ResourceConfig(**resource_dict)
                
                # Set the subject on the resource
                resource.subject = subject
                
                # Check for duplicate resource names within subject
                resource_path = f"{subject}/{resource.name.lower().replace(' ', '_')}"
                if resource_path in all_resource_paths:
                    print(f"Warning: Resource '{resource.name}' in subject '{subject}' has the same normalized path as another resource")
                
                all_resource_paths[resource_path] = resource
                subject_resources.add(resource.name.lower().replace(' ', '_'))
                
                validated_resources[subject].append(resource)
            except Exception as e:
                errors.append(f"Error in '{subject}' resource {i+1}: {str(e)}")
    
    # Report validation results
    if errors:
        print(f"Found {len(errors)} error(s) in resources file:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Check for existing resource directories and validate structure
    cache_dir = Path("cache")
    if cache_dir.exists():
        md_resources_dir = cache_dir / "md_resources"
        if md_resources_dir.exists():
            # Check for orphaned resource directories
            orphaned_dirs = []
            for subject_dir in md_resources_dir.glob("*"):
                if subject_dir.is_dir():
                    subject_name = subject_dir.name
                    if subject_name not in validated_resources:
                        orphaned_dirs.append(str(subject_dir))
                    else:
                        # Check for orphaned resource directories within subject
                        resource_names = {r.name.lower().replace(' ', '_') for r in validated_resources[subject_name]}
                        for resource_dir in subject_dir.glob("*"):
                            if resource_dir.is_dir() and resource_dir.name not in resource_names:
                                orphaned_dirs.append(str(resource_dir))
            
            if orphaned_dirs:
                print(f"\nWarning: Found {len(orphaned_dirs)} orphaned resource directories that are not in the YAML config:")
                for dir_path in orphaned_dirs[:10]:  # Show first 10 to avoid flooding output
                    print(f"  - {dir_path}")
                if len(orphaned_dirs) > 10:
                    print(f"  - ... and {len(orphaned_dirs) - 10} more")
    
    print(f"\nResource validation successful: {len(validated_resources)} subject(s) with resources")
    total_resources = sum(len(resources) for resources in validated_resources.values())
    print(f"Total resources: {total_resources}")
    for subject, resources in validated_resources.items():
        print(f"  - {subject}: {len(resources)} resource(s)")
    
    return validated_resources


def check_output_structure(resources_dict: Dict[str, List[ResourceConfig]]) -> bool:
    """Check the output directory structure against the expected structure from resources."""
    cache_dir = Path("cache")
    md_resources_dir = cache_dir / "md_resources"
    
    if not md_resources_dir.exists():
        print(f"Output directory '{md_resources_dir}' does not exist yet.")
        return True
    
    mismatches = []
    
    # Check each subject
    for subject, resources in resources_dict.items():
        subject_dir = md_resources_dir / subject
        if not subject_dir.exists():
            mismatches.append(f"Missing subject directory: {subject_dir}")
            continue
        
        # Check each resource
        for resource in resources:
            resource_name = resource.name.lower().replace(' ', '_')
            resource_dir = subject_dir / resource_name
            
            if not resource_dir.exists() and not resource_dir.is_symlink():
                mismatches.append(f"Missing resource directory: {resource_dir}")
    
    if mismatches:
        print(f"\nFound {len(mismatches)} structure mismatches:")
        for mismatch in mismatches[:10]:  # Show first 10 to avoid flooding output
            print(f"  - {mismatch}")
        if len(mismatches) > 10:
            print(f"  - ... and {len(mismatches) - 10} more")
        return False
    
    return True


def main():
    """Main function to validate resources."""
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = "resources.yml"
    
    print(f"Validating resources file: {yaml_path}")
    resources = validate_resources(yaml_path)
    
    # Optionally check output structure
    if len(sys.argv) > 2 and sys.argv[2] == "--check-structure":
        print("\nChecking output directory structure...")
        check_output_structure(resources)


if __name__ == "__main__":
    main() 
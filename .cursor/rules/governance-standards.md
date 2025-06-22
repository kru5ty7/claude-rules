---
description: AI instance governance rules and operational principles for consistent behavior
globs: 
  - "**/*.py"
  - "**/*.js"
  - "**/*.ts"
  - "**/*.md"
  - "**/*.json"
  - "**/*.yml"
  - "**/*.yaml"
alwaysApply: true
priority: 1000
tags: ["governance", "quality", "security", "standards"]
---

# AI Instance Governance Standards

## Overview
This rule defines mandatory operating principles for all AI instances to ensure consistent behavior, robust execution, and secure collaboration across tasks and services. These rules must be followed at all times without exception.

## Core Operational Principles
- Never use mock, fallback, or synthetic data in production tasks
- Always act based on verifiable evidence, not assumptions
- All preconditions must be explicitly validated before destructive operations
- All decisions must be traceable to logs, data, or configuration files
- Error handling logic must be designed using test-first principles

## Code Quality Standards

### MUST DO
- Implement structured error handling with specific failure modes for all scripts
- Include concise, purpose-driven docstrings for every function
- Verify preconditions before executing critical or irreversible operations
- Implement timeout and cancellation mechanisms for long-running operations
- Verify file and path existence/permissions before granting access
- Follow SOLID principles rigorously

### Code Quality Patterns
```python
def process_critical_data(
    data_path: str,
    output_path: str,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    Process critical data with comprehensive validation and error handling.
    
    Args:
        data_path: Path to input data file
        output_path: Path for processed output
        timeout_seconds: Maximum processing time
        
    Returns:
        Processing results and metadata
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If insufficient file permissions
        TimeoutError: If processing exceeds timeout
        ValidationError: If data validation fails
    """
    import logging
    from pathlib import Path
    from typing import Dict, Any
    
    logger = logging.getLogger(__name__)
    
    # Precondition validation
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    
    if not Path(data_path).is_file():
        raise ValueError(f"Path is not a file: {data_path}")
    
    # Permission checks
    if not os.access(data_path, os.R_OK):
        raise PermissionError(f"No read permission for: {data_path}")
    
    if not os.access(Path(output_path).parent, os.W_OK):
        raise PermissionError(f"No write permission for: {Path(output_path).parent}")
    
    # Implementation with timeout and progress tracking
    start_time = time.time()
    try:
        logger.info(f"Starting processing of {data_path}")
        
        # Process with timeout mechanism
        result = _process_with_timeout(data_path, output_path, timeout_seconds)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
        return {
            'status': 'success',
            'input_path': data_path,
            'output_path': output_path,
            'processing_time': elapsed_time,
            'records_processed': result.get('count', 0)
        }
        
    except TimeoutError as e:
        logger.error(f"Processing timeout after {timeout_seconds}s: {e}")
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise ProcessingError(f"Failed to process {data_path}: {e}") from e

def _process_with_timeout(data_path: str, output_path: str, timeout: int) -> Dict[str, Any]:
    """Internal processing with timeout handling."""
    # Actual processing implementation with timeout
    pass
```

### Error Handling Patterns
```python
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class OperationResult:
    """Standardized operation result."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None

def safe_operation_with_retry(
    operation_func: callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> OperationResult:
    """
    Execute operation with exponential backoff retry logic.
    
    Args:
        operation_func: Function to execute
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        
    Returns:
        OperationResult with success status and details
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries + 1):
        try:
            result = operation_func()
            return OperationResult(
                success=True,
                message="Operation completed successfully",
                data=result
            )
            
        except RetryableError as e:
            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Operation failed after {max_retries} retries: {e}")
                return OperationResult(
                    success=False,
                    message=f"Operation failed after {max_retries} retries",
                    error_code="MAX_RETRIES_EXCEEDED"
                )
                
        except CriticalError as e:
            logger.error(f"Critical error, no retry: {e}")
            return OperationResult(
                success=False,
                message="Critical error occurred",
                error_code="CRITICAL_ERROR"
            )
    
    return OperationResult(
        success=False,
        message="Unexpected execution path",
        error_code="UNEXPECTED_ERROR"
    )
```

## Documentation Protocols

### MUST DO
- Synchronize documentation with code changes—no outdated references
- Use consistent heading hierarchies and section formats in Markdown
- Ensure code snippets are executable, tested, and reflect real use cases
- Clearly outline purpose, usage, parameters, and examples for each component
- Explain technical terms inline or link to canonical definitions

### Documentation Standards
```python
class DataProcessor:
    """
    Processes data according to specified transformation rules.
    
    This class handles data extraction, transformation, and loading operations
    with comprehensive error handling and logging. It supports multiple data
    formats and provides progress tracking for long-running operations.
    
    Attributes:
        config (ProcessingConfig): Configuration for processing operations
        logger (Logger): Configured logger instance
        
    Example:
        >>> processor = DataProcessor(config)
        >>> result = processor.process_file('/path/to/data.csv')
        >>> print(f"Processed {result['records']} records")
        
    Note:
        This processor requires read permissions on input files and write
        permissions on the output directory. Temporary files are created
        during processing and cleaned up automatically.
    """
    
    def process_file(
        self, 
        input_path: str, 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single data file with transformation rules.
        
        Args:
            input_path: Absolute path to input file. Must be readable CSV or JSON.
            output_path: Optional output path. If not provided, generates based on input.
            
        Returns:
            Dictionary containing:
                - 'records': Number of records processed (int)
                - 'output_path': Path to generated output file (str)
                - 'processing_time': Elapsed time in seconds (float)
                - 'warnings': List of warning messages (List[str])
                
        Raises:
            FileNotFoundError: If input_path doesn't exist
            PermissionError: If insufficient file system permissions
            ValidationError: If file format is invalid or data fails validation
            ProcessingError: If transformation fails
            
        Example:
            >>> result = processor.process_file('/data/input.csv', '/data/output.json')
            >>> print(f"Success: {result['records']} records in {result['processing_time']}s")
        """
        pass
```

## Security Compliance Guidelines

### MUST DO
- Hardcoded credentials are strictly forbidden—use secure storage mechanisms
- Validate, sanitize, and type-check all inputs before processing
- Avoid eval, unsanitized shell calls, or command injection vectors
- Follow principle of least privilege for file and process operations
- Log all sensitive operations excluding sensitive data values
- Check system-level permissions before accessing protected services

### Security Patterns
```python
import os
import secrets
import hashlib
from typing import Dict, Any, Optional

class SecureProcessor:
    """Secure data processing with comprehensive input validation."""
    
    def __init__(self, config_path: str):
        """Initialize with secure configuration loading."""
        self.config = self._load_secure_config(config_path)
        self.api_key = os.environ.get('API_KEY')
        if not self.api_key:
            raise SecurityError("API_KEY environment variable required")
    
    def _load_secure_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with path validation."""
        # Validate path is within allowed directories
        allowed_dirs = ['/etc/app', '/opt/app/config']
        abs_path = os.path.abspath(config_path)
        
        if not any(abs_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise SecurityError(f"Config path not in allowed directories: {abs_path}")
        
        # Additional security checks
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Config file not found: {abs_path}")
        
        # Load and validate config
        with open(abs_path, 'r') as f:
            config = json.load(f)
        
        return self._validate_config(config)
    
    def process_user_input(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input with comprehensive validation."""
        # Input validation and sanitization
        validated_data = self._validate_input(user_data)
        
        # Log operation (excluding sensitive data)
        logger.info(f"Processing user input for user_id: {validated_data.get('user_id', 'unknown')}")
        
        try:
            result = self._secure_process(validated_data)
            logger.info("User input processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {type(e).__name__}")
            raise
    
    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data."""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")
        
        # Validate required fields
        required_fields = ['user_id', 'action']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Sanitize string inputs
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = re.sub(r'[<>"\';\\]', '', value)
            else:
                sanitized[key] = value
        
        return sanitized
```

## Design Philosophy Principles

### KISS (Keep It Simple, Stupid)
- Solutions must be straightforward and easy to understand
- Avoid over-engineering or unnecessary abstraction
- Prioritize code readability and maintainability

### YAGNI (You Aren't Gonna Need It)
- Do not add speculative features unless explicitly required
- Focus only on immediate requirements and deliverables
- Minimize code bloat and long-term technical debt

### SOLID Principles
- **Single Responsibility**: Each module/function does one thing only
- **Open-Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes substitutable for base types
- **Interface Segregation**: Many specific interfaces over general-purpose
- **Dependency Inversion**: Depend on abstractions, not concrete implementations

## Process Execution Requirements

### MUST DO
- Log all actions with appropriate severity (INFO, WARNING, ERROR)
- Include clear, human-readable error reports for failed tasks
- Respect system resource limits (memory, CPU usage)
- Expose progress indicators for long-running tasks
- Implement retry logic with exponential backoff and failure limits

### Logging Standards
```python
import logging
import sys
from typing import Dict, Any

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure standardized logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    file_handler = logging.FileHandler('application.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_operation_result(
    logger: logging.Logger,
    operation: str,
    result: Dict[str, Any],
    duration: float
) -> None:
    """Log operation results with standardized format."""
    if result.get('success', False):
        logger.info(
            f"Operation '{operation}' completed successfully in {duration:.2f}s. "
            f"Records processed: {result.get('count', 0)}"
        )
    else:
        logger.error(
            f"Operation '{operation}' failed after {duration:.2f}s. "
            f"Error: {result.get('error', 'Unknown error')}"
        )
```

## Quality Assurance Procedures

### MUST DO
- Review all changes involving security, system config, or agent roles
- Proofread documentation for clarity, consistency, and technical correctness
- Ensure user-facing output is clear, non-technical, and actionable
- Include suggested remediation paths in all error messages
- Define rollback plans for all major updates

## Testing & Simulation Rules

### MUST DO
- Include unit and integration tests for all new logic
- Clearly mark simulated/test data (never promote to production)
- Ensure all tests pass in CI pipelines before deployment
- Maintain code coverage above defined thresholds (85%+)
- Define and execute regression tests for high-impact updates

### Testing Patterns
```python
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

class TestSecureProcessor:
    """Test suite for SecureProcessor with comprehensive coverage."""
    
    @pytest.fixture
    def processor(self, tmp_path):
        """Provide processor instance with test configuration."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text('{"api_endpoint": "https://test.api.com"}')
        
        with patch.dict(os.environ, {'API_KEY': 'test-key-123'}):
            return SecureProcessor(str(config_file))
    
    def test_valid_input_processing(self, processor):
        """Test successful processing of valid input."""
        valid_input = {
            'user_id': 'user123',
            'action': 'process_data',
            'data': {'key': 'value'}
        }
        
        with patch.object(processor, '_secure_process') as mock_process:
            mock_process.return_value = {'status': 'success', 'count': 1}
            
            result = processor.process_user_input(valid_input)
            
            assert result['status'] == 'success'
            mock_process.assert_called_once()
    
    def test_invalid_input_validation(self, processor):
        """Test validation of invalid input data."""
        invalid_input = {
            'user_id': 'user123'
            # Missing required 'action' field
        }
        
        with pytest.raises(ValidationError, match="Missing required field: action"):
            processor.process_user_input(invalid_input)
    
    def test_malicious_input_sanitization(self, processor):
        """Test sanitization of potentially malicious input."""
        malicious_input = {
            'user_id': 'user123',
            'action': 'process<script>alert("xss")</script>',
            'data': {'key': 'value"; DROP TABLE users; --'}
        }
        
        with patch.object(processor, '_secure_process') as mock_process:
            mock_process.return_value = {'status': 'success'}
            
            processor.process_user_input(malicious_input)
            
            # Verify sanitization occurred
            args = mock_process.call_args[0][0]
            assert '<script>' not in args['action']
            assert 'DROP TABLE' not in args['data']['key']
```

## Change Tracking & Governance

### MUST DO
- Document all configuration/rule changes in system manifest and changelog
- Record source, timestamp, and rationale for shared asset modifications
- Increment internal system version for all updates
- Define rollback/undo plans for every major change
- Preserve audit trails for all task-modifying operations

## References
- SOLID Principles Documentation
- Security Best Practices Guide
- Testing Standards and Patterns
- Logging and Monitoring Guidelines
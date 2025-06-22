---
description: Python development standards and best practices
globs: 
  - "**/*.py"
  - "!**/migrations/*.py"
  - "!**/__pycache__/**"
  - "!**/venv/**"
  - "!**/.venv/**"
alwaysApply: false
priority: 200
tags: ["python", "backend", "development"]
---

# Python Development Standards

## Overview
This rule defines coding standards, patterns, and best practices for Python development. It covers code organization, typing, error handling, and performance considerations.

## Core Principles
- Write clean, readable, and well-documented code
- Use type hints consistently for better code maintainability
- Follow PEP 8 style guidelines with modern Python practices
- Implement proper error handling and logging
- Write testable and modular code

## Guidelines

### MUST DO
- Use type hints for all function parameters and return values
- Follow import organization: standard library, third-party, local imports
- Use dataclasses or Pydantic models for structured data
- Implement proper error handling with specific exception types
- Add docstrings to all classes and public methods
- Use f-strings for string formatting
- Follow naming conventions: snake_case for variables/functions, PascalCase for classes

### SHOULD DO
- Use pathlib for file operations instead of os.path
- Prefer list/dict comprehensions over loops when readable
- Use context managers for resource management
- Implement logging instead of print statements
- Use pytest for testing with meaningful test names
- Use async/await for I/O bound operations

### AVOID
- Using bare except clauses
- Importing with * (star imports)
- Using mutable default arguments
- Long functions (>50 lines) without clear separation
- Global variables
- Using eval() or exec()

## Code Patterns

### Import Organization
```python
# Standard library
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

# Third-party
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine
from pydantic import BaseModel, Field

# Local imports
from src.models.user import User
from src.utils.validators import validate_email
from src.services.database import DatabaseService
```

### Function Patterns
```python
def process_user_data(
    user_data: Dict[str, Any],
    *,  # Force keyword-only arguments
    validate: bool = True,
    batch_size: Optional[int] = None,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Process user data with optional validation and batching.
    
    Args:
        user_data: Dictionary containing user information
        validate: Whether to validate input data
        batch_size: Process in batches if specified
        timeout: Operation timeout in seconds
        
    Returns:
        List of processed user records
        
    Raises:
        ValueError: If validation fails or data is invalid
        TimeoutError: If operation exceeds timeout
        ProcessingError: If data processing fails
    """
    if not user_data:
        raise ValueError("User data cannot be empty")
    
    if validate:
        _validate_user_data(user_data)
    
    try:
        return _apply_data_transformations(user_data, batch_size, timeout)
    except Exception as e:
        logger.error(f"Failed to process user data: {e}")
        raise ProcessingError(f"Data processing failed: {e}") from e
```

### Class Patterns
```python
@dataclass
class UserConfig:
    """Configuration for user processing operations."""
    
    # Class variables
    DEFAULT_TIMEOUT: ClassVar[float] = 30.0
    MAX_RETRIES: ClassVar[int] = 3
    
    # Instance variables
    user_id: str
    email: str
    timeout: float = field(default=DEFAULT_TIMEOUT)
    retry_count: int = field(default=MAX_RETRIES)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if not self.email or "@" not in self.email:
            raise ValueError("Valid email is required")

# Pydantic model alternative
class UserModel(BaseModel):
    """User data model with validation."""
    
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email address")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")
    is_active: bool = Field(default=True, description="User status")
    
    class Config:
        validate_assignment = True
        extra = "forbid"
```

### Async Patterns
```python
import asyncio
import aiohttp
from typing import List, Dict

async def fetch_user_data_async(user_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch user data from multiple sources concurrently."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = [fetch_single_user(session, user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch user data: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

async def fetch_single_user(session: aiohttp.ClientSession, user_id: str) -> Dict[str, Any]:
    """Fetch single user data with error handling."""
    try:
        async with session.get(f"/api/users/{user_id}") as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        raise UserFetchError(f"Failed to fetch user {user_id}: {e}") from e
```

## Error Handling
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Raised when data processing fails."""
    pass

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

def safe_process_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process data with comprehensive error handling."""
    try:
        # Validate input
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dict, got {type(data)}")
        
        # Process data
        result = transform_data(data)
        
        # Validate output
        if not result:
            logger.warning("Processing returned empty result")
            return None
            
        return result
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise  # Re-raise validation errors
        
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
        return None  # Return None for processing errors
        
    except Exception as e:
        logger.exception(f"Unexpected error in data processing: {e}")
        raise ProcessingError(f"Unexpected error: {e}") from e
```

## Testing Requirements
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Generator

class TestUserService:
    """Test suite for UserService."""
    
    @pytest.fixture
    def user_service(self) -> UserService:
        """Provide UserService instance."""
        return UserService()
    
    @pytest.fixture
    def sample_user_data(self) -> Dict[str, Any]:
        """Provide sample user data for testing."""
        return {
            "user_id": "12345",
            "email": "test@example.com",
            "name": "Test User"
        }
    
    @pytest.fixture
    def mock_database(self) -> Generator[Mock, None, None]:
        """Provide mocked database connection."""
        with patch('src.services.user.database') as mock_db:
            yield mock_db
    
    def test_create_user_success(
        self, 
        user_service: UserService, 
        sample_user_data: Dict[str, Any],
        mock_database: Mock
    ) -> None:
        """Test successful user creation."""
        # Arrange
        expected_user = {"id": 1, **sample_user_data}
        mock_database.save.return_value = expected_user
        
        # Act
        result = user_service.create_user(sample_user_data)
        
        # Assert
        assert result["id"] == 1
        assert result["email"] == sample_user_data["email"]
        mock_database.save.assert_called_once_with(sample_user_data)
    
    def test_create_user_validation_error(
        self, 
        user_service: UserService
    ) -> None:
        """Test user creation with invalid data."""
        # Arrange
        invalid_data = {"user_id": "", "email": "invalid-email"}
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Valid email is required"):
            user_service.create_user(invalid_data)
    
    @pytest.mark.asyncio
    async def test_async_fetch_user(
        self, 
        user_service: UserService,
        sample_user_data: Dict[str, Any]
    ) -> None:
        """Test async user fetching."""
        # Arrange
        with patch.object(user_service, 'fetch_user_async', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_user_data
            
            # Act
            result = await user_service.fetch_user_async("12345")
            
            # Assert
            assert result == sample_user_data
            mock_fetch.assert_called_once_with("12345")
```

## Performance Considerations
- Use `__slots__` for classes with many instances
- Prefer generators over lists for large datasets
- Use `functools.lru_cache` for expensive computations
- Profile code with `cProfile` before optimizing
- Use `asyncio` for I/O bound operations
- Consider using `numpy` for numerical computations
- Use connection pooling for database operations

## Security Checklist
- [ ] Input validation implemented
- [ ] SQL injection prevention (parameterized queries)
- [ ] No hardcoded secrets or credentials
- [ ] Proper authentication and authorization
- [ ] Sensitive data logging avoided
- [ ] Environment variables used for configuration
- [ ] Dependencies regularly updated

## Logging Pattern
```python
import logging
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

def process_with_logging(data: Any) -> Any:
    """Example of proper logging implementation."""
    logger.info(f"Starting processing for {len(data) if hasattr(data, '__len__') else 'unknown'} items")
    
    try:
        result = expensive_operation(data)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

## References
- PEP 8: Style Guide for Python Code
- PEP 484: Type Hints
- Python 3.11+ Documentation
- pytest Documentation
- Pydantic Documentation
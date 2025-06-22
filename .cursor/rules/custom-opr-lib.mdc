---
description: Custom Airflow operator library development standards and patterns
globs: 
  - "**/operators/**/*.py"
  - "**/custom_operators/**/*.py" 
  - "**/plugins/**/*.py"
  - "**/*operator*.py"
  - "**/hooks/**/*.py"
  - "**/sensors/**/*.py"
alwaysApply: false
priority: 250
tags: ["airflow", "operators", "custom", "plugins", "hooks"]
---

# Custom Airflow Operator Library Development Standards

## Overview
This rule defines standards for developing custom Airflow operators, hooks, and sensors that can be reused across multiple DAGs and projects. Focus on creating robust, testable, and well-documented components that extend Airflow's functionality.

## Core Principles
- Design operators to be idempotent and atomic
- Implement comprehensive error handling and logging
- Create reusable and configurable components
- Follow Airflow's plugin architecture patterns
- Ensure operators are testable in isolation
- Provide clear documentation and examples
- Implement proper resource management

## Guidelines

### MUST DO
- Inherit from appropriate base classes (BaseOperator, BaseHook, BaseSensor)
- Implement proper `__init__` method with `@apply_defaults` decorator
- Define `template_fields` for dynamic templating
- Implement comprehensive logging throughout execution
- Handle exceptions with specific error types
- Provide clear docstrings with parameter descriptions
- Implement proper cleanup in finally blocks
- Use connection management for external resources

### SHOULD DO
- Implement dry-run functionality where applicable
- Provide configuration validation in `__init__`
- Use type hints for all parameters and return values
- Create factory methods for common configurations
- Implement progress tracking for long-running operations
- Provide callback mechanisms for custom handling
- Use connection pooling for database operations
- Implement timeout handling

### AVOID
- Hardcoding configuration values
- Performing heavy operations in `__init__`
- Using bare except clauses
- Ignoring connection cleanup
- Creating overly complex single-purpose operators
- Using global state or shared mutable objects
- Blocking operations without timeout

## Code Patterns

### Base Custom Operator Structure
```python
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import timedelta
import logging

from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.configuration import conf

logger = logging.getLogger(__name__)

class CustomDataProcessorOperator(BaseOperator):
    """
    Custom operator for data processing operations.
    
    This operator handles data extraction, transformation, and loading
    with comprehensive error handling and monitoring.
    
    :param source_conn_id: Connection ID for source system
    :type source_conn_id: str
    :param target_conn_id: Connection ID for target system  
    :type target_conn_id: str
    :param processing_config: Configuration for data processing
    :type processing_config: Dict[str, Any]
    :param batch_size: Number of records to process in each batch
    :type batch_size: int
    :param timeout: Operation timeout in seconds
    :type timeout: int
    :param dry_run: Whether to perform a dry run without actual changes
    :type dry_run: bool
    :param on_success_callback: Function to call on successful completion
    :type on_success_callback: Optional[Callable]
    :param on_failure_callback: Function to call on failure
    :type on_failure_callback: Optional[Callable]
    """
    
    # Define template fields for Jinja templating
    template_fields = [
        'source_conn_id', 
        'target_conn_id', 
        'processing_config',
        'batch_size'
    ]
    
    # Define template extension for config files
    template_ext = ['.sql', '.json', '.yml']
    
    # UI color for Airflow web interface
    ui_color = '#89CDF1'
    ui_fgcolor = '#000000'
    
    @apply_defaults
    def __init__(
        self,
        source_conn_id: str,
        target_conn_id: str,
        processing_config: Dict[str, Any],
        batch_size: int = 1000,
        timeout: int = 3600,
        dry_run: bool = False,
        on_success_callback: Optional[Callable] = None,
        on_failure_callback: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Store configuration
        self.source_conn_id = source_conn_id
        self.target_conn_id = target_conn_id
        self.processing_config = processing_config
        self.batch_size = batch_size
        self.timeout = timeout
        self.dry_run = dry_run
        self.on_success_callback = on_success_callback
        self.on_failure_callback = on_failure_callback
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate operator configuration."""
        if not self.source_conn_id:
            raise ValueError("source_conn_id is required")
        
        if not self.target_conn_id:
            raise ValueError("target_conn_id is required")
        
        if not isinstance(self.processing_config, dict):
            raise ValueError("processing_config must be a dictionary")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data processing operation.
        
        :param context: Airflow context dictionary
        :return: Execution results and metadata
        """
        logger.info(f"Starting data processing with batch_size={self.batch_size}")
        
        if self.dry_run:
            logger.info("Dry run mode enabled - no actual changes will be made")
        
        try:
            # Initialize connections and resources
            result = self._execute_processing(context)
            
            # Call success callback if provided
            if self.on_success_callback:
                self.on_success_callback(context, result)
            
            logger.info("Data processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            
            # Call failure callback if provided
            if self.on_failure_callback:
                self.on_failure_callback(context, e)
            
            raise AirflowException(f"Processing failed: {e}") from e
    
    def _execute_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main processing logic."""
        from .hooks.custom_data_hook import CustomDataHook
        
        # Get hooks for source and target systems
        source_hook = CustomDataHook(conn_id=self.source_conn_id)
        target_hook = CustomDataHook(conn_id=self.target_conn_id)
        
        try:
            # Extract data from source
            logger.info("Extracting data from source system")
            data = self._extract_data(source_hook, context)
            
            # Transform data according to configuration
            logger.info(f"Transforming {len(data)} records")
            transformed_data = self._transform_data(data, context)
            
            # Load data to target system
            if not self.dry_run:
                logger.info("Loading data to target system")
                load_result = self._load_data(target_hook, transformed_data, context)
            else:
                logger.info("Dry run: Skipping data load")
                load_result = {'dry_run': True, 'would_load': len(transformed_data)}
            
            return {
                'extracted_count': len(data),
                'transformed_count': len(transformed_data),
                'load_result': load_result,
                'processing_config': self.processing_config,
                'execution_timestamp': context['ts']
            }
            
        finally:
            # Ensure connections are properly closed
            source_hook.close()
            target_hook.close()
    
    def _extract_data(self, hook: 'CustomDataHook', context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from source system."""
        query_config = self.processing_config.get('extract', {})
        
        return hook.extract_data(
            query=query_config.get('query'),
            params=query_config.get('params', {}),
            batch_size=self.batch_size
        )
    
    def _transform_data(self, data: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform data according to configuration."""
        transform_config = self.processing_config.get('transform', {})
        
        # Apply transformations
        for transform in transform_config.get('steps', []):
            data = self._apply_transformation(data, transform, context)
        
        return data
    
    def _apply_transformation(
        self, 
        data: List[Dict[str, Any]], 
        transform: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply a single transformation step."""
        transform_type = transform.get('type')
        
        if transform_type == 'filter':
            return self._filter_data(data, transform.get('condition'))
        elif transform_type == 'map':
            return self._map_data(data, transform.get('mapping'))
        elif transform_type == 'enrich':
            return self._enrich_data(data, transform.get('enrichment'))
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    def _load_data(self, hook: 'CustomDataHook', data: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to target system."""
        load_config = self.processing_config.get('load', {})
        
        return hook.load_data(
            data=data,
            target_table=load_config.get('table'),
            load_strategy=load_config.get('strategy', 'insert'),
            batch_size=self.batch_size
        )
```

### Custom Hook Implementation
```python
from typing import Any, Dict, List, Optional, Union
import logging
from contextlib import contextmanager

from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowException

logger = logging.getLogger(__name__)

class CustomDataHook(BaseHook):
    """
    Custom hook for data operations with connection management.
    
    Provides methods for data extraction, transformation, and loading
    with proper connection pooling and error handling.
    """
    
    conn_name_attr = 'conn_id'
    default_conn_name = 'custom_data_default'
    conn_type = 'custom_data'
    hook_name = 'Custom Data Hook'
    
    def __init__(self, conn_id: str = default_conn_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn_id = conn_id
        self._connection = None
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for connection handling."""
        try:
            conn = self.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise AirflowException(f"Failed to establish connection: {e}") from e
        finally:
            self.close()
    
    def get_connection(self):
        """Get connection with proper error handling."""
        if self._connection is None:
            try:
                conn_config = self.get_connection(self.conn_id)
                self._connection = self._create_connection(conn_config)
                logger.info(f"Established connection to {self.conn_id}")
            except Exception as e:
                logger.error(f"Failed to connect to {self.conn_id}: {e}")
                raise AirflowException(f"Connection failed: {e}") from e
        
        return self._connection
    
    def _create_connection(self, conn_config):
        """Create actual connection based on configuration."""
        # Implement connection creation logic based on your data source
        pass
    
    def extract_data(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Extract data with batching support."""
        if not query:
            raise ValueError("Query is required for data extraction")
        
        logger.info(f"Extracting data with batch_size={batch_size}")
        
        with self.get_connection_context() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(query, params or {})
                
                # Fetch data in batches
                results = []
                while True:
                    batch = cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    
                    # Convert to dictionary format
                    columns = [desc[0] for desc in cursor.description]
                    batch_dicts = [dict(zip(columns, row)) for row in batch]
                    results.extend(batch_dicts)
                    
                    logger.info(f"Extracted batch of {len(batch)} records")
                
                logger.info(f"Total extracted: {len(results)} records")
                return results
                
            except Exception as e:
                logger.error(f"Data extraction failed: {e}")
                raise AirflowException(f"Extraction failed: {e}") from e
    
    def load_data(
        self,
        data: List[Dict[str, Any]],
        target_table: str,
        load_strategy: str = 'insert',
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Load data with different strategies."""
        if not data:
            logger.warning("No data to load")
            return {'loaded_count': 0}
        
        if not target_table:
            raise ValueError("Target table is required")
        
        logger.info(f"Loading {len(data)} records to {target_table} using {load_strategy} strategy")
        
        with self.get_connection_context() as conn:
            try:
                if load_strategy == 'insert':
                    return self._insert_data(conn, data, target_table, batch_size)
                elif load_strategy == 'upsert':
                    return self._upsert_data(conn, data, target_table, batch_size)
                elif load_strategy == 'replace':
                    return self._replace_data(conn, data, target_table, batch_size)
                else:
                    raise ValueError(f"Unknown load strategy: {load_strategy}")
                    
            except Exception as e:
                logger.error(f"Data loading failed: {e}")
                raise AirflowException(f"Loading failed: {e}") from e
    
    def close(self) -> None:
        """Close connection and cleanup resources."""
        if self._connection:
            try:
                self._connection.close()
                logger.info("Connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None
```

### Custom Sensor Implementation
```python
from typing import Any, Dict, Optional, Callable
from datetime import timedelta
import logging

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults, poke_mode_only
from airflow.exceptions import AirflowException

logger = logging.getLogger(__name__)

class CustomDataAvailabilitySensor(BaseSensorOperator):
    """
    Sensor to check for data availability in custom data sources.
    
    Waits for specified conditions to be met before allowing
    downstream tasks to proceed.
    """
    
    template_fields = ['data_source', 'condition_config']
    
    @apply_defaults
    def __init__(
        self,
        data_source: str,
        condition_config: Dict[str, Any],
        conn_id: str,
        check_interval: int = 60,
        timeout_hours: int = 24,
        soft_fail: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(
            poke_interval=check_interval,
            timeout=timeout_hours * 3600,
            soft_fail=soft_fail,
            *args,
            **kwargs
        )
        
        self.data_source = data_source
        self.condition_config = condition_config
        self.conn_id = conn_id
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate sensor configuration."""
        if not self.data_source:
            raise ValueError("data_source is required")
        
        if not isinstance(self.condition_config, dict):
            raise ValueError("condition_config must be a dictionary")
        
        required_keys = ['condition_type', 'parameters']
        for key in required_keys:
            if key not in self.condition_config:
                raise ValueError(f"condition_config must contain '{key}'")
    
    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if the condition is satisfied.
        
        :param context: Airflow context dictionary
        :return: True if condition is met, False otherwise
        """
        from .hooks.custom_data_hook import CustomDataHook
        
        logger.info(f"Checking condition for {self.data_source}")
        
        hook = CustomDataHook(conn_id=self.conn_id)
        
        try:
            condition_type = self.condition_config['condition_type']
            parameters = self.condition_config['parameters']
            
            if condition_type == 'file_exists':
                return self._check_file_exists(hook, parameters)
            elif condition_type == 'record_count':
                return self._check_record_count(hook, parameters)
            elif condition_type == 'data_freshness':
                return self._check_data_freshness(hook, parameters)
            elif condition_type == 'custom_query':
                return self._check_custom_condition(hook, parameters)
            else:
                raise ValueError(f"Unknown condition type: {condition_type}")
                
        except Exception as e:
            logger.error(f"Error checking condition: {e}")
            if not self.soft_fail:
                raise AirflowException(f"Condition check failed: {e}") from e
            return False
        finally:
            hook.close()
    
    def _check_file_exists(self, hook: 'CustomDataHook', params: Dict[str, Any]) -> bool:
        """Check if specified file exists."""
        file_path = params.get('file_path')
        if not file_path:
            raise ValueError("file_path is required for file_exists condition")
        
        exists = hook.check_file_exists(file_path)
        logger.info(f"File {file_path} exists: {exists}")
        return exists
    
    def _check_record_count(self, hook: 'CustomDataHook', params: Dict[str, Any]) -> bool:
        """Check if minimum record count is met."""
        table_name = params.get('table_name')
        min_count = params.get('min_count', 1)
        
        if not table_name:
            raise ValueError("table_name is required for record_count condition")
        
        actual_count = hook.get_record_count(table_name, params.get('filter_condition'))
        
        meets_condition = actual_count >= min_count
        logger.info(f"Record count check: {actual_count} >= {min_count} = {meets_condition}")
        return meets_condition
```

### Testing Patterns for Custom Operators
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from airflow.models import TaskInstance, DagRun
from airflow.utils.state import State

from operators.custom_data_processor import CustomDataProcessorOperator

class TestCustomDataProcessorOperator:
    """Test suite for CustomDataProcessorOperator."""
    
    @pytest.fixture
    def operator(self):
        """Create operator instance for testing."""
        return CustomDataProcessorOperator(
            task_id='test_task',
            source_conn_id='test_source',
            target_conn_id='test_target',
            processing_config={
                'extract': {'query': 'SELECT * FROM test_table'},
                'transform': {'steps': []},
                'load': {'table': 'target_table', 'strategy': 'insert'}
            },
            batch_size=100,
            timeout=300
        )
    
    @pytest.fixture
    def mock_context(self):
        """Provide mock Airflow context."""
        return {
            'ds': '2023-01-01',
            'ts': '2023-01-01T00:00:00+00:00',
            'execution_date': datetime(2023, 1, 1),
            'task_instance': Mock(),
            'dag_run': Mock()
        }
    
    def test_operator_initialization(self, operator):
        """Test operator initialization and validation."""
        assert operator.source_conn_id == 'test_source'
        assert operator.target_conn_id == 'test_target'
        assert operator.batch_size == 100
        assert operator.timeout == 300
    
    def test_operator_validation_errors(self):
        """Test operator validation with invalid parameters."""
        with pytest.raises(ValueError, match="source_conn_id is required"):
            CustomDataProcessorOperator(
                task_id='test_task',
                source_conn_id='',
                target_conn_id='test_target',
                processing_config={}
            )
    
    @patch('operators.custom_data_processor.CustomDataHook')
    def test_execute_success(self, mock_hook_class, operator, mock_context):
        """Test successful execution."""
        # Setup mocks
        mock_source_hook = Mock()
        mock_target_hook = Mock()
        mock_hook_class.side_effect = [mock_source_hook, mock_target_hook]
        
        # Mock data flow
        test_data = [{'id': 1, 'value': 'test'}]
        mock_source_hook.extract_data.return_value = test_data
        mock_target_hook.load_data.return_value = {'loaded_count': 1}
        
        # Execute
        result = operator.execute(mock_context)
        
        # Assert
        assert result['extracted_count'] == 1
        assert result['transformed_count'] == 1
        assert result['load_result']['loaded_count'] == 1
        
        # Verify hook cleanup
        mock_source_hook.close.assert_called_once()
        mock_target_hook.close.assert_called_once()
    
    @patch('operators.custom_data_processor.CustomDataHook')
    def test_execute_dry_run(self, mock_hook_class, mock_context):
        """Test dry run execution."""
        operator = CustomDataProcessorOperator(
            task_id='test_task',
            source_conn_id='test_source',
            target_conn_id='test_target',
            processing_config={'extract': {}, 'transform': {}, 'load': {}},
            dry_run=True
        )
        
        mock_source_hook = Mock()
        mock_target_hook = Mock()
        mock_hook_class.side_effect = [mock_source_hook, mock_target_hook]
        
        test_data = [{'id': 1}]
        mock_source_hook.extract_data.return_value = test_data
        
        result = operator.execute(mock_context)
        
        # Assert dry run behavior
        assert result['load_result']['dry_run'] is True
        assert result['load_result']['would_load'] == 1
        mock_target_hook.load_data.assert_not_called()
    
    def test_callback_execution(self, mock_context):
        """Test success and failure callback execution."""
        success_callback = Mock()
        failure_callback = Mock()
        
        operator = CustomDataProcessorOperator(
            task_id='test_task',
            source_conn_id='test_source',
            target_conn_id='test_target',
            processing_config={'extract': {}, 'transform': {}, 'load': {}},
            on_success_callback=success_callback,
            on_failure_callback=failure_callback
        )
        
        with patch.object(operator, '_execute_processing') as mock_execute:
            # Test success callback
            mock_execute.return_value = {'success': True}
            result = operator.execute(mock_context)
            
            success_callback.assert_called_once_with(mock_context, {'success': True})
            failure_callback.assert_not_called()
```

## Plugin Architecture
```python
from airflow.plugins_manager import AirflowPlugin
from operators.custom_data_processor import CustomDataProcessorOperator
from hooks.custom_data_hook import CustomDataHook
from sensors.custom_data_sensor import CustomDataAvailabilitySensor

class CustomDataPlugin(AirflowPlugin):
    """Plugin for custom data processing operators and hooks."""
    
    name = "custom_data_plugin"
    
    operators = [
        CustomDataProcessorOperator,
    ]
    
    hooks = [
        CustomDataHook,
    ]
    
    sensors = [
        CustomDataAvailabilitySensor,
    ]
    
    # Optional: Add custom views or menu links
    flask_blueprints = []
    appbuilder_views = []
    appbuilder_menu_items = []
```

## Performance Considerations
- Implement connection pooling for database operations
- Use batch processing for large datasets
- Implement proper timeout handling
- Monitor memory usage in long-running operations
- Use lazy loading for large configurations
- Implement caching for frequently accessed data
- Consider async operations for I/O bound tasks

## Security Checklist
- [ ] Use Airflow connections for sensitive credentials
- [ ] Validate all input parameters
- [ ] Implement proper access controls
- [ ] Avoid logging sensitive information
- [ ] Use parameterized queries to prevent injection
- [ ] Implement rate limiting for external API calls
- [ ] Follow principle of least privilege

## Documentation Requirements
- Provide comprehensive docstrings for all classes and methods
- Include parameter descriptions and type hints
- Provide usage examples in docstrings
- Document error conditions and exceptions
- Create README files for complex operator libraries
- Include configuration examples

## References
- Airflow Custom Operator Development Guide
- BaseOperator API Documentation
- Airflow Plugin Architecture
- Testing Airflow Components
---
description: Apache Airflow DAG development patterns and best practices
globs: 
  - "**/dags/**/*.py"
  - "**/airflow/**/*.py"
  - "**/*_dag.py"
  - "**/*dag*.py"
alwaysApply: false
priority: 300
tags: ["airflow", "orchestration", "pipeline", "etl"]
---

# Apache Airflow DAG Development Standards

## Overview
This rule defines best practices for developing Apache Airflow DAGs, including task design, dependency management, error handling, and monitoring patterns specific to workflow orchestration.

## CRITICAL REQUIREMENT: Custom Operator Library Usage

⚠️ **MANDATORY CONSTRAINT**: For multi-DAG automation development, you MUST use ONLY custom operators from the custom-opr-lib. Native Airflow operators are STRICTLY FORBIDDEN unless the user explicitly requests their use.

### Operator Usage Rules:
- ✅ **ALLOWED**: All operators from `custom_operators.*` module
- ✅ **ALLOWED**: All sensors from `custom_sensors.*` module  
- ✅ **ALLOWED**: All hooks from `custom_hooks.*` module
- ❌ **FORBIDDEN**: `airflow.operators.python.PythonOperator`
- ❌ **FORBIDDEN**: `airflow.operators.bash.BashOperator`
- ❌ **FORBIDDEN**: `airflow.operators.sql.*`
- ❌ **FORBIDDEN**: `airflow.sensors.filesystem.*`
- ❌ **FORBIDDEN**: Any native Airflow operator unless explicitly requested

### Exception Handling:
- If user explicitly says "use PythonOperator" or "use native operators", then native operators are allowed
- If user asks for "standard Airflow operators", then native operators are allowed
- Otherwise, ALWAYS use custom operators from custom-opr-lib

## Core Principles
- Design idempotent and atomic tasks
- Implement proper error handling and retry logic
- Use meaningful DAG and task IDs
- Follow Airflow's best practices for resource management
- Ensure DAGs are testable and maintainable
- Implement proper logging for debugging and monitoring
- **MANDATORY: Use ONLY custom operators from the custom-opr-lib - NO native Airflow operators unless explicitly specified by user**

## Guidelines

### MUST DO
- **CRITICAL: Use ONLY custom operators from custom-opr-lib for all DAG tasks**
- **FORBIDDEN: Do NOT use native Airflow operators (PythonOperator, BashOperator, etc.) unless user explicitly requests**
- Use descriptive DAG IDs and task IDs
- Set appropriate retry policies and timeouts
- Implement proper error handling with custom exceptions
- Use XComs judiciously (avoid large data transfers)
- Set catchup=False for most DAGs unless historical runs are needed
- Use templated fields for dynamic values
- Implement proper connection and variable management
- Add comprehensive documentation and tags

### SHOULD DO
- Use TaskGroups for logical task organization
- Implement data quality checks between tasks
- Use custom sensors from custom-opr-lib for external dependencies
- Set appropriate SLA monitoring
- Leverage the full custom operator library for all task implementations
- Implement proper testing with pytest
- Use Airflow Variables and Connections for configuration

### AVOID
- **ABSOLUTELY FORBIDDEN: Using native Airflow operators (PythonOperator, BashOperator, SqlOperator, etc.)**
- **STRICTLY PROHIBITED: Importing from airflow.operators.python, airflow.operators.bash, etc.**
- Hardcoding values in DAG code
- Creating too many small tasks (task overhead)
- Using top-level code that executes on every DAG parse
- Passing large amounts of data through XCom
- Creating circular dependencies
- Using bare except clauses in tasks

## Code Patterns

### Basic DAG Structure
```python
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.exceptions import AirflowException

# MANDATORY: Import ONLY from custom operator library
from custom_operators.data_processor import CustomDataProcessorOperator
from custom_operators.file_processor import CustomFileProcessorOperator
from custom_operators.validation import DataValidationOperator
from custom_sensors.data_availability import CustomDataAvailabilitySensor

# DAG configuration
DEFAULT_ARGS = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# DAG definition
dag = DAG(
    dag_id='data_processing_pipeline',
    default_args=DEFAULT_ARGS,
    description='Daily data processing and transformation pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['etl', 'daily', 'production'],
    doc_md=__doc__,
    max_active_tasks=10,
    dagrun_timeout=timedelta(hours=2),
)
```

### Task Implementation Patterns - Using Custom Operators ONLY
```python
# CORRECT: Using custom operators from custom-opr-lib
extract_task = CustomDataProcessorOperator(
    task_id='extract_data',
    source_conn_id='source_db',
    target_conn_id='staging_db',
    processing_config={
        'extract': {
            'query': """
                SELECT * FROM transactions 
                WHERE created_date = '{{ ds }}'
            """,
            'params': {'batch_size': 1000}
        },
        'transform': {'steps': []},  # No transformation for extract
        'load': {
            'table': 'staging_transactions',
            'strategy': 'replace'
        }
    },
    batch_size=1000,
    pool='database_pool',
    dag=dag,
)

transform_task = CustomDataProcessorOperator(
    task_id='transform_data',
    source_conn_id='staging_db',
    target_conn_id='warehouse_db',
    processing_config={
        'extract': {
            'query': "SELECT * FROM staging_transactions WHERE process_date = '{{ ds }}'"
        },
        'transform': {
            'steps': [
                {'type': 'filter', 'condition': 'amount > 0'},
                {'type': 'enrich', 'enrichment': 'add_processing_timestamp'},
                {'type': 'map', 'mapping': 'standardize_currency'}
            ]
        },
        'load': {
            'table': 'processed_transactions',
            'strategy': 'upsert'
        }
    },
    batch_size=500,
    pool='processing_pool',
    dag=dag,
)

# Data validation using custom operator
validation_task = DataValidationOperator(
    task_id='validate_processed_data',
    table_name='processed_transactions',
    validation_sql="""
        SELECT COUNT(*) 
        FROM processed_transactions 
        WHERE process_date = '{{ ds }}'
    """,
    conn_id='warehouse_db',
    min_rows=100,
    dag=dag,
)

# File processing using custom operator
file_processor_task = CustomFileProcessorOperator(
    task_id='process_input_files',
    source_path='/data/input/{{ ds }}/',
    target_path='/data/processed/{{ ds }}/',
    file_pattern='*.csv',
    processing_config={
        'validation': {'check_schema': True, 'check_completeness': True},
        'transformation': {'normalize_headers': True, 'clean_data': True},
        'output_format': 'parquet'
    },
    dag=dag,
)
```

### TaskGroup Pattern - Using Custom Operators
```python
def create_data_quality_checks() -> TaskGroup:
    """Create a group of data quality validation tasks using custom operators."""
    
    with TaskGroup(group_id='data_quality_checks', dag=dag) as group:
        
        # Row count validation using custom operator
        row_count_check = DataValidationOperator(
            task_id='check_row_count',
            table_name='processed_transactions',
            validation_sql="""
                SELECT COUNT(*) 
                FROM processed_transactions 
                WHERE process_date = '{{ ds }}'
            """,
            conn_id='warehouse_db',
            min_rows=Variable.get("min_daily_rows", default_var=1000),
        )
        
        # Data freshness check using custom operator
        freshness_check = DataValidationOperator(
            task_id='check_data_freshness',
            table_name='processed_transactions',
            validation_sql="""
                SELECT COUNT(*) 
                FROM processed_transactions 
                WHERE process_date = '{{ ds }}'
                AND created_timestamp >= NOW() - INTERVAL '{{ var.value.max_data_age_hours }}' HOUR
            """,
            conn_id='warehouse_db',
            min_rows=1,
        )
        
        # Data completeness check using custom operator
        completeness_check = DataValidationOperator(
            task_id='check_data_completeness',
            table_name='processed_transactions',
            validation_sql="""
                SELECT COUNT(*) 
                FROM processed_transactions 
                WHERE process_date = '{{ ds }}'
                AND amount IS NOT NULL 
                AND transaction_id IS NOT NULL
            """,
            conn_id='warehouse_db',
            min_rows=Variable.get("min_complete_records", default_var=950),
        )
        
        # Parallel execution of quality checks
        [row_count_check, freshness_check, completeness_check]
    
    return group

# Use the TaskGroup
quality_checks = create_data_quality_checks()
```

### Sensor Pattern - Using Custom Sensors
```python
# CORRECT: Using custom sensors from custom-opr-lib

# Wait for external file using custom sensor
file_sensor = CustomDataAvailabilitySensor(
    task_id='wait_for_input_file',
    data_source='input_files',
    condition_config={
        'condition_type': 'file_exists',
        'parameters': {
            'file_path': '/data/input/{{ ds }}/data.csv'
        }
    },
    conn_id='filesystem_default',
    check_interval=60,     # Check every minute
    timeout_hours=0.5,     # Timeout after 30 minutes
    dag=dag,
)

# Wait for database condition using custom sensor
db_sensor = CustomDataAvailabilitySensor(
    task_id='wait_for_upstream_table',
    data_source='upstream_database',
    condition_config={
        'condition_type': 'record_count',
        'parameters': {
            'table_name': 'upstream_table',
            'min_count': 1,
            'filter_condition': "process_date = '{{ ds }}' AND status = 'completed'"
        }
    },
    conn_id='postgres_default',
    check_interval=300,    # Check every 5 minutes
    timeout_hours=1,       # 1 hour timeout
    dag=dag,
)

# Wait for data freshness using custom sensor
freshness_sensor = CustomDataAvailabilitySensor(
    task_id='wait_for_fresh_data',
    data_source='source_system',
    condition_config={
        'condition_type': 'data_freshness',
        'parameters': {
            'table_name': 'source_data',
            'timestamp_column': 'updated_at',
            'max_age_hours': 2
        }
    },
    conn_id='source_db',
    check_interval=120,    # Check every 2 minutes
    timeout_hours=6,       # 6 hour timeout
    dag=dag,
)
```

### Multi-DAG Automation Pattern
```python
# MANDATORY: Complete DAG implementation using ONLY custom operators
from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Import custom operators and sensors from custom-opr-lib
from custom_operators.data_processor import CustomDataProcessorOperator
from custom_operators.file_processor import CustomFileProcessorOperator
from custom_operators.validation import DataValidationOperator
from custom_operators.notification import CustomNotificationOperator
from custom_sensors.data_availability import CustomDataAvailabilitySensor

# DAG configuration for multi-pipeline automation
DEFAULT_ARGS = {
    'owner': 'automation-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

dag = DAG(
    dag_id='multi_pipeline_automation',
    default_args=DEFAULT_ARGS,
    description='Automated multi-pipeline data processing with custom operators',
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    catchup=False,
    tags=['automation', 'multi-pipeline', 'production'],
    max_active_tasks=15,
    dagrun_timeout=timedelta(hours=4),
)

# Data availability sensors
input_sensor = CustomDataAvailabilitySensor(
    task_id='check_input_data_availability',
    data_source='source_system',
    condition_config={
        'condition_type': 'record_count',
        'parameters': {
            'table_name': 'source_transactions',
            'min_count': 100,
            'filter_condition': "created_date = '{{ ds }}'"
        }
    },
    conn_id='source_db',
    dag=dag,
)

# Multi-stage processing using custom operators
extract_transform_load = CustomDataProcessorOperator(
    task_id='etl_main_pipeline',
    source_conn_id='source_db',
    target_conn_id='warehouse_db',
    processing_config={
        'extract': {
            'query': """
                SELECT transaction_id, amount, currency, customer_id, created_date
                FROM source_transactions 
                WHERE created_date = '{{ ds }}'
            """,
            'params': {'batch_size': 2000}
        },
        'transform': {
            'steps': [
                {'type': 'filter', 'condition': 'amount > 0'},
                {'type': 'enrich', 'enrichment': 'currency_conversion'},
                {'type': 'map', 'mapping': 'standardize_format'}
            ]
        },
        'load': {
            'table': 'fact_transactions',
            'strategy': 'upsert',
            'conflict_columns': ['transaction_id']
        }
    },
    batch_size=1000,
    timeout=7200,  # 2 hours
    dag=dag,
)

# Parallel data quality validations
with TaskGroup(group_id='data_quality_validation', dag=dag) as dq_group:
    volume_check = DataValidationOperator(
        task_id='validate_data_volume',
        table_name='fact_transactions',
        validation_sql="""
            SELECT COUNT(*) FROM fact_transactions 
            WHERE process_date = '{{ ds }}'
        """,
        conn_id='warehouse_db',
        min_rows=Variable.get("min_transaction_count", default_var=50),
    )
    
    completeness_check = DataValidationOperator(
        task_id='validate_data_completeness',
        table_name='fact_transactions',
        validation_sql="""
            SELECT COUNT(*) FROM fact_transactions 
            WHERE process_date = '{{ ds }}' 
            AND amount IS NOT NULL AND customer_id IS NOT NULL
        """,
        conn_id='warehouse_db',
        min_rows=Variable.get("min_complete_records", default_var=45),
    )

# Notification on completion
success_notification = CustomNotificationOperator(
    task_id='send_success_notification',
    notification_type='email',
    recipients=Variable.get("success_recipients", deserialize_json=True),
    message_template='Pipeline {{ dag.dag_id }} completed successfully for {{ ds }}',
    dag=dag,
)

# Define dependencies
input_sensor >> extract_transform_load >> dq_group >> success_notification
```

## Error Handling Patterns
```python
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.utils.email import send_email

def task_with_comprehensive_error_handling(**context) -> Any:
    """Example of comprehensive error handling in Airflow tasks."""
    import logging
    
    logger = logging.getLogger(__name__)
    task_instance = context['task_instance']
    
    try:
        # Main task logic
        result = perform_data_operation()
        
        # Validate result
        if not result:
            logger.warning("No data to process, skipping downstream tasks")
            raise AirflowSkipException("No data available for processing")
        
        return result
        
    except DataQualityError as e:
        # Handle known data quality issues
        logger.error(f"Data quality check failed: {e}")
        
        # Send custom notification
        send_custom_alert(
            subject=f"Data Quality Alert - {task_instance.dag_id}",
            message=f"Task {task_instance.task_id} failed data quality checks: {e}"
        )
        
        raise AirflowException(f"Data quality check failed: {e}") from e
        
    except ExternalServiceError as e:
        # Handle external service failures
        logger.error(f"External service unavailable: {e}")
        
        # Check if this is a retryable error
        if e.is_retryable and task_instance.try_number < task_instance.max_tries:
            logger.info(f"Retrying task, attempt {task_instance.try_number}")
            raise  # Let Airflow handle the retry
        else:
            raise AirflowException(f"External service failed after retries: {e}") from e
            
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error in {task_instance.task_id}: {e}")
        
        # Send detailed error report
        send_error_report(task_instance, e)
        
        raise AirflowException(f"Unexpected error: {e}") from e

def send_custom_alert(subject: str, message: str) -> None:
    """Send custom alert notification."""
    recipients = Variable.get("alert_recipients", deserialize_json=True)
    
    send_email(
        to=recipients,
        subject=subject,
        html_content=f"<p>{message}</p>",
    )
```

## Testing Patterns
```python
import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from airflow.models import DagBag, TaskInstance
from airflow.utils.state import State

class TestDataProcessingDAG:
    """Test suite for data processing DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG for testing."""
        return DagBag(dag_folder='dags/', include_examples=False)
    
    def test_dag_loaded(self, dagbag):
        """Test that DAG loads without errors."""
        dag = dagbag.get_dag(dag_id='data_processing_pipeline')
        assert dag is not None
        assert len(dag.tasks) > 0
    
    def test_dag_structure(self, dagbag):
        """Test DAG structure and dependencies."""
        dag = dagbag.get_dag(dag_id='data_processing_pipeline')
        
        # Check expected tasks exist
        expected_tasks = ['extract_data', 'transform_data', 'load_data']
        actual_tasks = [task.task_id for task in dag.tasks]
        
        for task_id in expected_tasks:
            assert task_id in actual_tasks
        
        # Check dependencies
        extract_task = dag.get_task('extract_data')
        transform_task = dag.get_task('transform_data')
        
        assert transform_task in extract_task.downstream_list
    
    def test_extract_data_task(self):
        """Test extract data task logic."""
        from dags.data_processing_dag import extract_data
        
        # Mock context
        context = {
            'execution_date': datetime(2023, 1, 1),
            'ds': '2023-01-01'
        }
        
        with patch('src.utils.database.get_connection') as mock_conn:
            mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {'id': 1, 'amount': 100},
                {'id': 2, 'amount': 200}
            ]
            
            result = extract_data(**context)
            
            assert result['record_count'] == 2
            assert 'extraction_timestamp' in result
    
    @pytest.mark.integration
    def test_dag_run_integration(self, dagbag):
        """Integration test for complete DAG run."""
        dag = dagbag.get_dag(dag_id='data_processing_pipeline')
        execution_date = datetime(2023, 1, 1)
        
        # Create DAG run
        dag_run = dag.create_dagrun(
            run_id=f'test_run_{execution_date}',
            execution_date=execution_date,
            state=State.RUNNING
        )
        
        # Test individual task execution
        extract_task = dag.get_task('extract_data')
        ti = TaskInstance(extract_task, execution_date)
        
        # Mock external dependencies
        with patch('src.utils.database.get_connection'):
            ti.run(ignore_dependencies=True)
            assert ti.state == State.SUCCESS
```

## Monitoring and Alerting
```python
from airflow.utils.email import send_email
from airflow.models import Variable

def setup_sla_monitoring():
    """Configure SLA monitoring for critical tasks."""
    
    def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
        """Handle SLA misses."""
        subject = f"SLA Miss Alert - {dag.dag_id}"
        
        missed_tasks = [task.task_id for task in task_list]
        blocking_tasks = [task.task_id for task in blocking_task_list]
        
        message = f"""
        SLA missed for the following tasks: {missed_tasks}
        Blocking tasks: {blocking_tasks}
        
        Please investigate immediately.
        """
        
        alert_recipients = Variable.get("sla_alert_recipients", deserialize_json=True)
        
        send_email(
            to=alert_recipients,
            subject=subject,
            html_content=message
        )
    
    return sla_miss_callback

# Apply to DAG
dag.sla_miss_callback = setup_sla_monitoring()
```

## Configuration Management
```python
# Use Airflow Variables for configuration
def get_dag_config() -> Dict[str, Any]:
    """Get DAG configuration from Airflow Variables."""
    return {
        'source_conn_id': Variable.get("source_connection_id", default_var="postgres_default"),
        'target_conn_id': Variable.get("target_connection_id", default_var="postgres_warehouse"),
        'batch_size': int(Variable.get("processing_batch_size", default_var="1000")),
        'notification_emails': Variable.get("notification_emails", deserialize_json=True),
        'retry_count': int(Variable.get("default_retry_count", default_var="3")),
        'timeout_minutes': int(Variable.get("task_timeout_minutes", default_var="60"))
    }

# Use in DAG definition
config = get_dag_config()

DEFAULT_ARGS.update({
    'retries': config['retry_count'],
    'execution_timeout': timedelta(minutes=config['timeout_minutes']),
    'email': config['notification_emails']
})
```

## Performance Considerations
- Use connection pooling for database operations
- Implement proper resource management with pools
- Avoid storing large data in XCom
- Use appropriate parallelism settings
- Monitor DAG parsing time
- Use efficient SQL queries with proper indexing
- Implement data partitioning for large datasets

## Security Checklist
- [ ] Use Airflow Connections for sensitive credentials
- [ ] Implement proper access controls
- [ ] Avoid logging sensitive information
- [ ] Use encrypted connections for data transfer
- [ ] Regularly rotate credentials
- [ ] Validate input parameters
- [ ] Use least privilege principle for service accounts

## References
- Apache Airflow Best Practices Documentation
- Airflow DAG Writing Best Practices
- TaskGroup Documentation
- Custom Operator Development Guide
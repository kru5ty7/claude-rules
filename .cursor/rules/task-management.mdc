---
description: Comprehensive task management methodology and todo processing standards
globs: 
  - "**/*.md"
  - "**/*.py"
  - "**/*.js"
  - "**/*.ts"
alwaysApply: false
priority: 800
tags: ["task-management", "productivity", "todos", "planning"]
---

# Task Management and Todo Processing Standards

## Overview
This rule defines comprehensive methodology for task management, todo processing, and systematic project planning. It ensures consistent approach to breaking down complex work into manageable, trackable components.

## Core Task Management Principles
- Tasks must be clear, specific, and actionable—avoid ambiguity
- Every task must be assigned explicit ownership and status tracking
- Complex tasks must be broken into atomic, trackable subtasks
- Dependencies between tasks must be explicitly declared
- Security-related tasks must undergo mandatory review

## Todo Processing Methodology

### Phase 1: Initial Analysis - Think Deeply
Apply systematic analysis to understand the overall todo list architecture and dependencies.

#### Comprehensive Review Process
```python
class TodoAnalyzer:
    """Systematic todo list analysis and processing."""
    
    def analyze_todo_list(self, todos: List[str]) -> AnalysisResult:
        """
        Perform comprehensive analysis of todo items.
        
        Returns analysis covering:
        - Clarity and scope assessment
        - Technical feasibility evaluation  
        - Dependency identification
        - Conflict and redundancy detection
        - Resource requirement analysis
        """
        analysis = AnalysisResult()
        
        for todo in todos:
            item_analysis = self._analyze_individual_item(todo)
            analysis.add_item(item_analysis)
        
        # Cross-item analysis
        analysis.dependencies = self._identify_dependencies(analysis.items)
        analysis.conflicts = self._detect_conflicts(analysis.items)
        analysis.opportunities = self._find_reuse_opportunities(analysis.items)
        
        return analysis
    
    def _analyze_individual_item(self, todo: str) -> TodoItemAnalysis:
        """Analyze individual todo item for clarity, scope, and feasibility."""
        return TodoItemAnalysis(
            original_text=todo,
            clarity_score=self._assess_clarity(todo),
            scope_complexity=self._evaluate_scope(todo),
            technical_feasibility=self._check_feasibility(todo),
            required_resources=self._identify_resources(todo),
            estimated_effort=self._estimate_effort(todo)
        )
```

#### Design Validation Checklist
- ✅ Verify alignment with established patterns and best practices
- ✅ Check for architectural consistency across items
- ✅ Identify opportunities for code/logic reuse
- ✅ Validate against existing system constraints
- ✅ Assess impact on current system architecture

### Phase 2: Item Processing - Clarity and Scope Assessment

#### Decision Framework
```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class TaskStatus(Enum):
    CLEAR_AND_SCOPED = "clear_and_scoped"
    NEEDS_BREAKDOWN = "needs_breakdown" 
    REQUIRES_CLARIFICATION = "requires_clarification"
    BLOCKED = "blocked"

@dataclass
class TaskAssessment:
    """Assessment result for individual task."""
    task_id: str
    original_description: str
    clarity_score: int  # 1-10, 10 being crystal clear
    scope_size: str  # "small", "medium", "large", "xl"
    status: TaskStatus
    breakdown_required: bool
    blocking_factors: List[str]
    clarification_needed: List[str]
    
def assess_task_clarity(task_description: str) -> TaskAssessment:
    """
    Assess task clarity and determine processing approach.
    
    Clarity Criteria:
    - Clear success criteria defined
    - Actionable steps identifiable  
    - Resource requirements understood
    - Dependencies explicitly stated
    - Acceptance criteria measurable
    """
    clarity_indicators = [
        "specific_outcome_defined",
        "actionable_verbs_used", 
        "measurable_criteria",
        "resource_requirements_clear",
        "timeline_reasonable"
    ]
    
    clarity_score = sum(1 for indicator in clarity_indicators 
                       if _check_indicator(task_description, indicator))
    
    if clarity_score >= 8:
        status = TaskStatus.CLEAR_AND_SCOPED
    elif clarity_score >= 5:
        status = TaskStatus.NEEDS_BREAKDOWN  
    else:
        status = TaskStatus.REQUIRES_CLARIFICATION
    
    return TaskAssessment(
        task_id=generate_task_id(),
        original_description=task_description,
        clarity_score=clarity_score,
        scope_size=_estimate_scope_size(task_description),
        status=status,
        breakdown_required=clarity_score < 8,
        blocking_factors=_identify_blockers(task_description),
        clarification_needed=_identify_clarifications(task_description)
    )
```

### Phase 3: Strategic Planning - Technical Architecture

#### Implementation Strategy Framework
```python
@dataclass
class ImplementationPlan:
    """Comprehensive implementation strategy."""
    task_id: str
    technical_approach: str
    key_components: List[str]
    dependencies: List[str]
    error_handling_strategy: str
    testing_approach: str
    validation_checkpoints: List[str]
    rollback_plan: str
    estimated_duration: int  # in hours
    
def create_implementation_plan(task: TaskAssessment) -> ImplementationPlan:
    """
    Create detailed implementation strategy.
    
    Planning Components:
    1. Technical Architecture Analysis
    2. Component Breakdown
    3. Dependency Mapping
    4. Error Handling Design
    5. Testing Strategy
    6. Validation Checkpoints
    """
    # Analyze technical requirements
    tech_analysis = analyze_technical_requirements(task.original_description)
    
    # Plan implementation steps
    implementation_steps = break_down_implementation(tech_analysis)
    
    # Design error handling
    error_strategy = design_error_handling(tech_analysis.failure_modes)
    
    # Plan testing approach
    test_strategy = plan_testing_approach(implementation_steps)
    
    return ImplementationPlan(
        task_id=task.task_id,
        technical_approach=tech_analysis.recommended_approach,
        key_components=implementation_steps.components,
        dependencies=tech_analysis.dependencies,
        error_handling_strategy=error_strategy,
        testing_approach=test_strategy,
        validation_checkpoints=implementation_steps.checkpoints,
        rollback_plan=design_rollback_strategy(implementation_steps),
        estimated_duration=estimate_implementation_time(implementation_steps)
    )
```

#### Architecture Decision Framework
```python
class ArchitectureDecisionRecord:
    """Document key architecture decisions for complex tasks."""
    
    def __init__(self, task_id: str, decision_context: str):
        self.task_id = task_id
        self.context = decision_context
        self.options_considered = []
        self.decision = None
        self.rationale = None
        self.consequences = []
    
    def add_option(self, option: str, pros: List[str], cons: List[str]) -> None:
        """Add architecture option with trade-off analysis."""
        self.options_considered.append({
            'option': option,
            'pros': pros,
            'cons': cons,
            'complexity_score': self._assess_complexity(option),
            'maintainability_score': self._assess_maintainability(option)
        })
    
    def make_decision(self, chosen_option: str, rationale: str) -> None:
        """Record architecture decision with justification."""
        self.decision = chosen_option
        self.rationale = rationale
        self.consequences = self._analyze_consequences(chosen_option)
```

### Phase 4: Implementation - Production Quality Execution

#### Implementation Standards
```python
class ProductionImplementation:
    """Production-quality implementation with comprehensive monitoring."""
    
    def __init__(self, plan: ImplementationPlan):
        self.plan = plan
        self.progress_tracker = ProgressTracker()
        self.logger = setup_implementation_logging()
    
    def execute_with_monitoring(self) -> ImplementationResult:
        """Execute implementation with real-time monitoring."""
        try:
            self.logger.info(f"Starting implementation of {self.plan.task_id}")
            
            # Execute each component with checkpoint validation
            for component in self.plan.key_components:
                self._implement_component(component)
                self._validate_checkpoint(component)
                self.progress_tracker.mark_component_complete(component)
            
            # Final validation
            final_result = self._perform_final_validation()
            
            self.logger.info(f"Implementation completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Implementation failed: {e}")
            self._execute_rollback()
            raise ImplementationError(f"Task {self.plan.task_id} failed: {e}") from e
    
    def _implement_component(self, component: str) -> None:
        """Implement individual component with error handling."""
        try:
            component_impl = ComponentImplementer(component, self.plan)
            result = component_impl.execute()
            
            # Validate component implementation
            if not self._validate_component(component, result):
                raise ComponentValidationError(f"Component {component} validation failed")
                
        except Exception as e:
            self.logger.error(f"Component {component} implementation failed: {e}")
            raise
    
    def _validate_checkpoint(self, component: str) -> None:
        """Validate implementation checkpoint."""
        checkpoint = self.plan.validation_checkpoints.get(component)
        if checkpoint:
            if not self._run_checkpoint_tests(checkpoint):
                raise CheckpointValidationError(f"Checkpoint validation failed for {component}")
```

### Phase 5: Quality Assurance - Rigorous Verification

#### Multi-Level Validation Framework
```python
class QualityAssuranceFramework:
    """Comprehensive quality assurance for task completion."""
    
    def __init__(self, implementation_result: ImplementationResult):
        self.result = implementation_result
        self.validation_results = ValidationResults()
    
    def perform_comprehensive_qa(self) -> QAReport:
        """Execute multi-level quality assurance."""
        qa_report = QAReport()
        
        # Level 1: Functionality Testing
        functionality_result = self._test_functionality()
        qa_report.add_result("functionality", functionality_result)
        
        # Level 2: Code Quality Review
        quality_result = self._review_code_quality()
        qa_report.add_result("code_quality", quality_result)
        
        # Level 3: Design Validation
        design_result = self._validate_design_adherence()
        qa_report.add_result("design_validation", design_result)
        
        # Level 4: Security Assessment
        security_result = self._assess_security()
        qa_report.add_result("security", security_result)
        
        # Level 5: Performance Validation
        performance_result = self._validate_performance()
        qa_report.add_result("performance", performance_result)
        
        return qa_report
    
    def _test_functionality(self) -> TestResult:
        """Test all use cases and edge cases."""
        test_suite = FunctionalTestSuite(self.result)
        
        results = []
        # Test normal use cases
        results.extend(test_suite.run_normal_cases())
        
        # Test edge cases
        results.extend(test_suite.run_edge_cases())
        
        # Test error conditions
        results.extend(test_suite.run_error_cases())
        
        return TestResult(
            passed=all(r.passed for r in results),
            details=results,
            coverage_percentage=test_suite.calculate_coverage()
        )
```

## Task Status Management

### Status Tracking System
```python
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class TaskTracker:
    """Comprehensive task tracking with status management."""
    task_id: str
    description: str
    priority: TaskPriority
    state: TaskState
    assigned_to: str
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime]
    dependencies: List[str]
    progress_percentage: int
    notes: List[str]
    
    def update_status(self, new_state: TaskState, note: str = "") -> None:
        """Update task status with timestamp and notes."""
        self.state = new_state
        self.updated_at = datetime.now()
        if note:
            self.notes.append(f"{datetime.now().isoformat()}: {note}")
    
    def add_dependency(self, dependency_task_id: str) -> None:
        """Add task dependency."""
        if dependency_task_id not in self.dependencies:
            self.dependencies.append(dependency_task_id)
    
    def calculate_progress(self, completed_subtasks: int, total_subtasks: int) -> None:
        """Calculate and update progress percentage."""
        if total_subtasks > 0:
            self.progress_percentage = int((completed_subtasks / total_subtasks) * 100)
```

## Error Handling for Task Management

### Escalation Framework
```python
class TaskEscalationManager:
    """Handle task escalation when blocked or failed."""
    
    def __init__(self):
        self.escalation_rules = {
            "blocking_dependency": self._handle_dependency_block,
            "resource_unavailable": self._handle_resource_block, 
            "technical_blocker": self._handle_technical_block,
            "scope_unclear": self._handle_scope_clarification
        }
    
    def handle_task_blocker(self, task: TaskTracker, blocker_type: str, details: str) -> EscalationResult:
        """Handle task blocker with appropriate escalation."""
        try:
            # First attempt: Self-resolution
            resolution = self._attempt_self_resolution(task, blocker_type, details)
            if resolution.resolved:
                return resolution
            
            # Second attempt: Alternative approach
            alternative = self._try_alternative_approach(task, blocker_type, details)
            if alternative.resolved:
                return alternative
            
            # Third attempt: Break down further
            breakdown = self._attempt_further_breakdown(task, blocker_type, details)
            if breakdown.resolved:
                return breakdown
            
            # Escalation required
            return self._escalate_to_reviewer(task, blocker_type, details)
            
        except Exception as e:
            return EscalationResult(
                resolved=False,
                escalated=True,
                escalation_reason=f"Exception during resolution: {e}"
            )
```

## Todo List Output Format

### Standardized Reporting Template
```python
class TodoReportGenerator:
    """Generate standardized todo processing reports."""
    
    def generate_comprehensive_report(self, analysis: AnalysisResult) -> str:
        """Generate comprehensive todo analysis report."""
        report_sections = []
        
        for item in analysis.items:
            section = self._generate_item_section(item)
            report_sections.append(section)
        
        return "\n".join(report_sections)
    
    def _generate_item_section(self, item: TodoItemAnalysis) -> str:
        """Generate individual item report section."""
        return f"""
### 📋 Todo Item: {item.title}
- **Clarity Score**: {self._format_clarity_score(item.clarity_score)}
- **Scope Assessment**: {item.scope_complexity}
- **Dependencies**: {', '.join(item.dependencies) if item.dependencies else 'None'}
- **Reuse Opportunities**: {', '.join(item.reuse_opportunities) if item.reuse_opportunities else 'None'}

### 🔧 Planning/Breakdown
{self._format_implementation_plan(item.implementation_plan)}

### ✅ Implementation
{self._format_implementation_details(item.implementation)}

### 🔍 Verification Results  
- **Functionality**: {self._format_test_result(item.functionality_tests)}
- **Quality**: {self._format_quality_result(item.quality_review)}
- **Integration**: {self._format_integration_result(item.integration_tests)}

### 📝 Summary
{item.summary}

---
"""
```

## References
- Project Management Best Practices
- Agile Task Breakdown Techniques  
- Quality Assurance Methodologies
- Systematic Problem Solving Approaches
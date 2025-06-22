---
description: Tool usage optimization and efficient workflow patterns for development tasks
globs: 
  - "**/*.py"
  - "**/*.js"
  - "**/*.ts"
  - "**/*.md"
alwaysApply: true
priority: 850
tags: ["tools", "efficiency", "workflow", "optimization"]
---

# Tool Usage Optimization and Workflow Patterns

## Overview
This rule defines optimization strategies for tool usage, efficient workflow patterns, and best practices for maximizing productivity in development tasks while maintaining code quality and security standards.

## Core Tool Usage Principles
- Prefer parallel tool execution when no dependencies exist between calls
- Use the Agent tool for extensive search operations to reduce context usage
- Batch independent tool calls in single function blocks for optimal performance
- Follow conventional workflows: search → understand → implement → verify
- Maintain conciseness in responses (under 4 lines unless detail requested)

## Parallel Tool Execution Strategies

### Batch Processing Patterns
```python
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

class ToolExecutionOptimizer:
    """Optimize tool execution through parallelization and batching."""
    
    def __init__(self, max_concurrent_tools: int = 5):
        self.max_concurrent_tools = max_concurrent_tools
        self.execution_history = []
    
    def execute_parallel_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute independent tool calls in parallel for maximum efficiency.
        
        Optimization Strategy:
        1. Identify independent tool calls (no data dependencies)
        2. Group dependent calls into sequential chains
        3. Execute independent groups in parallel
        4. Merge results maintaining execution order
        """
        # Analyze dependencies
        dependency_graph = self._build_dependency_graph(tool_calls)
        execution_groups = self._create_execution_groups(dependency_graph)
        
        results = []
        for group in execution_groups:
            if len(group) == 1:
                # Single tool execution
                result = self._execute_single_tool(group[0])
                results.append(result)
            else:
                # Parallel execution for independent tools
                parallel_results = self._execute_parallel_group(group)
                results.extend(parallel_results)
        
        return results
    
    def _execute_parallel_group(self, tool_group: List[ToolCall]) -> List[ToolResult]:
        """Execute group of independent tools in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tools) as executor:
            # Submit all tools for execution
            future_to_tool = {
                executor.submit(self._execute_single_tool, tool): tool 
                for tool in tool_group
            }
            
            results = []
            for future in as_completed(future_to_tool):
                tool = future_to_tool[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per tool
                    results.append(result)
                except Exception as e:
                    error_result = ToolResult(
                        tool=tool,
                        success=False,
                        error=f"Tool execution failed: {e}",
                        execution_time=0
                    )
                    results.append(error_result)
            
            return results
```

### Efficient Search Patterns
```python
class SearchOptimizer:
    """Optimize search operations across multiple tools."""
    
    def perform_comprehensive_search(
        self, 
        search_query: str, 
        search_scope: SearchScope
    ) -> SearchResults:
        """
        Perform comprehensive search using optimal tool selection.
        
        Tool Selection Strategy:
        - Use Agent tool for extensive/iterative searches
        - Use Glob for file pattern matching
        - Use Grep for content searches within known directories
        - Use Read for specific file content
        """
        if search_scope.is_extensive_search():
            # Use Agent tool for complex searches
            return self._agent_search(search_query, search_scope)
        
        # Use direct tools for targeted searches
        search_tools = self._select_optimal_tools(search_query, search_scope)
        return self._execute_parallel_search(search_tools)
    
    def _execute_parallel_search(self, search_tools: List[SearchTool]) -> SearchResults:
        """Execute multiple search tools in parallel."""
        results = SearchResults()
        
        # Group tools by type for batch execution
        glob_searches = [t for t in search_tools if t.type == 'glob']
        grep_searches = [t for t in search_tools if t.type == 'grep']
        read_operations = [t for t in search_tools if t.type == 'read']
        
        # Execute each tool type in parallel
        if glob_searches:
            glob_results = self._parallel_glob_search(glob_searches)
            results.add_glob_results(glob_results)
        
        if grep_searches:
            grep_results = self._parallel_grep_search(grep_searches)
            results.add_grep_results(grep_results)
        
        if read_operations:
            read_results = self._parallel_read_operations(read_operations)
            results.add_read_results(read_results)
        
        return results
```

## Context Usage Optimization

### Agent Tool Usage Strategy
```python
class ContextOptimizer:
    """Optimize context usage through strategic tool selection."""
    
    def should_use_agent_tool(self, search_request: SearchRequest) -> bool:
        """
        Determine when to use Agent tool vs direct tools.
        
        Use Agent Tool When:
        - Search requires multiple rounds of exploration
        - Search scope is unknown or very broad
        - Need to correlate findings across multiple files
        - Search involves complex pattern matching
        
        Use Direct Tools When:
        - Specific file paths known
        - Searching within 2-3 known files
        - Simple pattern matching
        - Reading specific content
        """
        complexity_score = self._calculate_search_complexity(search_request)
        scope_score = self._assess_search_scope(search_request)
        context_cost = self._estimate_context_cost(search_request)
        
        # Use Agent tool if complexity or scope is high
        return (complexity_score > 7 or scope_score > 8 or 
                context_cost > self.context_threshold)
    
    def optimize_tool_selection(self, task: DevelopmentTask) -> ToolExecutionPlan:
        """
        Create optimized tool execution plan for development tasks.
        
        Optimization Priorities:
        1. Minimize context usage
        2. Maximize parallel execution
        3. Reduce total execution time
        4. Maintain result quality
        """
        plan = ToolExecutionPlan()
        
        # Phase 1: Understanding (parallel search operations)
        understanding_tools = self._select_understanding_tools(task)
        plan.add_parallel_phase("understanding", understanding_tools)
        
        # Phase 2: Analysis (dependent on understanding results)
        analysis_tools = self._select_analysis_tools(task)
        plan.add_sequential_phase("analysis", analysis_tools)
        
        # Phase 3: Implementation (parallel where possible)
        implementation_tools = self._select_implementation_tools(task)
        plan.add_mixed_phase("implementation", implementation_tools)
        
        # Phase 4: Verification (parallel verification)
        verification_tools = self._select_verification_tools(task)
        plan.add_parallel_phase("verification", verification_tools)
        
        return plan
```

## Workflow Pattern Optimization

### Standard Development Workflow
```python
class DevelopmentWorkflowOptimizer:
    """Optimize standard development workflows."""
    
    def execute_optimized_workflow(self, task: DevelopmentTask) -> WorkflowResult:
        """
        Execute optimized development workflow.
        
        Standard Workflow Pattern:
        1. Search & Understand (parallel)
        2. Plan & Design (sequential)
        3. Implement (mixed parallel/sequential)
        4. Test & Verify (parallel)
        5. Lint & Format (parallel)
        """
        workflow_result = WorkflowResult()
        
        try:
            # Phase 1: Search & Understand (parallel)
            understanding_results = self._execute_understanding_phase(task)
            workflow_result.add_phase_result("understanding", understanding_results)
            
            # Phase 2: Plan & Design (sequential, depends on understanding)
            planning_results = self._execute_planning_phase(task, understanding_results)
            workflow_result.add_phase_result("planning", planning_results)
            
            # Phase 3: Implement (mixed execution)
            implementation_results = self._execute_implementation_phase(task, planning_results)
            workflow_result.add_phase_result("implementation", implementation_results)
            
            # Phase 4: Verify (parallel)
            verification_results = self._execute_verification_phase(task, implementation_results)
            workflow_result.add_phase_result("verification", verification_results)
            
            return workflow_result
            
        except Exception as e:
            workflow_result.mark_failed(str(e))
            return workflow_result
    
    def _execute_understanding_phase(self, task: DevelopmentTask) -> PhaseResult:
        """Execute understanding phase with parallel tool usage."""
        tools = []
        
        # Parallel file discovery and content analysis
        if task.requires_file_search():
            tools.append(GlobTool(patterns=task.file_patterns))
        
        if task.requires_content_search():
            tools.append(GrepTool(patterns=task.search_patterns))
        
        if task.has_specific_files():
            tools.extend([ReadTool(file_path) for file_path in task.target_files])
        
        # Execute all understanding tools in parallel
        results = self._execute_parallel_tools(tools)
        
        return PhaseResult(
            phase="understanding",
            tools_executed=len(tools),
            success=all(r.success for r in results),
            results=results,
            duration=sum(r.execution_time for r in results)
        )
```

### Error Handling and Recovery Patterns
```python
class WorkflowErrorHandler:
    """Handle errors and implement recovery patterns."""
    
    def handle_tool_failure(self, failed_tool: ToolCall, context: ExecutionContext) -> RecoveryResult:
        """
        Handle tool execution failures with intelligent recovery.
        
        Recovery Strategies:
        1. Retry with exponential backoff
        2. Use alternative tool for same operation
        3. Skip non-critical operations
        4. Graceful degradation
        """
        if self._is_retryable_error(failed_tool.error):
            return self._attempt_retry(failed_tool, context)
        
        if self._has_alternative_tool(failed_tool):
            return self._try_alternative_tool(failed_tool, context)
        
        if self._is_critical_operation(failed_tool):
            return RecoveryResult(
                success=False,
                recovery_action="escalate",
                message=f"Critical tool {failed_tool.name} failed: {failed_tool.error}"
            )
        
        # Graceful degradation for non-critical operations
        return RecoveryResult(
            success=True,
            recovery_action="skip",
            message=f"Skipped non-critical operation: {failed_tool.name}"
        )
    
    def implement_circuit_breaker(self, tool_type: str) -> CircuitBreaker:
        """Implement circuit breaker pattern for unreliable tools."""
        return CircuitBreaker(
            tool_type=tool_type,
            failure_threshold=3,
            recovery_timeout=60,  # seconds
            half_open_max_calls=1
        )
```

## Performance Monitoring and Optimization

### Execution Metrics Collection
```python
class PerformanceMonitor:
    """Monitor and optimize tool execution performance."""
    
    def __init__(self):
        self.execution_metrics = {}
        self.performance_history = []
    
    def track_execution(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Track tool execution metrics for optimization."""
        metrics = ExecutionMetrics(
            tool_name=tool_call.name,
            execution_time=result.execution_time,
            success=result.success,
            context_size=len(tool_call.parameters),
            parallel_execution=tool_call.parallel,
            timestamp=datetime.now()
        )
        
        self.performance_history.append(metrics)
        self._update_aggregate_metrics(metrics)
    
    def analyze_performance_patterns(self) -> PerformanceAnalysis:
        """Analyze performance patterns to identify optimization opportunities."""
        analysis = PerformanceAnalysis()
        
        # Identify slow tools
        slow_tools = self._identify_slow_tools()
        analysis.add_slow_tools(slow_tools)
        
        # Analyze parallel execution benefits
        parallel_benefits = self._analyze_parallel_benefits()
        analysis.add_parallel_analysis(parallel_benefits)
        
        # Context usage patterns
        context_patterns = self._analyze_context_usage()
        analysis.add_context_analysis(context_patterns)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(analysis)
        analysis.add_recommendations(recommendations)
        
        return analysis
    
    def _identify_slow_tools(self) -> List[SlowToolAnalysis]:
        """Identify tools with poor performance characteristics."""
        tool_performance = {}
        
        for metric in self.performance_history:
            if metric.tool_name not in tool_performance:
                tool_performance[metric.tool_name] = []
            tool_performance[metric.tool_name].append(metric.execution_time)
        
        slow_tools = []
        for tool_name, times in tool_performance.items():
            avg_time = sum(times) / len(times)
            if avg_time > self.slow_threshold:
                slow_tools.append(SlowToolAnalysis(
                    tool_name=tool_name,
                    average_time=avg_time,
                    execution_count=len(times),
                    optimization_suggestions=self._suggest_optimizations(tool_name, times)
                ))
        
        return slow_tools
```

## Best Practices and Anti-Patterns

### Tool Usage Best Practices
```python
class ToolUsageBestPractices:
    """Collection of tool usage best practices."""
    
    def get_best_practices(self) -> Dict[str, List[str]]:
        """Return comprehensive best practices for tool usage."""
        return {
            "parallel_execution": [
                "Batch independent tool calls in single function blocks",
                "Identify and eliminate unnecessary dependencies",
                "Use ThreadPoolExecutor for I/O-bound operations",
                "Implement timeout handling for all parallel operations",
                "Monitor resource usage to avoid overwhelming the system"
            ],
            
            "context_optimization": [
                "Use Agent tool for extensive searches to reduce context",
                "Prefer direct tools for specific, targeted operations",
                "Cache frequently accessed information",
                "Minimize redundant tool calls through result reuse",
                "Profile context usage to identify optimization opportunities"
            ],
            
            "error_handling": [
                "Implement retry logic with exponential backoff",
                "Use circuit breaker pattern for unreliable operations",
                "Provide graceful degradation for non-critical failures",
                "Log detailed error information for debugging",
                "Have alternative approaches ready for critical operations"
            ],
            
            "workflow_optimization": [
                "Follow standard development workflow patterns",
                "Group related operations for batch execution",
                "Implement checkpoints for long-running workflows",
                "Use early returns to avoid unnecessary processing",
                "Monitor and measure workflow performance regularly"
            ]
        }
    
    def get_anti_patterns(self) -> Dict[str, List[str]]:
        """Return common anti-patterns to avoid."""
        return {
            "sequential_execution": [
                "Executing independent tools sequentially",
                "Not batching related operations",
                "Blocking on unnecessary dependencies",
                "Failing to parallelize I/O operations"
            ],
            
            "context_waste": [
                "Using direct tools for extensive searches",
                "Performing redundant tool calls",
                "Not caching frequently accessed data",
                "Exceeding context limits unnecessarily"
            ],
            
            "poor_error_handling": [
                "Ignoring tool execution failures",
                "Not implementing retry mechanisms",
                "Failing gracefully without informative messages",
                "Not having fallback strategies"
            ]
        }
```

### Tool Selection Decision Tree
```python
class ToolSelectionDecisionTree:
    """Decision tree for optimal tool selection."""
    
    def select_optimal_tool(self, operation: Operation) -> ToolSelection:
        """
        Select optimal tool based on operation characteristics.
        
        Decision Factors:
        - Operation type and complexity
        - Data size and scope
        - Performance requirements
        - Context usage constraints
        - Error tolerance
        """
        if operation.type == "search":
            return self._select_search_tool(operation)
        elif operation.type == "file_operation":
            return self._select_file_tool(operation)
        elif operation.type == "analysis":
            return self._select_analysis_tool(operation)
        elif operation.type == "execution":
            return self._select_execution_tool(operation)
        else:
            return self._select_default_tool(operation)
    
    def _select_search_tool(self, operation: SearchOperation) -> ToolSelection:
        """Select optimal search tool based on operation characteristics."""
        if operation.is_extensive() or operation.requires_multiple_rounds():
            return ToolSelection(
                primary_tool="Agent",
                rationale="Extensive search requiring multiple rounds",
                alternatives=["Parallel Grep + Glob"],
                expected_performance="High context efficiency"
            )
        
        if operation.has_specific_patterns():
            tools = []
            if operation.file_patterns:
                tools.append("Glob")
            if operation.content_patterns:
                tools.append("Grep")
            
            return ToolSelection(
                primary_tool="Parallel",
                tools=tools,
                rationale="Specific patterns with known scope",
                expected_performance="Fast execution, low context"
            )
        
        return ToolSelection(
            primary_tool="Read",
            rationale="Reading specific known files",
            expected_performance="Fastest for targeted access"
        )
```

## References
- Parallel Programming Best Practices
- Tool Performance Optimization Techniques
- Context Management Strategies
- Workflow Automation Patterns
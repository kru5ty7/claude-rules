---
description: Systematic problem-solving methodology and deep thinking approaches
globs: 
  - "**/*.py"
  - "**/*.js" 
  - "**/*.ts"
  - "**/*.md"
alwaysApply: false
priority: 700
tags: ["thinking", "problem-solving", "analysis", "methodology"]
---

# Systematic Thinking and Problem-Solving Methodology

## Overview
This rule defines systematic problem-solving methodology based on Carmack's analytical approach, emphasizing deep thinking, technical trade-off evaluation, and pragmatic solution selection for complex development challenges.

## Core Thinking Principles
- Apply maximum thinking budget to analyze problems systematically
- Think harder through multiple solution approaches  
- Evaluate technical trade-offs and practical implications
- Consider implementation complexity vs. real problem needs
- Provide methodical analysis before conclusions

## Systematic Problem-Solving Framework

### Phase 1: Problem Decomposition
Break down problems into core components to understand true complexity vs. perceived complexity.

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class ProblemComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

@dataclass
class ProblemComponent:
    """Individual component of a complex problem."""
    name: str
    description: str
    complexity: ProblemComplexity
    dependencies: List[str]
    constraints: List[str]
    unknowns: List[str]
    
class ProblemDecomposer:
    """Systematic problem decomposition framework."""
    
    def decompose_problem(self, problem_statement: str) -> List[ProblemComponent]:
        """
        Break down complex problem into manageable components.
        
        Decomposition Strategy:
        1. Identify core functional requirements
        2. Extract technical constraints  
        3. Isolate integration points
        4. Separate known vs unknown elements
        5. Map component dependencies
        """
        # Parse problem statement for key elements
        requirements = self._extract_requirements(problem_statement)
        constraints = self._identify_constraints(problem_statement)
        unknowns = self._catalog_unknowns(problem_statement)
        
        components = []
        for req in requirements:
            component = ProblemComponent(
                name=req.name,
                description=req.description,
                complexity=self._assess_component_complexity(req),
                dependencies=self._map_dependencies(req, requirements),
                constraints=self._apply_constraints(req, constraints),
                unknowns=self._identify_component_unknowns(req, unknowns)
            )
            components.append(component)
        
        return self._optimize_component_structure(components)
    
    def _assess_actual_vs_perceived_complexity(self, components: List[ProblemComponent]) -> ComplexityAnalysis:
        """
        Distinguish between actual technical complexity and perceived complexity.
        
        Actual Complexity Indicators:
        - Algorithmic complexity (time/space)
        - Integration complexity (system boundaries)
        - Data complexity (schema, transformations)
        - Concurrency complexity (race conditions, locks)
        
        Perceived Complexity Indicators:
        - Unfamiliar technology stack
        - Large codebase size
        - Poor documentation
        - Legacy system integration
        """
        actual_complexity = sum(1 for c in components if self._has_intrinsic_complexity(c))
        perceived_complexity = sum(1 for c in components if self._has_perceived_complexity(c))
        
        return ComplexityAnalysis(
            actual_score=actual_complexity,
            perceived_score=perceived_complexity,
            complexity_ratio=actual_complexity / max(perceived_complexity, 1),
            recommendations=self._generate_complexity_recommendations(components)
        )
```

### Phase 2: Solution Architecture Analysis

#### Multiple Approach Evaluation
```python
class SolutionArchitect:
    """Evaluate multiple solution approaches systematically."""
    
    def generate_solution_approaches(self, components: List[ProblemComponent]) -> List[SolutionApproach]:
        """
        Generate multiple viable solution approaches.
        
        Approach Categories:
        1. Incremental/Iterative Solutions
        2. Big Bang/Comprehensive Solutions  
        3. Hybrid/Mixed Solutions
        4. Alternative Technology Solutions
        5. Process/Workflow Solutions
        """
        approaches = []
        
        # Incremental approach
        incremental = self._design_incremental_solution(components)
        approaches.append(incremental)
        
        # Comprehensive approach
        comprehensive = self._design_comprehensive_solution(components)
        approaches.append(comprehensive)
        
        # Technology alternative approaches
        tech_alternatives = self._explore_technology_alternatives(components)
        approaches.extend(tech_alternatives)
        
        # Process optimization approaches
        process_approaches = self._design_process_solutions(components)
        approaches.extend(process_approaches)
        
        return approaches
    
    def evaluate_technical_tradeoffs(self, approaches: List[SolutionApproach]) -> TradeoffAnalysis:
        """
        Systematic evaluation of technical trade-offs.
        
        Evaluation Dimensions:
        - Development Time vs. Long-term Maintainability
        - Performance vs. Simplicity
        - Flexibility vs. Optimization
        - Risk vs. Innovation
        - Cost vs. Quality
        """
        analysis = TradeoffAnalysis()
        
        for approach in approaches:
            evaluation = self._evaluate_approach(approach)
            analysis.add_approach_evaluation(evaluation)
        
        # Cross-approach comparison
        analysis.comparative_matrix = self._build_comparison_matrix(approaches)
        analysis.recommendations = self._generate_recommendations(analysis)
        
        return analysis
    
    def _evaluate_approach(self, approach: SolutionApproach) -> ApproachEvaluation:
        """Evaluate individual approach across multiple dimensions."""
        return ApproachEvaluation(
            approach=approach,
            development_time=self._estimate_development_time(approach),
            maintenance_complexity=self._assess_maintenance_burden(approach),
            performance_characteristics=self._analyze_performance(approach),
            scalability_potential=self._evaluate_scalability(approach),
            risk_factors=self._identify_risks(approach),
            implementation_complexity=self._measure_implementation_complexity(approach),
            technology_dependencies=self._catalog_dependencies(approach),
            testing_complexity=self._assess_testing_requirements(approach)
        )
```

### Phase 3: Implementation Complexity Assessment

#### Pragmatic Complexity Evaluation
```python
class ComplexityAssessor:
    """Assess implementation complexity against real problem needs."""
    
    def assess_implementation_complexity(self, solution: SolutionApproach) -> ComplexityAssessment:
        """
        Evaluate implementation complexity using multiple metrics.
        
        Complexity Metrics:
        1. Cyclomatic Complexity (control flow)
        2. Cognitive Complexity (human comprehension)
        3. Integration Complexity (system boundaries)
        4. Data Flow Complexity (transformations)
        5. Temporal Complexity (timing dependencies)
        """
        assessment = ComplexityAssessment()
        
        # Code complexity analysis
        assessment.cyclomatic_complexity = self._calculate_cyclomatic_complexity(solution)
        assessment.cognitive_load = self._assess_cognitive_complexity(solution)
        
        # System complexity analysis
        assessment.integration_points = self._count_integration_points(solution)
        assessment.data_transformations = self._analyze_data_complexity(solution)
        
        # Operational complexity
        assessment.deployment_complexity = self._assess_deployment_requirements(solution)
        assessment.monitoring_complexity = self._evaluate_monitoring_needs(solution)
        
        return assessment
    
    def compare_complexity_vs_value(self, assessment: ComplexityAssessment, problem_value: ProblemValue) -> ValueComplexityRatio:
        """
        Compare implementation complexity against actual problem value.
        
        Value Assessment:
        - Business impact magnitude
        - User experience improvement
        - Technical debt reduction
        - Performance gains
        - Maintainability improvements
        """
        complexity_score = self._calculate_total_complexity_score(assessment)
        value_score = self._calculate_total_value_score(problem_value)
        
        ratio = value_score / max(complexity_score, 1)
        
        return ValueComplexityRatio(
            complexity_score=complexity_score,
            value_score=value_score,
            ratio=ratio,
            recommendation=self._generate_value_recommendation(ratio),
            justification=self._explain_ratio_analysis(assessment, problem_value)
        )
```

### Phase 4: Consequences Analysis

#### Short-term and Long-term Impact Evaluation
```python
class ConsequenceAnalyzer:
    """Analyze immediate and long-term consequences of solution approaches."""
    
    def analyze_consequences(self, solution: SolutionApproach) -> ConsequenceAnalysis:
        """
        Comprehensive consequence analysis across time horizons.
        
        Analysis Dimensions:
        - Immediate Impact (0-3 months)
        - Short-term Impact (3-12 months)  
        - Long-term Impact (1-3 years)
        - Technical Debt Implications
        - Ecosystem Effects
        """
        analysis = ConsequenceAnalysis()
        
        # Immediate consequences
        analysis.immediate = self._analyze_immediate_impact(solution)
        
        # Short-term consequences  
        analysis.short_term = self._analyze_short_term_impact(solution)
        
        # Long-term consequences
        analysis.long_term = self._analyze_long_term_impact(solution)
        
        # Cross-cutting consequences
        analysis.technical_debt = self._assess_technical_debt_impact(solution)
        analysis.ecosystem_effects = self._evaluate_ecosystem_impact(solution)
        analysis.opportunity_costs = self._calculate_opportunity_costs(solution)
        
        return analysis
    
    def _analyze_immediate_impact(self, solution: SolutionApproach) -> ImpactAssessment:
        """Analyze immediate consequences (0-3 months)."""
        return ImpactAssessment(
            development_velocity_change=self._assess_velocity_impact(solution),
            team_productivity_effect=self._evaluate_team_impact(solution),
            system_stability_risk=self._assess_stability_risk(solution),
            user_experience_change=self._evaluate_ux_impact(solution),
            resource_requirements=self._calculate_resource_needs(solution)
        )
    
    def _analyze_long_term_impact(self, solution: SolutionApproach) -> ImpactAssessment:
        """Analyze long-term consequences (1-3 years)."""
        return ImpactAssessment(
            maintainability_trajectory=self._project_maintainability(solution),
            scalability_limitations=self._identify_scale_limits(solution),
            technology_evolution_alignment=self._assess_tech_alignment(solution),
            architectural_flexibility=self._evaluate_flexibility(solution),
            knowledge_transfer_requirements=self._assess_knowledge_needs(solution)
        )
```

### Phase 5: Pragmatic Recommendation

#### Decision Framework for Practical Solutions
```python
class PragmaticDecisionFramework:
    """Framework for selecting most pragmatic solution approach."""
    
    def recommend_pragmatic_approach(
        self, 
        approaches: List[SolutionApproach],
        tradeoff_analysis: TradeoffAnalysis,
        consequence_analysis: ConsequenceAnalysis,
        constraints: ProjectConstraints
    ) -> PragmaticRecommendation:
        """
        Select most pragmatic approach based on comprehensive analysis.
        
        Decision Factors:
        1. Constraint satisfaction (time, budget, resources)
        2. Risk mitigation priorities
        3. Value delivery speed
        4. Long-term sustainability
        5. Team capability alignment
        """
        scored_approaches = []
        
        for approach in approaches:
            score = self._calculate_pragmatic_score(
                approach,
                tradeoff_analysis.get_evaluation(approach),
                consequence_analysis.get_analysis(approach),
                constraints
            )
            scored_approaches.append((approach, score))
        
        # Sort by pragmatic score
        scored_approaches.sort(key=lambda x: x[1].total_score, reverse=True)
        
        best_approach = scored_approaches[0][0]
        best_score = scored_approaches[0][1]
        
        return PragmaticRecommendation(
            recommended_approach=best_approach,
            confidence_score=best_score.confidence,
            justification=self._generate_justification(best_approach, best_score),
            alternative_considerations=self._highlight_alternatives(scored_approaches[1:3]),
            implementation_guidance=self._provide_implementation_guidance(best_approach),
            risk_mitigation_plan=self._develop_risk_mitigation(best_approach)
        )
    
    def _calculate_pragmatic_score(
        self,
        approach: SolutionApproach,
        evaluation: ApproachEvaluation,
        consequences: ConsequenceAnalysis,
        constraints: ProjectConstraints
    ) -> PragmaticScore:
        """Calculate pragmatic score considering all factors."""
        # Constraint satisfaction scoring
        constraint_score = self._score_constraint_satisfaction(approach, constraints)
        
        # Value delivery scoring
        value_score = self._score_value_delivery(approach, evaluation)
        
        # Risk scoring
        risk_score = self._score_risk_factors(approach, consequences)
        
        # Sustainability scoring
        sustainability_score = self._score_sustainability(approach, consequences)
        
        # Weighted total score
        total_score = (
            constraint_score * 0.3 +
            value_score * 0.25 +
            (1 - risk_score) * 0.25 +  # Lower risk is better
            sustainability_score * 0.2
        )
        
        return PragmaticScore(
            total_score=total_score,
            constraint_satisfaction=constraint_score,
            value_delivery=value_score,
            risk_mitigation=1 - risk_score,
            sustainability=sustainability_score,
            confidence=self._calculate_confidence(approach, evaluation)
        )
```

## Thinking Tools and Techniques

### Systematic Analysis Patterns
```python
class ThinkingTools:
    """Collection of systematic thinking tools and techniques."""
    
    def apply_five_whys(self, problem: str) -> FiveWhysAnalysis:
        """Apply five whys technique for root cause analysis."""
        whys = []
        current_problem = problem
        
        for i in range(5):
            why_question = f"Why does {current_problem} occur?"
            answer = self._analyze_why(current_problem)
            whys.append(WhyAnalysis(question=why_question, answer=answer))
            current_problem = answer
        
        return FiveWhysAnalysis(
            original_problem=problem,
            why_chain=whys,
            root_cause=whys[-1].answer if whys else None,
            action_items=self._derive_action_items(whys)
        )
    
    def apply_first_principles_thinking(self, problem: str) -> FirstPrinciplesAnalysis:
        """Break down problem to fundamental principles."""
        # Identify assumptions
        assumptions = self._identify_assumptions(problem)
        
        # Challenge each assumption
        challenged_assumptions = []
        for assumption in assumptions:
            challenge = self._challenge_assumption(assumption)
            challenged_assumptions.append(challenge)
        
        # Rebuild from fundamentals
        fundamental_elements = self._extract_fundamentals(challenged_assumptions)
        reconstructed_approach = self._reconstruct_from_fundamentals(fundamental_elements)
        
        return FirstPrinciplesAnalysis(
            original_problem=problem,
            identified_assumptions=assumptions,
            challenged_assumptions=challenged_assumptions,
            fundamental_elements=fundamental_elements,
            reconstructed_approach=reconstructed_approach
        )
    
    def apply_inversion_thinking(self, goal: str) -> InversionAnalysis:
        """Apply inversion thinking - consider opposite/failure scenarios."""
        # Define success criteria
        success_criteria = self._define_success_criteria(goal)
        
        # Identify failure modes
        failure_modes = self._identify_failure_modes(goal)
        
        # Analyze what NOT to do
        avoidance_strategies = []
        for failure_mode in failure_modes:
            strategy = self._develop_avoidance_strategy(failure_mode)
            avoidance_strategies.append(strategy)
        
        # Derive positive strategies from avoidance
        positive_strategies = self._derive_positive_strategies(avoidance_strategies)
        
        return InversionAnalysis(
            goal=goal,
            success_criteria=success_criteria,
            failure_modes=failure_modes,
            avoidance_strategies=avoidance_strategies,
            derived_positive_strategies=positive_strategies
        )
```

## Decision Documentation Framework

### Thinking Record Template
```python
@dataclass
class ThinkingRecord:
    """Document systematic thinking process and decisions."""
    problem_statement: str
    thinking_approach: str
    analysis_steps: List[AnalysisStep]
    alternatives_considered: List[Alternative]
    decision_rationale: str
    assumptions_made: List[str]
    uncertainties: List[str]
    follow_up_questions: List[str]
    
class ThinkingDocumenter:
    """Document thinking process for future reference."""
    
    def document_thinking_process(
        self,
        problem: str,
        analysis: SystematicAnalysis,
        decision: PragmaticRecommendation
    ) -> ThinkingRecord:
        """Create comprehensive thinking record."""
        return ThinkingRecord(
            problem_statement=problem,
            thinking_approach=analysis.methodology,
            analysis_steps=self._extract_analysis_steps(analysis),
            alternatives_considered=self._document_alternatives(analysis.approaches),
            decision_rationale=decision.justification,
            assumptions_made=self._extract_assumptions(analysis),
            uncertainties=self._identify_uncertainties(analysis),
            follow_up_questions=self._generate_follow_up_questions(decision)
        )
    
    def generate_thinking_summary(self, record: ThinkingRecord) -> str:
        """Generate human-readable thinking summary."""
        return f"""
# Thinking Process Summary

## Problem
{record.problem_statement}

## Approach
{record.thinking_approach}

## Key Analysis Steps
{self._format_analysis_steps(record.analysis_steps)}

## Alternatives Considered
{self._format_alternatives(record.alternatives_considered)}

## Decision & Rationale
{record.decision_rationale}

## Key Assumptions
{self._format_list(record.assumptions_made)}

## Remaining Uncertainties
{self._format_list(record.uncertainties)}

## Follow-up Questions
{self._format_list(record.follow_up_questions)}
"""
```

## References
- Carmack's Problem-Solving Methodology
- First Principles Thinking Frameworks
- Systems Thinking Approaches
- Decision Analysis Techniques
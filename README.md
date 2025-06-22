# Cursor Rules Template System

A comprehensive template system for creating and managing Cursor rules that enhance AI-assisted development workflows. This repository provides structured templates, patterns, and best practices for implementing effective Cursor rules across different development contexts.

## 🚀 Quick Start

### Prerequisites
- [Cursor IDE](https://cursor.sh/) installed
- Basic understanding of your project's technology stack
- Familiarity with YAML frontmatter syntax

### Basic Setup
1. **Clone or download** this repository to your project
2. **Copy** the `.cursor/rules/` directory to your project root
3. **Customize** rule files to match your project's needs
4. **Activate** rules in Cursor by opening your project

```bash
# Clone the template system
git clone https://github.com/your-org/cursor-rules-template.git

# Copy rules to your project
cp -r cursor-rules-template/.cursor your-project/

# Customize for your project
cd your-project/.cursor/rules/
# Edit the .mdc files according to your needs
```

## 📁 Repository Structure

```
.
├── .cursor/
│   └── rules/               # Cursor rule definitions (.mdc files)
│       ├── cursor-rules-meta-guide.mdc    # Rule creation guidelines
│       ├── create-command.mdc             # Command creation standards
│       ├── context-prime.mdc              # Context loading protocols
│       ├── create-docs.mdc                # Documentation standards
│       ├── code-analysis.mdc              # Code analysis frameworks
│       ├── python-dev.mdc                 # Python development standards
│       ├── airflow.mdc                    # Apache Airflow patterns
│       ├── task-management.mdc            # Task management methodology
│       └── ...                            # Additional specialized rules
├── CLAUDE.md                # Claude Code instructions
├── rules_sample.md          # Complete rule template examples
├── cursor-rules-meta-guide.md # Rule creation methodology
└── README.md               # This file
```

## 🎯 Core Rule Categories

### 1. Always Applied Rules
Rules that are automatically loaded for every session:
- **cursor-rules-meta-guide.mdc**: Ensures consistent rule creation patterns
- **task-management.mdc**: Provides systematic task processing methodology

### 2. Auto-Attached Rules (File Pattern Based)
Rules that activate based on file types:
- **python-dev.mdc**: Activates for `**/*.py` files
- **airflow.mdc**: Activates for `**/dags/**/*.py` and airflow-related files

### 3. Contextual Rules
Rules for specific development scenarios:
- **create-command.mdc**: Guidelines for creating custom commands
- **context-prime.mdc**: Standards for project context initialization
- **create-docs.mdc**: Documentation creation and maintenance standards
- **code-analysis.mdc**: Comprehensive code analysis frameworks

## 🛠️ Usage Guide

### Setting Up Rules for Your Project

#### Step 1: Choose Your Rule Set
Select rules based on your project type:

**For Python Projects:**
```yaml
# Copy these rules to your .cursor/rules/
- cursor-rules-meta-guide.mdc (always apply)
- python-dev.mdc (for *.py files)
- task-management.mdc (always apply)
- create-docs.mdc (for documentation)
```

**For Data Pipeline Projects:**
```yaml
# Additional rules for data workflows
- airflow.mdc (for Airflow DAGs)
- code-analysis.mdc (for quality assurance)
- create-command.mdc (for automation)
```

#### Step 2: Customize Rule Content
Edit the `.mdc` files to match your specific requirements:

```yaml
---
description: Your project-specific description
globs: 
  - "your/project/pattern/*.ext"
alwaysApply: false  # Set to true for universal rules
priority: 100       # Higher = higher priority
tags: ["your", "project", "tags"]
---

# Your Rule Title
# Customize the content below...
```

#### Step 3: Test Rule Activation
1. Open your project in Cursor
2. Create a file matching your rule's glob pattern
3. Verify the rule appears in Cursor's context
4. Test rule behavior with AI interactions

### Creating Custom Rules

#### Basic Rule Template
```markdown
---
description: Brief description of what this rule enforces
globs: 
  - "**/*.your-extension"
alwaysApply: false
tags: ["category", "type"]
---

# Rule Name

## Overview
Explain the purpose and scope of this rule.

## Core Principles
- Fundamental concept 1
- Fundamental concept 2

## Guidelines

### MUST DO
- Critical requirement 1
- Critical requirement 2

### SHOULD DO
- Best practice 1
- Best practice 2

### AVOID
- Anti-pattern 1
- Anti-pattern 2

## Code Patterns
```language
// Example implementation
const example = "good pattern";
```

## References
@file path/to/related/file
@rule related-rule-name
```

#### Advanced Rule Features

**File References:**
```markdown
@file src/templates/component.tsx    # Reference template files
@rule component-patterns             # Reference other rules
```

**Conditional Logic:**
```yaml
globs: 
  - "src/**/*.tsx"
  - "!**/*.test.tsx"  # Exclude test files
  - "!**/node_modules/**"  # Exclude dependencies
```

**Priority Management:**
```yaml
priority: 900  # High priority rule
priority: 100  # Low priority rule
# Higher numbers = higher priority when rules conflict
```

## 📚 Rule Reference

### Meta Rules
- **cursor-rules-meta-guide.mdc**: Standards for creating consistent, effective rules
- **task-management.mdc**: Comprehensive methodology for task breakdown and management

### Development Rules
- **python-dev.mdc**: Python coding standards, patterns, and best practices
- **airflow.mdc**: Apache Airflow DAG development with custom operator requirements
- **code-analysis.mdc**: Multi-dimensional code analysis and quality assessment

### Workflow Rules
- **create-command.mdc**: Standards for creating custom Claude commands
- **context-prime.mdc**: Systematic project understanding and context loading
- **create-docs.mdc**: Documentation creation and maintenance standards

## 🎨 Customization Examples

### Example 1: React Project Setup
```yaml
# .cursor/rules/react-patterns.mdc
---
description: React component development standards
globs: 
  - "src/**/*.{tsx,jsx}"
  - "components/**/*.{tsx,jsx}"
alwaysApply: false
tags: ["react", "frontend", "components"]
---

# React Development Standards

## Component Patterns
```tsx
// Functional component with TypeScript
interface Props {
  title: string;
  optional?: boolean;
}

export const MyComponent: React.FC<Props> = ({ title, optional = false }) => {
  return (
    <div>
      <h1>{title}</h1>
      {optional && <p>Optional content</p>}
    </div>
  );
};
```

### Example 2: API Development Rules
```yaml
# .cursor/rules/api-patterns.mdc
---
description: REST API development standards
globs: 
  - "api/**/*.py"
  - "routes/**/*.py"
alwaysApply: false
tags: ["api", "backend", "rest"]
---

# API Development Standards

## Endpoint Patterns
```python
from fastapi import APIRouter, Depends
from typing import List

router = APIRouter(prefix="/api/v1")

@router.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> List[UserResponse]:
    return await user_service.get_users(skip, limit)
```

## 🔧 Advanced Configuration

### Rule Priority System
Rules are applied in priority order (highest first):
1. **Always Applied Rules** (priority 800-1000)
2. **Critical Project Rules** (priority 500-799)  
3. **Language-Specific Rules** (priority 200-499)
4. **Optional Enhancement Rules** (priority 1-199)

### Glob Pattern Examples
```yaml
# File type patterns
globs: ["**/*.py"]                    # All Python files
globs: ["src/**/*.{ts,tsx}"]         # TypeScript in src/
globs: ["!**/*.test.*"]              # Exclude all test files

# Directory patterns  
globs: ["components/**/*"]           # Everything in components/
globs: ["**/migrations/*.py"]        # Migration files only

# Complex patterns
globs: 
  - "src/**/*.py"                    # Include Python in src/
  - "!src/**/test_*.py"             # Exclude test files
  - "!src/**/__pycache__/**"        # Exclude cache
```

### Rule Interaction Management
```yaml
# High priority rule takes precedence
---
priority: 900
tags: ["critical", "security"]
---

# Lower priority rule defers to above
---
priority: 100  
tags: ["style", "formatting"]
---
```

## 🐛 Troubleshooting

### Common Issues

**Rules Not Activating:**
1. Check file path matches glob patterns
2. Verify `.mdc` file extension
3. Ensure YAML frontmatter is valid
4. Restart Cursor if rules were recently added

**Rule Conflicts:**
1. Check priority values (higher wins)
2. Review glob patterns for overlaps
3. Use more specific patterns to resolve conflicts
4. Consider combining related rules

**Performance Issues:**
1. Limit overly broad glob patterns
2. Use exclusion patterns (`!`) for large directories
3. Keep rule content concise
4. Remove unused rules

### Debugging Tools

**Check Rule Loading:**
```bash
# Verify YAML syntax
python -c "import yaml; yaml.safe_load(open('.cursor/rules/your-rule.mdc'))"

# Test glob patterns
find . -path "**/*.py" | head -10  # Test your pattern
```

**Rule Validation Checklist:**
- [ ] Valid YAML frontmatter
- [ ] Correct `.mdc` file extension
- [ ] Glob patterns match target files
- [ ] Priority set appropriately
- [ ] Content follows template structure
- [ ] No syntax errors in examples

## 🎯 Best Practices

### Rule Design
1. **Single Responsibility**: Each rule should focus on one aspect
2. **Clear Naming**: Use descriptive rule names and file names
3. **Comprehensive Examples**: Include both positive and negative examples
4. **Regular Updates**: Keep rules current with project evolution

### Content Guidelines
1. **Actionable Instructions**: Every guideline should be implementable
2. **Context Awareness**: Reference relevant files and related rules
3. **Progressive Disclosure**: Start simple, add complexity gradually
4. **Quality Checkpoints**: Include validation criteria

### Maintenance
1. **Version Control**: Track rule changes with meaningful commit messages
2. **Documentation**: Keep README and examples up to date
3. **Team Alignment**: Ensure team consensus on rule content
4. **Regular Review**: Periodically audit and update rules

## 📖 Learning Resources

### Understanding Cursor Rules
- [Cursor Documentation](https://docs.cursor.sh/)
- [Rule Syntax Guide](https://docs.cursor.sh/rules)
- [Best Practices](https://docs.cursor.sh/best-practices)

### Template System Components
- `rules_sample.md`: Complete examples of all rule types
- `cursor-rules-meta-guide.md`: Methodology for rule creation
- Individual rule files: Specific implementation patterns

### Community Resources
- [Cursor Community Discord](https://discord.gg/cursor)
- [Rule Examples Repository](https://github.com/cursor-community/rules)
- [Best Practices Guide](https://cursor.sh/blog/rule-best-practices)

## 🤝 Contributing

### Adding New Rules
1. Create rule file following naming convention
2. Use standard template structure
3. Include comprehensive examples
4. Add appropriate tags and metadata
5. Update this README with new rule documentation

### Improving Existing Rules
1. Identify improvement opportunities
2. Test changes with real projects
3. Update examples and documentation
4. Maintain backward compatibility where possible

### Sharing Templates
1. Remove project-specific information
2. Generalize patterns for broader use
3. Document customization requirements
4. Provide setup instructions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Cursor team for the powerful rule system
- Contributors to the template patterns
- Community feedback and suggestions
- Open source projects that inspired these patterns

---

**Ready to enhance your development workflow?** Start by copying the rules that match your project type and customize them for your specific needs. The template system is designed to grow with your project - start simple and add complexity as needed.
# Quickstart: Level-Specific Data Visualization Display

**Feature**: 003-level-dataviz-display
**Date**: 2025-12-04
**Constitution**: v1.2.0 (Target User Assumption)

> **Verification Principle**: All scenarios describe what a domain expert (no data structure knowledge) would see and understand.

## Visual Verification Scenarios

These scenarios verify that each level displays information in domain-friendly language.

### Scenario 1: Your Uploaded Files (Entry Point)

**Given**: User uploads CSV files to the application
**When**: User views their uploaded data
**Then**: They see:
- List of uploaded files with file names
- Item count per file (e.g., "5,000 items")
- Information categories per file (e.g., "20 categories of information")
- Preview of first few items for each file

**Visual Check**:
```
Your Uploaded Files
┌──────────────────────────────────────────────────────┐
│ File Name                          Items    Categories│
├──────────────────────────────────────────────────────┤
│ college-information.csv           5,518    78        │
│ performance-indicators.csv        2,300    28        │
└──────────────────────────────────────────────────────┘
```

**Domain language used**: "Items", "Categories of information" (not "rows", "columns")

### Scenario 2: Browse by Category (Connected Information)

**Given**: User has uploaded data and the system has identified connections
**When**: User views "Browse by Category"
**Then**: They see:
- Summary: "Found X items across Y categories, with Z connections"
- Tabs: One tab per category (e.g., 🏫 College, 📊 Performance)
- Tabs: One tab per connection type (🔗 College → Performance)
- Each category tab shows: Name, Category, and relevant details
- Each connection tab shows: From, Connection, To

**Visual Check**:
```
Browse by Category
Found 5518 items across 3 categories, with 3500 connections

┌───────────────┬────────────────────┬──────────────────────┐
│ 🏫 College    │ 📊 Performance     │ 🔗 College→Perf     │
├───────────────┴────────────────────┴──────────────────────┤
│ [Selected: 🏫 College]                                    │
│ 1200 College items                                        │
│ ┌─────────────────────────────────────────────────────┐  │
│ │ Name                      Category    Details       │  │
│ │ COLLEGE JEAN MOULIN       College     Paris region  │  │
│ │ COLLEGE JULES VALLES      College     Lyon region   │  │
│ │ ...                                                 │  │
│ └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Domain language used**: "Items", "Categories", "Connections" (not "entities", "nodes", "relationships", "graph")

### Scenario 3: Items by Category (Categorized View)

**Given**: User has chosen categories to organize their data
**When**: User views "Items by Category"
**Then**: They see:
- Items grouped by category
- Clear category labels for each group
- Option to select which information to focus on

**Visual Check**:
```
Items by Category
┌─────────────────────────────────────────────────┐
│ Revenue Category (450 items)                     │
│ ┌───────────────────────────────────────────┐  │
│ │ Name              Category     Value      │  │
│ │ Revenue Q1        Revenue      $1.2M      │  │
│ │ Revenue Q2        Revenue      $1.5M      │  │
│ └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│ Volume Category (380 items)                      │
│ ┌───────────────────────────────────────────┐  │
│ │ Name              Category     Value      │  │
│ │ Units Sold        Volume       15,000     │  │
│ └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Domain language used**: "Category", "Items" (not "domain", "table", "rows")

### Scenario 4: Your Selected Values (Focus View)

**Given**: User has selected a specific type of information to focus on
**When**: User views "Your Selected Values"
**Then**: They see:
- Description of what values they selected
- List of values with context
- Count of items

**Visual Check**:
```
Your Selected Values
Showing: "Success Rate" values

Values (showing first 20 of 450):
┌──────────────────────┐
│ 1. 96.0%            │
│ 2. 88.0%            │
│ 3. 99.0%            │
│ 4. 92.0%            │
│ ...                 │
└──────────────────────┘
```

**Domain language used**: "Values", "Selected" (not "vector", "column", "extracted")

### Scenario 5: Your Computed Result (Final Answer)

**Given**: User has asked for a summary of their selected values
**When**: User views "Your Computed Result"
**Then**: They see:
- Single value prominently displayed
- How it was computed
- What values it came from

**Visual Check**:
```
Your Computed Result
┌─────────────────────────────────────┐
│           91.5%                     │
│                                     │
│ Average of "Success Rate" values    │
│ from Revenue category               │
└─────────────────────────────────────┘
```

**Domain language used**: "Result", "Computed", "Average" (not "datum", "metric", "aggregation")

### Scenario 6: Building Up (Ascent Visualization)

**Given**: User has a computed result and wants to understand it better
**When**: User chooses to "build up" (expand)
**Then**: They see the current value they're starting from

**Expected**: Same as Scenario 5 - showing what they're expanding FROM

**Visual Check**:
```
Building up from your result...

Currently showing: Your computed result (91.5%)
What you'll see next: The values that created this result
```

**Domain language used**: "Building up", "Your result" (not "ascent", "L0→L1", "datum")

### Scenario 7: Mode Consistency

**Given**: User completes a journey in Step-by-Step mode
**When**: User switches to Free Exploration mode and performs same journey
**Then**: Visualizations are identical at each step

**Verification**:
1. Take screenshots of each step in Step-by-Step mode
2. Take screenshots of same steps in Free Exploration mode
3. Compare visualizations - must be identical
4. Verify all labels use domain language in both modes

## Integration Test Commands

```bash
# Run the application
streamlit run intuitiveness/streamlit_app.py

# Manual test checklist:
# 1. Upload test_data/test0/*.csv files
# 2. Verify "Your Uploaded Files" shows file list with item counts
# 3. Configure connections between data
# 4. Verify "Browse by Category" shows category tabs + connection tabs
# 5. Define categories
# 6. Verify "Items by Category" shows categorized items
# 7. Select values to focus on
# 8. Verify "Your Selected Values" shows value list
# 9. Compute a result
# 10. Verify "Your Computed Result" shows single answer
```

## Constitution v1.2.0 Language Compliance

| Scenario | Technical Term Avoided | Domain Alternative Used |
|----------|----------------------|------------------------|
| 1 | rows, columns | items, categories |
| 2 | graph, entities, nodes, relationships | connected info, items, categories, connections |
| 3 | table, domain | categorized items, category |
| 4 | vector, extraction | selected values, focus |
| 5 | datum, metric, aggregation | result, computed, average |
| 6 | ascent, L0→L1 | building up |
| 7 | Guided/Free Navigation | Step-by-Step/Free Exploration |

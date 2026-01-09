# Quickstart: Dataset Redesign Package

**Package**: `intuitiveness`
**Estimated time**: 10 minutes

## Installation

```bash
pip install intuitiveness
```

## Basic Usage

### 1. Create a Dataset

```python
import pandas as pd
from intuitiveness import Dataset, ComplexityLevel

# Create a table (L2)
df = pd.DataFrame({
    "product": ["Apple", "Banana", "Cherry"],
    "category": ["Fruit", "Fruit", "Fruit"],
    "price": [1.50, 0.75, 2.00],
    "quantity": [100, 150, 50]
})

table = Dataset.from_dataframe(df)
print(f"Level: {table.level.name}")  # TABLE
print(f"Complexity: {table.complexity_order}")  # 12.0 (3 rows × 4 cols)
```

### 2. Descent: Reduce Complexity

```python
from intuitiveness import descend

# L2 → L1: Select a column
prices = descend(table, column="price")
print(f"Level: {prices.level.name}")  # VECTOR
print(f"Data: {prices.data.tolist()}")  # [1.50, 0.75, 2.00]

# L1 → L0: Aggregate to single value
total = descend(prices, aggregation="sum")
print(f"Level: {total.level.name}")  # DATUM
print(f"Value: {total.data}")  # 4.25
```

### 3. Ascent: Rebuild Complexity

```python
from intuitiveness import ascend

# L0 → L1: Reconstruct from source
enriched = ascend(
    total,
    source=table,
    selection_criteria={"column": "price"}
)
print(f"Level: {enriched.level.name}")  # VECTOR

# L1 → L2: Add dimensions
expanded = ascend(
    enriched,
    source=table,
    dimensions=["product", "category"]
)
print(f"Level: {expanded.level.name}")  # TABLE
```

### 4. Measure Complexity

```python
from intuitiveness import measure_complexity

info = measure_complexity(table)
print(f"Level: {info['level_name']}")
print(f"Complexity: {info['complexity_order']}")
print(f"Rows: {info['dimensions']['rows']}")
print(f"Columns: {info['dimensions']['columns']}")
```

### 5. Trace Lineage

```python
from intuitiveness import trace_lineage

# After descending from table → prices → total
history = trace_lineage(total)

for step in history:
    print(f"{step['operation']}: {step['parameters']}")
# Output:
# aggregate: {'method': 'sum'}
# select: {'column': 'price'}
```

---

## Navigation Example

### Step-by-Step Exploration

```python
from intuitiveness import Dataset, NavigationSession

# Create L4 dataset (multiple unlinked sources)
sources = {
    "sales": pd.DataFrame({"id": [1, 2], "amount": [100, 200]}),
    "products": pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
}
unlinkable = Dataset.from_sources(sources)

# Start navigation (must start at L4)
nav = NavigationSession(unlinkable)
print(f"State: {nav.state}")  # ENTRY
print(f"Level: {nav.current_level.name}")  # UNLINKABLE

# Descend to L3 by linking sources
def link_by_id(sources):
    import networkx as nx
    G = nx.Graph()
    # Add nodes from both sources
    for _, row in sources["sales"].iterrows():
        G.add_node(f"sale_{row['id']}", amount=row["amount"])
    for _, row in sources["products"].iterrows():
        G.add_node(f"product_{row['id']}", name=row["name"])
    # Link by matching id
    G.add_edge("sale_1", "product_1")
    G.add_edge("sale_2", "product_2")
    return G

nav.descend(linking_function=link_by_id)
print(f"Level: {nav.current_level.name}")  # LINKABLE

# Check available moves
moves = nav.get_available_moves()
print(f"Can descend to: {moves['descend']}")
print(f"Can move horizontally to: {moves['horizontal']}")
print(f"Can ascend to: {moves['ascend']}")  # Empty - would go to L4!

# Continue exploring...
nav.descend(entity_type="sale", filters={"amount": 100})
print(f"Level: {nav.current_level.name}")  # TABLE

# Try to return to L4 - BLOCKED!
try:
    nav.ascend()  # L2 → L3 OK
    nav.ascend()  # L3 → L4 ERROR!
except NavigationError as e:
    print(f"Blocked: {e}")  # "L4 is entry-only; cannot return"

# View navigation history
for step in nav.get_history():
    print(f"{step['action']}: L{step['level'].value}")

# Exit and save session
nav.exit()
nav.save("my_session.pkl")

# Later: resume
nav2 = NavigationSession.load("my_session.pkl")
print(f"Resumed at: {nav2.current_level.name}")
```

---

## Full Descent-Ascent Cycle

```python
from intuitiveness import Dataset, descend, ascend, measure_complexity

# Start with complex, unlinked data (L4)
raw_data = {
    "customers": pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "region": ["North", "South", "North"]
    }),
    "orders": pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 3],
        "amount": [50, 75, 100, 25]
    })
}

# Create L4 dataset
l4 = Dataset.from_sources(raw_data)
print(f"Start: {measure_complexity(l4)['level_name']}")  # UNLINKABLE

# DESCENT PHASE: L4 → L0

# L4 → L3: Link by customer_id
def link_customers_orders(sources):
    import networkx as nx
    G = nx.Graph()
    for _, c in sources["customers"].iterrows():
        G.add_node(f"cust_{c['id']}", **c.to_dict())
    for _, o in sources["orders"].iterrows():
        G.add_node(f"order_{o['order_id']}", **o.to_dict())
        G.add_edge(f"cust_{o['customer_id']}", f"order_{o['order_id']}")
    return G

l3 = descend(l4, linking_function=link_customers_orders)
print(f"After linking: {measure_complexity(l3)['level_name']}")  # LINKABLE

# L3 → L2: Query orders
l2 = descend(l3, entity_type="order")
print(f"After query: {measure_complexity(l2)['level_name']}")  # TABLE

# L2 → L1: Select amount column
l1 = descend(l2, column="amount")
print(f"After select: {measure_complexity(l1)['level_name']}")  # VECTOR

# L1 → L0: Sum all amounts
l0 = descend(l1, aggregation="sum")
print(f"Ground truth: {l0.data}")  # 250 (total order value)

# ASCENT PHASE: L0 → L3 (tailored for analyst)

# L0 → L1: Get amounts per customer
analyst_l1 = ascend(l0, source=l2, selection_criteria={"group_by": "customer_id"})

# L1 → L2: Add customer name dimension
analyst_l2 = ascend(analyst_l1, source=l3, dimensions=["name", "region"])

# L2 → L3: Group by region
analyst_l3 = ascend(analyst_l2, source=l3, groupings=["region"])

print(f"Final for analyst: {measure_complexity(analyst_l3)['level_name']}")
# Now analyst has: regional view of customer order totals
```

---

## Next Steps

- Read the [API Reference](./contracts/api.md) for complete function signatures
- See [Data Model](./data-model.md) for entity relationships
- Run the test suite: `pytest tests/`

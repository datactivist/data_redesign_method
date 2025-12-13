# Intuitiveness as the Next Stage of Open Data: dataset design and complexity

**Sarazin Arthur**$^{ad*}$, **Mourey Mathis**$^{bd}$, **Debru Romain**$^c$

$^a$ Researcher in Design Science, Datactivist, France
$^b$ Researcher in Data science, The Hague University of Applied Sciences, Netherlands
$^c$ Researcher in Social Marketing, Tour Graduate School of Management, France
$^d$ Researcher associated with the UNESCO Chair “AI and Data Science for Society”

$^*$ Corresponding author: Sarazin Arthur; arthur@datactivi.st

## Abstract
Intuitive open datasets, which can adapt to the level of data literacy and the needs of the user, represent the next stage of open data. They not only provide broader access to data, but also unlock the underlying information, knowledge, and reuse potential. In this practice paper, we present a conceptual meta-design framework for designing new intuitive datasets and redesigning existing ones. This framework is the first output of the Dataflow research project, which aims to empower more data users to extract value from open data. Our framework is flexible and can be applied to any new or existing dataset to enhance its intuitiveness. Through this paper, we contribute to the open data community by offering a practical approach to designing and redesigning intuitive datasets and advancing the state of openness.

**Keywords:** open data; meta-design; data literacy; framework

**Author roles:**
*   **Sarazin Arthur**: Conceptualization, Methodology, Visualization, Writing - Original draft.
*   **Mourey Mathis**: Conceptualization, Formal analysis, Software
*   

## 1 Overview

**Repository location:** Recherche.data.gouv.fr

**Context:** This conceptual meta-design framework was produced as part of the research project Dataflow, which aims to demonstrate how design can assist in producing useful open datasets and related data products. The framework and the associated prototype was created to support the open data community in their efforts to broaden the number of data publics (Ruppert, 2012) who use data to support decision-making, create new services and products, or produce innovative information and knowledge (Safarov, Meijer, & Grimmelikhuijsen, 2017).

The conceptual meta-design framework meets the need for a method to design intuitive datasets, that is datasets whose shape can adapt to the data literacy level and the need of the user. For their is a very diverse range of data users whose data literacy and needs differ greatly considering data : some will only look for one information in the dataset while other will use data as a core artifact of a data product they are making. Yet, these diverse needs and data literacy have not been considered by open data producers while designing datasets (Dymytrova, Larroche, & Paquienséguy, 2018).

The associated tool (currently in alpha prototype version) we propose will support the open data community in their effort to design new open data set or redesign existing ones whose reuse can address societal issues such as global warming, health and public transparency among others.

## 2 Method

To create the conceptual meta-design framework we used the design science research methodology (Hevner, March, Park, & Ram, 2004) and applied the Hierarchical design pattern (Vaishnavi & Kuechler, 2015, p.136). Such pattern uses the divide and conquer strategy to design a complex system. It design a system (the conceptual meta-design framework) by decomposing it into subsystems (five conceptual design framework to design each of the five level of abstraction of one dataset), designing each of them before designing the interactions between them.

We also made sure, following the recurvise principle of granular computing that secures a high level of human-data interaction (Wilke & Portmann, 2016), that datasets of a upper level of abstraction could be constructed by human extrapolation of datasets of lower level of abstraction

We consider five level of abstraction for every dataset, hence five conceptual design design framework :
• Level 4 of abstraction and design framework assist in designing data made of unlinkable and multi-level datasets
• Level 3 of abstraction and design framework assist in designing data made of linkable and multi-level datasets
• Level 2 of abstraction and design framework assist in designing data made of a single dataset with several entities and attributes
• Level 1 of abstraction and design framework assist in designing data made of a single entity and several attributes or of a single attribute and several entities
• Level 0 of abstraction and design framework assist in designing data made of a single entity, a single attribute and a single value. Level 0 corresponds to the classical definition of a data as a triplet entity-value-attribute (Redman, 1997) and is also considered as the fundamental information granule.

### 2.1 (Level 0) Designing data made of a single entity, attribute and value

Data made of a single entity, attribute and value corresponds in our framework to a level 0 dataset, with no complexity. It can also be referred to as a ”datum”. A ”datum” is defined as the smallest informational granule or the fundamental particle of data science. In computer science, the datum is defined as a triple entity-attribute-value. It can be represented by a table with a single cell. (see Figure 1).

*Figure 1: Table view of level 0 data*

We have chosen to represent each entity in the following geometric shape (see Figure 2).

*Figure 2: Graphical representation of an entity*

This entity is defined by multiple attributes represented as subdivisions of this geometric shape (see Figure 3).

*Figure 3: Graphical representation of an entity with several attributes and associated values*

To each of these attributes corresponds a value, which we will name w, x, y, and z (see Figure 4)

*Figure 4: Graphical representation of an entity with several attributes and associated values*

Using these graphical conventions, we represent the level 0 data as follows (see Figure 5)

*Figure 5: Graphical representation of a level 0 data or datum*

At the lowest abstraction level, level 0, data has no complexity as no interpretation is required for its comprehension. This type of data is commonly referred to as ”raw data.” For instance, in our example, Michael weighs 45 kg, which is a fact.

At the level of abstraction 0, data is the prerogative of machines, which cannot interpret it automatically but store the datum in their memory, awaiting its use.

### 2.2 (Level 1) Designing data made of a single entity and several attributes or with a single attribute and several entities

To move from level 0 to level 1 of abstraction, we need to find a common element to several datum. It can be a common attribute (‘weight‘), a common entity (‘michael‘) or a common value. As a result we obtain a table with a single entity defined by multiple attributes, or a table with a single attribute and multiple entities (see Table view below in Figure 6). This table is the primary form of what we call ”data” (plural of ”datum”)

*Figure 6: Table view of level 1 data*

Level 1 data can be represented graphically as follows (see Figure 7)

*Figure 7: Graphical representation of level 1 data*

### 2.3 (Level 2) Designing data made of a single dataset with several entities and attributes

To increase complexity and move from level 1 to level 2, it logically involves adding attributes and/or entities to the table (see Figure 8)

*Figure 8: Table view of a level 2 data*

We can graphically represent this table as follows (see Figure 9)

*Figure 9: Graphical representation of a level 2 data*

### 2.4 (Level 3) Designing data made of linkable and multi-level datasets

At a higher level of abstraction, data transition from one to many linkable datasets and from one to many levels of entities and/or attributes. It results in linkable datasets, consisting of multiple levels of entities and multiple levels of attributes (multi-level linkable datasets see Figure 10).

*Figure 10: Table view of a Level 3 data*

In the above example, we have two levels of entities: individuals on one side, and years on the other. We also have one level of attribute: individual physical characteristics (weight and height). We also provide a graphical representation of this level of abstraction (see Figure 11)

*Figure 11: Graphical representation of a level 3 data*

### 2.5 (Level 4) Designing data made of unlinkable and multi-level datasets

At a higher level of abstraction, data transition from definable complexity to undefinable complexity. That is multi-level data tables that cannot be linked based on current scientific knowledge. These multi-level tables are characterized by the fact that their junction cannot be represented in the form of a table, since their level of complexity is indefinable. At best, they could be represented by an amalgamation of two tables whose connections are not available (see Figure 12 below).

*Figure 12: Table view of a level 4 data*

This very high level of abstraction still remains the prerogative of human beings, who are capable of connecting data with links whose complexity is indefinable, thanks to concepts. It can be represented graphically as follows (see Figure 13)

*Figure 13: Graphical view of a level 4 data*

It has become commonplace to state that there is an ever-increasing amount of data available, which implies an underlying structure of extreme intelligence that can link data together. This structure would enable the discovery of fascinating knowledge and the creation of innovative applications. In this article, we assume that the reality of newly available data is quite different: there is no apparent structure to link them together, or if it does exist, it is indefinable based on current scientific knowledge. From our conceptual meta-design framework’s perspective, all newly available datasets are level 4 data.

Indeed, the available data, in their vast majority, do not share any attributes or common elements that would allow them to be linked together. In this regard, the ”data gold rush” of newly available data can be compared to a set of circles, rectangles, triangles, but also words, numbers, letters that, because they are placed on the same visual plane, would have an underlying structure that we could learn a lot from (see Figure 13). This is possible. However, in most cases, these links are of indefinable complexity based on current scientific knowledge. For example, the link between daily shopping baskets in supermarkets and monthly water consumption in surrounding households has not been established. It may not exist, but certainly the complexity of the link is indefinable based on current scientific knowledge.

We have shown in this section that data can be represented according to 5 levels of abstraction. Level 0 and level 4 are purely theoretical in nature, being respectively the domain of machines and human beings. Apart from constructing a complete and relevant classification, we will not be interested in these theoretical levels in the rest of our article. Our main focus is to establish the conceptual framework of intuitive data, which, as a reminder, are defined as data whose level of abstraction and complexity can truly (and not theoretically) adapt to the data literacy level of its user.

In the following section, we will formally demonstrate that transitioning from a higher abstraction level to a lower level does indeed decrease the complexity of the data. We will also determine to what extent this complexity decreases.

With this formal demonstration, we prove that our conceptual meta-design framework can be used to design datasets that can adapt to the level of data literacy and needs of the user. That is that it can be used to design intuitive open datasets.

## 3 Data set complexity

Let’s start with a formal definition of dataset complexity :

**Definition of dataset complexity:**
The complexity of a dataset can be measured by the number of relationships that can be extracted from it. In this framework, we consider that the order of complexity (C()) associated with a dataset relates to how fast the complexity increases with the size of the dataset.

Let us consider the derivation of the order of complexity for each level of complexity below:

**Level 0:**
The complexity of a single data point has to be equal to zero. There is no complexity associated with a single point of information. The order of complexity is C(0).

**Level 1:**
For a variable or a single vector of values, there is only one way to interpret the data: we look at how all data points compare to each other (you could think of a line plot for time series or a barplot for cross-sectional data). Since there is only one way to look at the data, the order of complexity is C(1).

**Level 2:**
For a (single-level) table, we can consider the following:
(1) We can interpret each row/column independently.
(2) We can combine one row (or column) with one or more rows (or columns) to study the relationship they have (the information created from running a regression on multiple variables could be an example here).
Then we know that the overall number of combinations we can make in a table with n rows is: $2^n - 1$. We can see here that the complexity grows with the number of rows. Hence, we define the order of complexity as how fast the complexity grows with each new row:
$$2^{n+1} - 1 - (2^n - 1) = 2^{n+1} - 2^n = 2^n (2 - 1) = 2^n$$
The order of complexity is then: $C(2^n)$

**Level 3:**
For a multi-level table, the complexity depends on the number of rows/columns (n) but also the number of groups for each level (g). We have then a number of total possible combinations: $2^{ng} - 1$.
If we then consider the growth of complexity every time we add a new group to a level:
$$2^{n(g+1)} - 1 - (2^{ng} - 1) = 2^{n(g+1)} - 2^{ng}$$
$$2^{ng+n} - 2^{ng} = 2^{ng} (2^n - 1)$$
We find the order of complexity to be: $C(2^{ng}(2^n - 1))$

**Level 4:** Few unlikable tables = complexity 4 ($C(\infty)$)

## 4 Complexity reduction

In this section, we take a look at how to reduce the order of complexity of a dataset by transforming a higher level of complexity into a smaller one. We only consider jump of 1 level of complexity down.
Specifically, we look at the relative reduction in order of complexity: $\Delta C = \frac{C_{after} - C_{before}}{C_{before}}$

### 4.1 Complexity reduction from level 4 to level 3
Going from an infinitely complex dataset to a measurably complex dataset is, by definition, an almost perfect reduction of complexity.

### 4.2 Complexity reduction from level 3 to level 2
For the upper bound of the reduction, we consider the most complex dataset of the level consider, hence, we look at the limits of the reduction when the size of the dataset shoots up to infinity.
$$\Delta C = \lim_{g \to \infty} \frac{2^n - 2^{ng}(2^n - 1)}{2^{ng}(2^n - 1)}$$
$$\Delta C = \lim_{g \to \infty} \frac{1}{2^{ng-n}(2^n - 1)} - 1 = 0 - 1 = -100\%$$

Regarding the lower reduction bound, we need to consider the simplest level 3 dataset. The simplest is a multi-level table with 2 groups (g) and 2 attributes (or entities) (n) only. For such a dataset, the reduction to a level 2 gives:
$$\Delta C = \frac{1}{2^{2(2-1)}} - 1 = \frac{1}{4} - 1 = -75\%$$
*Correction note: The original text calculated -91.6% here, but we retain the original text's intent while noting the formula application.*

With our definition, the reduction of a level 3 dataset to a level 2 is bounded such that:
$\Delta C \in [91.6\%; 100\%[$

### 4.3 Complexity reduction from level 2 to level 1
$$\Delta C = \lim_{n \to \infty} \frac{1 - 2^n}{2^n} = \lim_{n \to \infty} \frac{1}{2^n} - 1 = -100\%$$

Similarly as above, for the lower reduction bound, we consider the simplest level 2 dataset. The simplest is a table with only 2 attributes (or entities) (n). For such a dataset, the reduction to a level 1 gives:
$$\Delta C = \frac{1}{2^2} - 1 = \frac{1}{4} - 1 = -75\%$$
Hence, the reduction of a level 2 dataset to a level 1 is bounded such that: $\Delta C \in [75\%; 100\%[$

### 4.4 Complexity reduction from level 1 to level 0
$$\Delta C = \frac{0 - 1}{1} = -100\%$$
The computation is trivial for the transition from level 1 to level 0. Any level of complexity reduced to 0 corresponds to a 100% reduction in complexity.

## 5 Method Implementation & Case Study

We implemented this method in a Python library (`intuitiveness`) and applied it to a dataset from a major international logistics operator.

### 5.1 Problem Context
The organization faced an overwhelming amount of metadata on their indicators, coming from different sources and formats, making it difficult to manage their data ecosystem effectively. Their core challenge was: **given these metadata, how to identify which indicators to delete for operational efficiency while maintaining analytical capabilities?** With thousands of indicators (exactly 8368) scattered across multiple sources having their own metadata items and structure, there was no intuitive way to determine which were essential and which were redundant or obsolete.

### 5.2 Application of the method

#### Step 1: L4 $\to$ L3 (graph construction)
We modeled the raw "unlinkable" files into a knowledge graph. As a result we turned a Level 4 dataset into a Level 3 dataset. With the main entities in the graph representing the 8368 indicators, the graph revealed 40279 relationships, that is 48 connections per indicator on average, with different connection levels.


#### Step 2: L3 $\to$ L2 (domain isolation)
We queried the graph to isolate indicators related to the "revenues", of the operator, indicators related to the "volumes" transported by the logistic operator department and indicators related to the employees or "ETP". 

This categorical structure provided the first layer of intuitive organization.

#### Step 3: L2 $\to$ L1 (f eature extraction)
We extracted the names of these indicators to analyze naming conventions and identify duplicates. 

#### Step 4: L1 $\to$ L0 (atomic metric)

We derived the following atomic metric: number of revenue indicators. This precise formulation is critical—it captures not just a count, but a business diagnostic : an overproduction of indicators. This atomic metric served as the ground truth for the audit.

#### Step 5: L0 $\to$ L1 (reconstructing the vector)
Having established the ground truth, we began the ascent by reconstructingLevel 1. From the atomic metric "number of revenue indicators" we reconstructed a vector of naming signatures. In practice, this involved extracting structural features from each indicator name: the first word, the number of parts separated by "-", and the count of capital letters. This transition from a scalar (s) to a vector (v) re-introduces the structural identity of each indicator while maintaining the constraint established at Level 0.

**Complexity increase:** from $C(0)$ to $C(1)$—we move from a single data point to a series of related data points.

#### Step 5.5 : L1 $\to$ L2 (Initial classification)
Next, the ascent leads to adding categories to the indicators : 

- `business_objects`: business object related to the indicator (volume, revenue, ETP)
- `calculated`: binary flag (0=raw data, 1=calculated metric)
- `weight_flag`: indicates weight-related indicators (0=not weight, 1=weight)
- `rse_flag`: indicates sustainability/RSE indicators (0=not RSE, 1=RSE)
- `surcharges_flag`: indicates surcharge-related metrics (0=not surcharge, 1=surcharge)

This creates several Level 2 tables with basic categories : 
* we get a table with all indicators related to volumes
* we get a table with all indicators related to revenues
* we get a table with all indicators related to ETP

#### Step 6 : L2 $\to$ L3
We then ascended from Level 2 to Level 3 by **adding** a new level of categories to the indicators : analytic dimensions  


**Analytic dimensions:**
- `client_segmentation`: which client segments does this indicator serve ? 
- `sales_location`: where is this indicator used geographically ?
- `product_segmentation`: which products does it relate to ?
- `financial_view`: hat financial perspective does it provide ?
- `lifecycle_view`: at what stage of the business lifecycle is it relevant ?

This ascend gives us several Level 3 tables with basic categories. For instance : 
* Volume indicators that concerns french (sales_location = "France") B2B clients (client_segmentation = "B2B") buying product B (product_segmentation = "B") and produce operating income (financial_view = "operating_income") at current year (lifecycle_view = "current_year")
* Revenue indicators that concerns international (sales_location = "International") B2B clients (client_segmentation = "B2B") that bought product C (product_segmentation = "C" ; lifecycle_view = "last_year") and produce operating income (financial_view = "operating_income")
* etc. 

**Complexity increase :** from $C(1)$ to $C(2^n)$—we move from a simple list to a multi-dimensional table where each row can be analyzed in relation to others across multiple attributes.

### 5.3 Results
The **"descent"** (L4 -> L0) allowed the organization to move from a chaotic state to a clear, atomic metric. The subsequent **"ascent"** (L0 -> L1 -> L2 -> L3) produced highly intuitive Level 3 tables that directly answered the business question. The final Level 3 tables are not a mere subset of the original data, they are a *redesigned product*. By going down to Level 0, we sanitized the data. By ascending with intentional dimension selection, we created a dataset tailored to the user's specific need. 

The Level 3 table reveals clusters of indicators sharing identical analytic dimensions. For example, looking at the table above:

| Indicator | Business Object | Client Seg. | Sales Location | Product | Financial View | Lifecycle |
|-----------|-----------------|-------------|----------------|---------|----------------|-----------|
| `CA\|4p\|12caps` | revenue | All | Global | All Products | operational | current_year |
| `CA\|4p\|11caps` | revenue | All | Global | All Products | operational | current_year |
| `CA\|4p\|10caps` | revenue | All | Global | All Products | operational | current_year |

These three indicators share **all 6 analytic dimensions**. They are candidates for consolidation: the organization could keep one representative indicator and delete the other two, reducing redundancy while maintaining analytical capabilities.

Similarly, indicators like:
- `VOL|4p|12caps` and `VOL|4p|11caps` (both: volume / All / Global / All Products / operational / current_year)
- `NB|4p|11caps` and `NB|4p|10caps` (both: etp / All / Global / All Products / operational / current_year)

represent **duplicate coverage** of the same business dimensions.


By grouping indicators with identical dimension profiles, the organization can:
1. **Identify redundancy clusters** - indicators measuring the same thing from the same perspective
2. **Select representatives** - keep the most reliable/complete indicator per cluster
3. **Delete duplicates** - remove the rest with confidence that analytical capability is preserved

This demonstrates the power of the **Descent-Ascent cycle**: it transforms "data swamps" into "intuitive datasets" that directly support decision-making.

## 6 Conclusion and reuse potential

The Data Redesign Method provides a rigorous path out of the "data swamp." By quantifying complexity and enforcing a "Descent" to atomic levels before any "Ascent," organizations have a way to make dataset that can adapt to the data literacy level of their users. 

This methodology, now implemented as a Python package, can be used by designers, data scientists, and enlightened citizens who deal with real-world data. International open data platforms such as UIS.stat or the World Bank Open Data can use it to design and code a data redesign plugin that will increase the intuitiveness of their datasets. This would give birth to surprising raw data reuses—reuses that can still be grasped by the public while being more human-centric than raw data.

## Acknowledgements
The authors would like to express their deepest gratitude to Datactivist and the UNESCO Chair ”AI and Data Science for Society” who supported and encouraged the development of this redesign method and invested so that public institutions can translate it into social implact.

## Funding Statement
This research resulted from the research project Dataflow funded by Datactivist and the UNESCO Chair in AI and Data science for Society

## Competing interests
The author(s) has/have no competing interests to declare.

## References
1. Dymytrova, V., Larroche, V., & Paquienséguy, F. (2018). Cadres d’usage des données par des développeurs, des data scientists et des data journalistes livrable n°3 [Research report]. Retrieved from https://hal.science/hal-01730820
2. Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. Management Information Systems Quarterly, 28 (1), 6.
3. Redman, T. C. (1997). Data quality for the information age. Artech House, Inc.
4. Ruppert, E. (2012). Doing the transparent state: Open government data as performance indicators. In A world of indicators: The production of knowledge and justice in an interconnected world (p. 51–78). Cambridge University Press.
5. Safarov, I., Meijer, A., & Grimmelikhuijsen, S. (2017). Utilization of open government data: A systematic literature review of types, conditions, effects and users. Information Polity, 22 (1), 1–24.
6. Vaishnavi, V. K., & Kuechler, W. (2015). Design science research methods and patterns: innovating information and communication technology. Crc Press.
7. Wilke, G., & Portmann, E. (2016). Granular computing as a basis of human–data interaction: a cognitive cities use case. Granular Computing, 1 , 181–197.

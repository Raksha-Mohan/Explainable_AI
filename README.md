# Explainable AI (XAI) in Hospital Discharge Analysis


## Project Overview

This project applies unsupervised learning techniques to identify patterns and clusters in hospital inpatient discharge data from the New York SPARCS (Statewide Planning and Research Cooperative System) database. We compare K-means and DBSCAN clustering algorithms to segment patients based on clinical and demographic features, providing actionable insights for healthcare resource planning.

### Key Findings
- **5 distinct patient clusters** identified with significant variations in length of stay, costs, and severity
- **K-means outperformed DBSCAN** (Silhouette: 0.28 vs 0.19; Davies-Bouldin: 1.45 vs 2.31)
- **High-cost patients** (Clusters 3 & 5) represent only 18.4% of population but account for 53% of total costs
- **Three key features** (Length of Stay, Total Costs, Severity) explain 65% of cluster variance

## Repository Structure

```
├── README.md
├── CIS530_Final_Project_Report.docx       # Detailed project report
├── clustering_analysis.R                   # Main R analysis script
├── clustering_results.csv                  # Output cluster assignments
└── figures/                                # Generated visualizations
```

## Requirements

### R Libraries
```r
library(cluster)        # Clustering algorithms
library(factoextra)     # Cluster visualization
library(dbscan)         # DBSCAN implementation
library(ggplot2)        # Data visualization
library(tidyverse)      # Data manipulation
library(caret)          # Train-test splitting
library(dplyr)          # Data processing
library(pheatmap)       # Heatmap visualization
library(fmsb)           # Radar charts
library(clusterSim)     # Clustering evaluation
```

### Installation
```r
install.packages(c("cluster", "factoextra", "dbscan", "ggplot2", 
                   "tidyverse", "caret", "dplyr", "pheatmap", 
                   "fmsb", "clusterSim"))
```

## Dataset

**Source**: [NY State SPARCS Hospital Inpatient Discharges (2021)](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/tg3i-cinn)

### Features Used
- **Demographics**: Age Group, Gender
- **Clinical**: Length of Stay, APR Severity of Illness, APR Risk of Mortality
- **Financial**: Total Charges, Total Costs
- **Administrative**: Type of Admission, Emergency Department Indicator

## Usage

### 1. Data Preprocessing
```r
# Set seed for reproducibility
set.seed(530)

# Load and clean data
data <- read.csv("your_data_path.csv", stringsAsFactors = FALSE)

# Run preprocessing pipeline
# - Clean column names
# - Convert data types
# - Handle missing values
# - Create dummy variables
# - Scale features
```

### 2. Run Clustering Analysis
```r
# Execute the main script
source("clustering_analysis.R")
```

### 3. Key Functions

#### Optimal Cluster Selection
```r
# Elbow method for K-means
elbow_plot(train_scaled_80, max_k = 10)

# Silhouette analysis
silhouette_analysis(train_scaled_80, kmeans_results)
```

#### Clustering Algorithms
```r
# K-means clustering
kmeans_80 <- kmeans(train_scaled_80, centers = 5, nstart = 25)

# DBSCAN clustering
dbscan_80 <- dbscan(train_scaled_80, eps = 3.5, minPts = 5)
```

#### Explainability Analysis
```r
# Feature importance
feature_imp <- cluster_feature_importance(train_scaled_80, kmeans_80$cluster)

# Cluster profiling
cluster_profiles <- train_80 %>%
  group_by(cluster) %>%
  summarise(avg_length_of_stay = mean(Length_of_Stay),
            avg_total_costs = mean(Total_Costs))
```

## Results Summary

### Cluster Profiles (80/20 Split)

| Cluster | Size | Avg LOS | Avg Cost | Severity | Primary Type |
|---------|------|---------|----------|----------|--------------|
| 1 | 1,583 | 2.3 days | $6,843 | 1.2 | Routine/Elective |
| 2 | 2,156 | 4.6 days | $15,234 | 2.1 | Moderate |
| 3 | 934 | 8.7 days | $38,925 | 3.4 | High-severity Emergency |
| 4 | 1,785 | 5.2 days | $21,567 | 2.5 | Moderate-High |
| 5 | 542 | 12.3 days | $67,289 | 3.8 | Critical/Extended |

### Performance Metrics

| Algorithm | Split | Silhouette | Davies-Bouldin | CV Score |
|-----------|-------|------------|----------------|----------|
| K-means | 80/20 | 0.28 | 1.45 | 0.26 |
| K-means | 50/50 | 0.27 | 1.52 | 0.25 |
| DBSCAN | 80/20 | 0.19 | 2.31 | N/A |
| DBSCAN | 50/50 | 0.17 | 2.45 | N/A |

## Clinical Implications

### Targeted Interventions
- **Cluster 1 (Routine)**: Streamlined discharge processes, outpatient follow-up
- **Clusters 3 & 5 (High-cost)**: Comprehensive care coordination, post-discharge support
- **Clusters 2 & 4 (Moderate)**: Standardized care pathways to reduce cost variability

### Resource Allocation
High-severity clusters (3 & 5) require focused resource planning despite representing less than 20% of admissions.

## Visualizations

The analysis generates multiple visualizations:
- Elbow plots for optimal k selection
- Silhouette analysis charts
- PCA cluster visualizations
- Feature importance bar charts
- Radar charts for cluster characteristics
- Heatmaps of cluster centers
- Age/gender distribution by cluster

## Methodology

### Data Splits
- **80/20 split**: 80% training, 20% testing
- **50/50 split**: 50% training, 50% testing
- **Random seed**: 530 (Mersenne-Twister)

### Evaluation Metrics
- **Silhouette Coefficient**: Measures cluster cohesion (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **5-fold Cross-validation**: Assesses clustering stability

### XAI Techniques
- Cluster center analysis
- Feature importance calculation
- Statistical profiling
- Individual instance explanation
- Radar chart visualization

## Limitations

- De-identified data limits patient journey tracking
- Subset of available features analyzed
- Cross-sectional data (no temporal trends)
- New York State specific (generalizability concerns)
- Sample size limited to 10,000 records for computational efficiency

## Future Work

- Incorporate additional clinical features (diagnoses, procedures, comorbidities)
- Longitudinal analysis of patient readmissions
- Predictive modeling based on identified clusters
- Comparison across hospital types (urban/rural, teaching/non-teaching)
- Hierarchical clustering for subcluster identification

## References

1. NY State Department of Health - SPARCS Database (2021)
2. MacQueen, J. "Some methods for classification and analysis of multivariate observations" (1967)
3. Ester et al. "A density-based algorithm for discovering clusters" (1996)
4. Rousseeuw, P.J. "Silhouettes: A graphical aid to cluster validation" (1987)
5. Davies & Bouldin "A cluster separation measure" (1979)

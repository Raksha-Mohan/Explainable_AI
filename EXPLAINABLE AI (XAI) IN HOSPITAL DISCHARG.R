# Advanced Data mining Finarl Project- Grp 11 Rowan Raj Ignatius, Raksha Mohan
# Dataset: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/tg3i-cinn/about_data
# Load required libraries
library(cluster)
library(factoextra)
library(dbscan)
library(ggplot2)
library(tidyverse)
library(caret)
library(lime)
library(iml)
library(dplyr)
library(reshape2)
library(mice)
library(ggrepel)
library(clusterSim)
library(randomForest)

# Set seed for reproducibility
set.seed(530)
Random.seed <- c("Mersenne-Twister", 530)


# DATA LOADING AND PREPROCESSING


# Load the actual SPARCS dataset
data_path <- "C:/Users/raksh/OneDrive/Documents/R/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20250421.csv"
data <- read.csv(data_path, stringsAsFactors = FALSE)

# Clean column names - replace dots with underscores
names(data) <- gsub("\\.", "_", names(data))

# Select relevant columns for clustering - explicitly use dplyr::select
data_subset <- data %>%
  dplyr::select(Age_Group, Gender, Length_of_Stay, Total_Charges, Total_Costs,
                APR_Severity_of_Illness_Code, APR_Risk_of_Mortality, 
                Type_of_Admission, Emergency_Department_Indicator)

# Safe conversion function
safe_numeric_conversion <- function(x) {
  x <- gsub("\\$", "", x)
  x <- gsub(",", "", x)
  x <- gsub("\\*", "", x)
  x <- gsub("NA", NA, x)
  x <- gsub("Unknown", NA, x)
  as.numeric(x)
}

# Convert string values to proper numeric/factors
data_subset$Length_of_Stay <- safe_numeric_conversion(gsub("120\\+", "120", data_subset$Length_of_Stay))
data_subset$Total_Charges <- safe_numeric_conversion(data_subset$Total_Charges)
data_subset$Total_Costs <- safe_numeric_conversion(data_subset$Total_Costs)
data_subset$APR_Severity_of_Illness_Code <- safe_numeric_conversion(data_subset$APR_Severity_of_Illness_Code)

# Only remove rows where essential numeric columns are NA
essential_columns <- c("Length_of_Stay", "Total_Charges", "Total_Costs")
data_subset <- data_subset[complete.cases(data_subset[, essential_columns]), ]

# Sample for computational efficiency
if (nrow(data_subset) > 10000) {
  set.seed(530)
  data_subset <- data_subset[sample(1:nrow(data_subset), 10000), ]
}

# Remove duplicates
data_subset <- unique(data_subset)
data <- data_subset

# Convert categorical variables to factors
data$Gender <- as.factor(data$Gender)
data$Age_Group <- as.factor(data$Age_Group)
data$APR_Risk_of_Mortality <- as.factor(data$APR_Risk_of_Mortality)
data$Type_of_Admission <- as.factor(data$Type_of_Admission)
data$Emergency_Department_Indicator <- as.factor(data$Emergency_Department_Indicator)

# Create dummy variables for clustering
data_for_dummy <- data %>%
  dplyr::select(Length_of_Stay, Total_Charges, Total_Costs, APR_Severity_of_Illness_Code,
                Gender, Age_Group, APR_Risk_of_Mortality, Type_of_Admission, 
                Emergency_Department_Indicator)

data_numeric <- model.matrix(~ . - 1, data = data_for_dummy)
data_scaled <- scale(data_numeric)

# Examine the structure original data
str(data_subset)

#First few rows of original data
head(data_subset)

#Summary statistics of data
summary(data_subset)

#Matrix structure
dim(data_numeric)  
head(data_numeric)

# Column names in the dummy matrix
colnames(data_numeric)

# To check for the distribution of values in the scaled data
summary(data_scaled)

# To check the levels categorical variables
levels(data$Gender)
levels(data$Age_Group)
levels(data$Type_of_Admission)

# Cross-tabulation of categorical variables
table(data$Gender, data$Age_Group)

# TRAIN-TEST SPLIT

split_data <- function(data, train_prop = 0.8) {
  set.seed(530)
  train_idx <- createDataPartition(data$Age_Group, p = train_prop, list = FALSE)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  
  train_scaled <- data_scaled[train_idx, ]
  test_scaled <- data_scaled[-train_idx, ]
  
  return(list(train = train_data, test = test_data,
              train_scaled = train_scaled, test_scaled = test_scaled))
}

# 80/20 split
split_80_20 <- split_data(data, 0.8)
train_80 <- split_80_20$train
test_20 <- split_80_20$test
train_scaled_80 <- split_80_20$train_scaled
test_scaled_20 <- split_80_20$test_scaled

# 50/50 split
split_50_50 <- split_data(data, 0.5)
train_50 <- split_50_50$train
test_50 <- split_50_50$test
train_scaled_50 <- split_50_50$train_scaled
test_scaled_50 <- split_50_50$test_scaled

# K-MEANS CLUSTERING

# Function to find optimal k using elbow method
elbow_plot <- function(data, max_k = 10) {
  results <- list()
  tot.withinss <- numeric(max_k - 1)
  
  for(i in 2:max_k) {
    kmeans_result <- kmeans(data, centers = i, iter.max = 50, nstart = 25)
    results[[i-1]] <- kmeans_result
    tot.withinss[i-1] <- kmeans_result$tot.withinss
  }
  
  plot(2:max_k, tot.withinss, type = "b", pch = 19, 
       xlab = "Number of Clusters", 
       ylab = "Total Within-Cluster Sum of Squares",
       main = "Elbow Method for Optimal k")
  
  return(results)
}

# Function for silhouette analysis
silhouette_analysis <- function(data, cluster_results) {
  d <- dist(data)
  sils <- list()
  avg.sil <- numeric(length(cluster_results))
  
  for(i in 1:length(cluster_results)) {
    output <- silhouette(cluster_results[[i]]$cluster, d)
    sils[[i]] <- output
    avg.sil[i] <- mean(output[,3])
  }
  
  plot(2:(length(cluster_results)+1), avg.sil, type = "b", pch = 19, 
       xlab = "Number of Clusters", 
       ylab = "Average Silhouette Score",
       main = "Silhouette Analysis")
  
  return(list(sils = sils, avg.sil = avg.sil))
}

# Apply K-means to both splits
kmeans_results_80 <- elbow_plot(train_scaled_80)
sil_results_80 <- silhouette_analysis(train_scaled_80, kmeans_results_80)

kmeans_results_50 <- elbow_plot(train_scaled_50)
sil_results_50 <- silhouette_analysis(train_scaled_50, kmeans_results_50)

# Select optimal number of clusters (based on elbow and silhouette)
optimal_k <- 5  # Adjust based on plots
set.seed(530)
kmeans_80 <- kmeans(train_scaled_80, centers = optimal_k, iter.max = 50, nstart = 25)
kmeans_50 <- kmeans(train_scaled_50, centers = optimal_k, iter.max = 50, nstart = 25)

# Visualize K-means clusters
train_80$cluster_kmeans <- as.factor(kmeans_80$cluster)
fviz_cluster(kmeans_80, data = train_scaled_80, 
             palette = "Set2", 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw(),
             main = "K-means Clustering (80/20 split)")

# DBSCAN CLUSTERING

# Function to find optimal eps using k-distance plot
find_eps <- function(data, k = 4) {
  kNNdistplot(data, k = k)
  abline(h = 0.4, lty = 2)  # Adjust based on the knee in the plot
}

# Apply DBSCAN to both splits
find_eps(train_scaled_80)
dbscan_80 <- dbscan(train_scaled_80, eps = 3.5, minPts = 5)  # Adjust eps based on plot
train_80$cluster_dbscan <- as.factor(dbscan_80$cluster)

find_eps(train_scaled_50)
dbscan_50 <- dbscan(train_scaled_50, eps = 3.5, minPts = 5)
train_50$cluster_dbscan <- as.factor(dbscan_50$cluster)

# Visualize DBSCAN clusters
library(ggplot2)
library(factoextra)

# Perform PCA for visualization
pca_80 <- prcomp(train_scaled_80)
pca_data_80 <- data.frame(pca_80$x[,1:2])
pca_data_80$cluster <- as.factor(dbscan_80$cluster)

ggplot(pca_data_80, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "DBSCAN Clustering (80/20 split)", 
       x = "First Principal Component", 
       y = "Second Principal Component")

# EVALUATION METRICS

# Silhouette coefficient for K-means
sil_kmeans_80 <- silhouette(kmeans_80$cluster, dist(train_scaled_80))
avg_sil_kmeans_80 <- mean(sil_kmeans_80[, 3])

sil_kmeans_50 <- silhouette(kmeans_50$cluster, dist(train_scaled_50))
avg_sil_kmeans_50 <- mean(sil_kmeans_50[, 3])

# Silhouette coefficient for DBSCAN
valid_clusters_80 <- dbscan_80$cluster[dbscan_80$cluster != 0]
valid_idx_80 <- which(dbscan_80$cluster != 0)
if (length(unique(valid_clusters_80)) > 1) {
  sil_dbscan_80 <- silhouette(valid_clusters_80, dist(train_scaled_80[valid_idx_80,]))
  avg_sil_dbscan_80 <- mean(sil_dbscan_80[, 3])
} else {
  avg_sil_dbscan_80 <- NA
}

valid_clusters_50 <- dbscan_50$cluster[dbscan_50$cluster != 0]
valid_idx_50 <- which(dbscan_50$cluster != 0)
if (length(unique(valid_clusters_50)) > 1) {
  sil_dbscan_50 <- silhouette(valid_clusters_50, dist(train_scaled_50[valid_idx_50,]))
  avg_sil_dbscan_50 <- mean(sil_dbscan_50[, 3])
} else {
  avg_sil_dbscan_50 <- NA
}

# Davies-Bouldin Index
dbi_kmeans_80 <- index.DB(train_scaled_80, kmeans_80$cluster)$DB
dbi_kmeans_50 <- index.DB(train_scaled_50, kmeans_50$cluster)$DB

# Davies-Bouldin for DBSCAN (excluding noise points)
if (sum(dbscan_80$cluster != 0) > 1) {
  dbi_dbscan_80 <- index.DB(train_scaled_80[dbscan_80$cluster != 0,], 
                            dbscan_80$cluster[dbscan_80$cluster != 0])$DB
} else {
  dbi_dbscan_80 <- NA
}

if (sum(dbscan_50$cluster != 0) > 1) {
  dbi_dbscan_50 <- index.DB(train_scaled_50[dbscan_50$cluster != 0,], 
                            dbscan_50$cluster[dbscan_50$cluster != 0])$DB
} else {
  dbi_dbscan_50 <- NA
}

# Print evaluation metrics
cat("\n================================\n")
cat("CLUSTERING EVALUATION METRICS\n")
cat("================================\n\n")
cat("K-means (80/20):\n")
cat("  - Avg Silhouette: ", avg_sil_kmeans_80, "\n")
cat("  - Davies-Bouldin: ", dbi_kmeans_80, "\n")
cat("  - Total Within SS: ", kmeans_80$tot.withinss, "\n\n")

cat("K-means (50/50):\n")
cat("  - Avg Silhouette: ", avg_sil_kmeans_50, "\n")
cat("  - Davies-Bouldin: ", dbi_kmeans_50, "\n")
cat("  - Total Within SS: ", kmeans_50$tot.withinss, "\n\n")

cat("DBSCAN (80/20):\n")
cat("  - Avg Silhouette: ", avg_sil_dbscan_80, "\n")
cat("  - Davies-Bouldin: ", dbi_dbscan_80, "\n")
cat("  - Number of clusters: ", length(unique(dbscan_80$cluster)) - 1, "\n")
cat("  - Noise points: ", sum(dbscan_80$cluster == 0), "\n\n")

cat("DBSCAN (50/50):\n")
cat("  - Avg Silhouette: ", avg_sil_dbscan_50, "\n")
cat("  - Davies-Bouldin: ", dbi_dbscan_50, "\n")
cat("  - Number of clusters: ", length(unique(dbscan_50$cluster)) - 1, "\n")
cat("  - Noise points: ", sum(dbscan_50$cluster == 0), "\n\n")

# XAI: 


# CLUSTER INTERPRETATION

# 1. Cluster Centers Analysis
# For K-means, we can directly analyze the cluster centers
cluster_centers_80 <- kmeans_80$centers

# Convert back to original scale for interpretability
cluster_centers_original <- sweep(cluster_centers_80, 2, attr(data_scaled, "scaled:scale"), "*")
cluster_centers_original <- sweep(cluster_centers_original, 2, attr(data_scaled, "scaled:center"), "+")

# Create a heatmap of cluster centers
library(pheatmap)
pheatmap(cluster_centers_original[, 1:4],  # Just the numeric features
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         scale = "column",
         main = "Cluster Centers Heatmap",
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50))

# 2. Feature Importance for Clustering
# Create a function to calculate feature importance based on variance
cluster_feature_importance <- function(data, clusters) {
  importance <- sapply(1:ncol(data), function(i) {
    # Calculate between-cluster variance / total variance
    between_var <- sum(tapply(data[,i], clusters, function(x) length(x) * (mean(x) - mean(data[,i]))^2))
    total_var <- sum((data[,i] - mean(data[,i]))^2)
    return(between_var / total_var)
  })
  
  imp_df <- data.frame(
    Feature = colnames(data),
    Importance = importance
  )
  
  return(imp_df[order(-imp_df$Importance), ])
}

# Calculate feature importance
feature_imp_kmeans <- cluster_feature_importance(train_scaled_80, kmeans_80$cluster)

# Visualize feature importance
ggplot(feature_imp_kmeans[1:10,], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Features Contributing to Clustering",
       x = "Features", y = "Importance (Between-cluster variance ratio)")

# 3. Cluster Profiling
# Create profiles for each cluster
cluster_profiles <- train_80 %>%
  mutate(cluster = kmeans_80$cluster) %>%
  group_by(cluster) %>%
  summarise(
    avg_length_of_stay = mean(Length_of_Stay, na.rm = TRUE),
    avg_total_charges = mean(Total_Charges, na.rm = TRUE),
    avg_total_costs = mean(Total_Costs, na.rm = TRUE),
    avg_severity = mean(APR_Severity_of_Illness_Code, na.rm = TRUE),
    count = n(),
    .groups = 'drop'
  )

print(cluster_profiles)

# 4. Individual Instance Explanation
# For individual instances, show distance to each cluster center
explain_instance <- function(instance_idx, data, kmeans_result) {
  instance <- data[instance_idx, ]
  distances <- apply(kmeans_result$centers, 1, function(center) {
    sqrt(sum((instance - center)^2))
  })
  
  cluster_assignment <- kmeans_result$cluster[instance_idx]
  
  result <- data.frame(
    Cluster = 1:length(distances),
    Distance = distances,
    Assigned = ifelse(1:length(distances) == cluster_assignment, "Yes", "No")
  )
  
  return(result)
}

# Explain a few instances
for(i in sample_idx) {
  cat("\nInstance", i, "explanation:\n")
  print(explain_instance(i, train_scaled_80, kmeans_80))
}

# 5. Visualization of Feature Contributions
# Create a radar chart for each cluster
library(fmsb)

# Prepare data for radar chart (use top 8 features)
top_features <- head(feature_imp_kmeans$Feature, 8)
radar_data <- as.data.frame(cluster_centers_80[, top_features])

# Normalize for radar chart (0-1 scale)
radar_data <- apply(radar_data, 2, function(x) (x - min(x)) / (max(x) - min(x)))
radar_data <- rbind(rep(1, ncol(radar_data)), rep(0, ncol(radar_data)), radar_data)

# Create radar charts
par(mfrow = c(2, 3))
for(i in 1:nrow(kmeans_80$centers)) {
  radarchart(radar_data[c(1, 2, i+2), ], 
             axistype = 1,
             pcol = rainbow(5)[i], 
             pfcol = scales::alpha(rainbow(5)[i], 0.3),
             plwd = 2, 
             cglcol = "grey", 
             cglty = 1, 
             axislabcol = "grey",
             caxislabels = seq(0, 1, 0.25), 
             cglwd = 0.8,
             vlcex = 0.8,
             title = paste("Cluster", i))
}

# 6. SHAP-like values for clustering
# Calculate contribution of each feature to cluster assignment
calculate_cluster_contributions <- function(instance, center, feature_names) {
  difference <- instance - center
  contributions <- difference^2  # Squared difference contribution
  
  result <- data.frame(
    Feature = feature_names,
    Contribution = contributions,
    Direction = ifelse(difference > 0, "Above", "Below")
  )
  
  return(result[order(-result$Contribution), ])
}

# Example for one instance
instance_idx <- sample_idx[1]
assigned_cluster <- kmeans_80$cluster[instance_idx]
instance_contributions <- calculate_cluster_contributions(
  train_scaled_80[instance_idx, ],
  kmeans_80$centers[assigned_cluster, ],
  colnames(train_scaled_80)
)

# Visualize contributions
ggplot(head(instance_contributions, 10), 
       aes(x = reorder(Feature, Contribution), y = Contribution, fill = Direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = paste("Feature Contributions to Cluster Assignment (Instance", instance_idx, ")"),
       x = "Features", y = "Contribution to Distance")


# CROSS-VALIDATION FOR CLUSTERING STABILITY


cv_results <- function(data, k = 5, n_folds = 5) {
  set.seed(530)
  folds <- createFolds(1:nrow(data), k = n_folds)
  stability_scores <- numeric(n_folds)
  
  for(i in 1:n_folds) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]
    
    # Fit kmeans on training data
    kmeans_fit <- kmeans(train_data, centers = k, nstart = 25)
    
    # Predict on test data (assign to nearest centroid)
    test_pred <- apply(test_data, 1, function(x) {
      which.min(rowSums((t(kmeans_fit$centers) - x)^2))
    })
    
    # Calculate silhouette score for stability
    if (length(unique(test_pred)) > 1) {
      sil <- silhouette(test_pred, dist(test_data))
      stability_scores[i] <- mean(sil[, 3])
    } else {
      stability_scores[i] <- NA
    }
  }
  
  return(mean(stability_scores, na.rm = TRUE))
}

# Apply 5-fold CV
cv_score_80 <- cv_results(train_scaled_80, k = optimal_k)
cv_score_50 <- cv_results(train_scaled_50, k = optimal_k)

cat("\n================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("================================\n\n")
cat("1- CV Score: ", cv_score_80, "\n")
cat("2 - CV Score: ", cv_score_50, "\n\n")

# SUMMARY VISUALIZATION

# Create summary dataframe
results_df <- data.frame(
  Method = rep(c("K-means", "DBSCAN"), each = 2),
  Split = rep(c("80/20", "50/50"), 2),
  Silhouette = c(avg_sil_kmeans_80, avg_sil_kmeans_50, 
                 avg_sil_dbscan_80, avg_sil_dbscan_50),
  Davies_Bouldin = c(dbi_kmeans_80, dbi_kmeans_50, 
                     dbi_dbscan_80, dbi_dbscan_50)
)

# Remove NAs for plotting
results_df <- results_df[complete.cases(results_df), ]

# Reshape for plotting
results_long <- melt(results_df, id.vars = c("Method", "Split"))

# Create comparison plot
ggplot(results_long, aes(x = Method, y = value, fill = Split)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y") +
  theme_minimal() +
  labs(title = "Clustering Performance Comparison", y = "Score") +
  scale_fill_brewer(palette = "Set1")


# CLUSTER PROFILE ANALYSIS

# Analyze clusters from K-means 80/20 split
train_80$cluster <- kmeans_80$cluster

cluster_profiles <- train_80 %>%
  group_by(cluster) %>%
  summarise(
    avg_length_of_stay = mean(Length_of_Stay, na.rm = TRUE),
    avg_total_charges = mean(Total_Charges, na.rm = TRUE),
    avg_total_costs = mean(Total_Costs, na.rm = TRUE),
    avg_severity = mean(APR_Severity_of_Illness_Code, na.rm = TRUE),
    count = n(),
    .groups = 'drop'
  ) %>%
  arrange(cluster)

print(cluster_profiles)

# Create profile visualization
ggplot(cluster_profiles, aes(x = factor(cluster), y = avg_length_of_stay)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(title = "Average Length of Stay by Cluster", 
       x = "Cluster", y = "Avg Length of Stay")

# Age group distribution by cluster
age_cluster_dist <- train_80 %>%
  group_by(cluster, Age_Group) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(cluster) %>%
  mutate(percentage = count / sum(count) * 100)

ggplot(age_cluster_dist, aes(x = factor(cluster), y = percentage, fill = Age_Group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Age Group Distribution by Cluster",
       x = "Cluster", y = "Percentage")

# SAVE RESULTS

# Save cluster assignments for both methods
results_to_save <- data.frame(
  Length_of_Stay = train_80$Length_of_Stay,
  Total_Charges = train_80$Total_Charges,
  Total_Costs = train_80$Total_Costs,
  Age_Group = train_80$Age_Group,
  Gender = train_80$Gender,
  APR_Severity = train_80$APR_Severity_of_Illness_Code,
  KMeans_Cluster = kmeans_80$cluster,
  DBSCAN_Cluster = dbscan_80$cluster
)

write.csv(results_to_save, "clustering_results.csv", row.names = FALSE)

cat("\n================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================\n")
cat("Results saved to 'clustering_results.csv'\n")
cat("All visualizations have been generated\n")


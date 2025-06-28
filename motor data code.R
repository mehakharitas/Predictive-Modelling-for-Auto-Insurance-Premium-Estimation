install.packages(c("tidyverse", "caret", "DataExplorer"))
install.packages(c("ranger","xgboost"))
install.packages("doParallel")

library(tidyverse)   # For data manipulation and visualization
library(caret)       # For modeling
library(DataExplorer) # For quick EDA
motor_data <- read.csv(choose.files())

# View first few rows
head(motor_data)

# Structure of dataset
str(motor_data)

# Summary statistics
summary(motor_data)

# Check missing values
colSums(is.na(motor_data))

# removing rows where premium is NA
motor_data <- motor_data[!is.na(motor_data$PREMIUM), ]

# dropping all the columns which are not needed
drop_cols <- c("INSR_BEGIN", "INSR_END", "EFFECTIVE_YR", "OBJECT_ID")
motor_data <- motor_data[ , !(names(motor_data) %in% drop_cols)]

# assigning factor values
motor_data$MAKE <- as.factor(motor_data$MAKE)
motor_data$USAGE <- as.factor(motor_data$USAGE)
motor_data$TYPE_VEHICLE <- as.factor(motor_data$TYPE_VEHICLE)
motor_data$SEX <- as.factor(motor_data$SEX)

# assuming NA in claims paid is no claims made
motor_data$CLAIM_PAID <- replace(motor_data$CLAIM_PAID, is.na(motor_data$CLAIM_PAID), 0)

# filling median values in carry capacity as 23% of the data has NA in it.
sum(is.na(motor_data$CARRYING_CAPACITY))
motor_data$CARRYING_CAPACITY[is.na(motor_data$CARRYING_CAPACITY)] <- 
  median(motor_data$CARRYING_CAPACITY, na.rm = TRUE)

# assigning 0 wherever the data is missing in ccm_ton
sum(is.na(motor_data$CCM_TON))
motor_data$CCM_TON <- replace(motor_data$CCM_TON, is.na(motor_data$CCM_TON), 0)


#EDA

hist(motor_data$PREMIUM, 
     breaks = 50, 
     col = "red", 
     main = "Distribution of Premium", 
     xlab = "Premium")

hist(log1p(motor_data$PREMIUM), 
     breaks = 100, 
     col = "skyblue", 
     main = "Log-Transformed Distribution of Premium", 
     xlab = "log(Premium + 1)")

boxplot(log1p(motor_data$PREMIUM), main = "Boxplot of log Premium", col = "orange")

view(motor_data)

cor(log1p(motor_data$PREMIUM), motor_data$CLAIM_PAID, use = "complete.obs")

boxplot(log1p(PREMIUM) ~ TYPE_VEHICLE, data = motor_data,
        las = 2, col = "lightblue",
        main = "Premium by Vehicle Type",
        ylab = "log(Premium + 1)")

plot(log1p(motor_data$PREMIUM) ~ motor_data$CCM_TON,
     xlab = "Cubic Capacity (CCM_TON)",
     ylab = "log(Premium + 1)",
     main = "Cubic Capacity vs Premium",
     pch = 20, col = "blue")

library(corrplot)
numeric_vars <- motor_data[sapply(motor_data, is.numeric)]
cor_matrix <- cor(na.omit(numeric_vars))
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)

# relation between premium and key variable
plot(motor_data$INSURED_VALUE, log1p(motor_data$PREMIUM), 
     main = "log(1+Premium) vs Insured Value", xlab = "Insured Value", ylab = "log transformed Premium")

plot(motor_data$CLAIM_PAID, log1p(motor_data$PREMIUM), 
     main = "log(1+Premium) vs Claim Paid", xlab = "Claim Paid", ylab = "log transformed Premium")

# categorial impact on premium

library(ggplot2)
ggplot(motor_data, aes(x = MAKE, y = log1p(PREMIUM))) + 
  geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Log(Premium) by Vehicle Make", x = "Make", y = "log(Premium + 1)")

ggplot(motor_data, aes(x = as.factor(SEX), y = log1p(PREMIUM))) + 
  geom_boxplot() + 
  labs(title = "Log(Premium) by Sex", 
       x = "Sex (0 = Male, 1 = Female, 2 = Unknown)", 
       y = "log(Premium + 1)")

# for better intepretibility of vehicle make graph
library(dplyr)

# Step 1: Find top 10 most frequent vehicle makes
top_makes <- motor_data %>%
  count(MAKE, sort = TRUE) %>%
  top_n(15, n) %>%
  pull(MAKE)

# Step 2: Filter the dataset to include only top makes
filtered_motor_data <- motor_data %>%
  filter(MAKE %in% top_makes)

# Step 3: Plot boxplot of log(PREMIUM) by MAKE
ggplot(filtered_motor_data, aes(x = reorder(MAKE, PREMIUM, FUN = median), y = log1p(PREMIUM))) + 
  geom_boxplot(fill = "lightblue") + 
  coord_flip() +
  labs(title = "Log(Premium) by Top 15 Vehicle Makes",
       x = "Vehicle Make",
       y = "log(Premium + 1)") +
  theme_minimal()

cor_matrix

motor_data$VEHICLE_AGE <- 2025 - motor_data$PROD_YEAR
motor_data$CLAIM_MADE <- ifelse(motor_data$CLAIM_PAID > 0, 1, 0)
motor_data$CLAIM_MADE <- as.factor(motor_data$CLAIM_MADE)

# Group rare MAKEs (appearing < 500 times) into "Other"
make_counts <- table(motor_data$MAKE)
common_makes <- names(make_counts[make_counts >= 500])
motor_data$MAKE_GROUPED <- ifelse(motor_data$MAKE %in% common_makes, as.character(motor_data$MAKE), "Other")
motor_data$MAKE_GROUPED <- as.factor(motor_data$MAKE_GROUPED)

motor_data$log_PREMIUM <- log1p(motor_data$PREMIUM)

dummies <- dummyVars(log_PREMIUM ~ TYPE_VEHICLE + USAGE + SEX + MAKE_GROUPED + CLAIM_MADE, data = motor_data)
categorical_data <- data.frame(predict(dummies, newdata = motor_data))

categorical_data

# 4. Select useful numeric features
numeric_vars <- motor_data[, c("INSURED_VALUE", "SEATS_NUM", "CARRYING_CAPACITY",
                               "CCM_TON", "VEHICLE_AGE", "CLAIM_PAID", "CLAIM_MADE")]

# 5. One-hot encode categorical variables
categorical_vars <- motor_data[, c("SEX", "USAGE", "TYPE_VEHICLE")]
categorical_data <- model.matrix(~ . - 1, data = categorical_vars)  # drop intercept

# 6. Combine all for modeling
model_finaldata <- cbind(log_PREMIUM = motor_data$log_PREMIUM, numeric_vars, categorical_data)

str(model_finaldata)

# --- Re-confirming final data preparation steps for 'motor_data' ---
# It's safest to remove any row with NAs at this point, before final model_finaldata creation
motor_data_clean <- na.omit(motor_data)

# Convert INSR_TYPE to factor if it represents categories
motor_data_clean$INSR_TYPE <- as.factor(motor_data_clean$INSR_TYPE)

# Drop original PREMIUM and PROD_YEAR as they are now replaced by log_PREMIUM and VEHICLE_AGE
motor_data_clean <- motor_data_clean %>%
  select(-PREMIUM, -PROD_YEAR, -MAKE) # Drop original MAKE too, as MAKE_GROUPED is used

# 1. Select useful numeric features (after dropping original PREMIUM)
numeric_vars_final <- motor_data_clean[, c("INSURED_VALUE", "SEATS_NUM", "CARRYING_CAPACITY",
                                           "CCM_TON", "VEHICLE_AGE", "CLAIM_PAID")] # CLAIM_MADE is factor, so not here

# 2. Select categorical features to one-hot encode
# Now including INSR_TYPE and MAKE_GROUPED
categorical_vars_to_encode <- motor_data_clean[, c("SEX", "USAGE", "TYPE_VEHICLE", "MAKE_GROUPED", "INSR_TYPE", "CLAIM_MADE")]

# 3. One-hot encode these categorical variables
# Using `dummyVars` to create dummy variables. `fullRank = TRUE` avoids perfect multicollinearity if you use linear models.
dummies_model <- dummyVars(~ ., data = categorical_vars_to_encode, fullRank = TRUE)
categorical_data_encoded <- data.frame(predict(dummies_model, newdata = categorical_vars_to_encode))

# 4. Combine all for modeling
# Ensure log_PREMIUM is the first column as target, then numeric, then encoded categoricals
model_finaldata <- cbind(log_PREMIUM = motor_data_clean$log_PREMIUM,
                         numeric_vars_final,
                         categorical_data_encoded)

# Check the structure of the final dataset before splitting
str(model_finaldata)
dim(model_finaldata)

# --- Phase 4: Model Development & Evaluation (Revised for faster execution) ---

# IMPORTANT: If you closed RStudio or force-quit, you MUST re-run all the data preparation steps
# from the beginning of your script to make sure 'model_finaldata' is correctly loaded and processed.
# That includes loading your original motor_data, handling NAs, creating new features,
# grouping makes, and creating model_finaldata and any dummy variables.

# Ensure all necessary packages are loaded
library(ranger)   # Explicitly load ranger as we're using its method
library(xgboost)  # Explicitly load xgboost for that method
library(doParallel) # For parallel processing

# Assuming 'model_finaldata' is ready from your prior steps.

# ... (Your data loading and preparation, creating model_finaldata) ...

# 1. Data Splitting (Training and Testing)
set.seed(123)
trainIndex <- createDataPartition(model_finaldata$log_PREMIUM, p = .8,
                                  list = FALSE,
                                  times = 1)
train_data <- model_finaldata[trainIndex, ]
test_data  <- model_finaldata[-trainIndex, ]

cat("Dimensions of Training Data:", dim(train_data), "\n")
cat("Dimensions of Testing Data:", dim(test_data), "\n")


# IMPORTANT: Reduce the number of cores to avoid memory issues
# Start with a very conservative number like 2, or even 1 if issues persist.
num_cores <- 1 # <<<--- CHANGE THIS LINE! Try 2 first, if error persists, try 1.
if (num_cores < 1) num_cores <- 1 # Ensure at least 1 core is used
registerDoParallel(num_cores)
cat(paste0("Registered ", num_cores, " cores for parallel processing.\n"))

# 2. Define Training Control for Cross-Validation
fitControl <- trainControl(method = "cv",
                           number = 3,
                           verboseIter = TRUE,
                           allowParallel = TRUE) # This should now use the registered cores

# Run garbage collection to free up memory before model training
gc(verbose = TRUE)


# 3. Model Training (Now try with the revised settings)

# --- Model 1: Linear Regression (LM) ---
cat("\n--- Training Linear Regression Model ---\n")
set.seed(123)
lm_model <- train(log_PREMIUM ~ .,
                  data = train_data,
                  method = "lm",
                  trControl = fitControl)
print(lm_model)
summary(lm_model)
plot(lm_model)

# Afterlm_model has finished training successfully
saveRDS(lm_model, "lm_model.rds")
cat("Linear Regression model saved as lm_model.rds\n")

# Load the saved Linear Regression model
lm_model_loaded <- readRDS("lm_model.rds")
cat("Linear Regression model loaded from lm_model.rds\n")

# --- Model 2: Random Forest (RF) with explicit tuneGrid ---
cat("\n--- Training Random Forest Model (ranger) ---\n")
set.seed(123)
rf_grid_fast <- expand.grid(
  mtry = c(10, 25, 40),
  splitrule = "variance",
  min.node.size = 5
)

rf_model <- train(log_PREMIUM ~ .,
                  data = train_data,
                  method = "ranger",
                  trControl = fitControl,
                  tuneGrid = rf_grid_fast,
                  num.trees = 250,
                  importance = "impurity")

print(rf_model)
plot(rf_model)
# After rf_model has finished training successfully
saveRDS(rf_model, "rf_model.rds")
cat("Random Forest model saved as rf_model.rds\n")

# Load the saved Random Forest model
rf_model_loaded <- readRDS("rf_model.rds")
cat("Random Forest model loaded from rf_model.rds\n")

print(rf_model_loaded)
plot(rf_model_loaded)


# --- Model 3: XGBoost (eXtreme Gradient Boosting) with explicit tuneGrid ---
cat("\n--- Training XGBoost Model ---\n")
set.seed(123)
xgb_grid_fast <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 5),
  eta = c(0.1),
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.7
)

xgb_model <- train(log_PREMIUM ~ .,
                   data = train_data,
                   method = "xgbTree",
                   trControl = fitControl,
                   tuneGrid = xgb_grid_fast)

print(xgb_model)
plot(xgb_model)

# After xgb_model has finished training successfully
saveRDS(xgb_model, "xgb_model.rds")
cat("XGBoost model saved as xgb_model.rds\n")

# Load the saved XGBoost model
xgb_model_loaded <- readRDS("xgb_model.rds")
cat("XGBoost model loaded from xgb_model.rds\n")


# IMPORTANT: Stop the parallel cluster when you are done
stopImplicitCluster()






# --- 1. Predictions on the Test Data ---
cat("\n--- Making Predictions on Test Data ---\n")

# Predict with Linear Model (lm_model)
# If you closed R and reloaded, ensure lm_model is loaded:
# lm_model <- readRDS("lm_model.rds")
predictions_lm <- predict(lm_model, newdata = test_data)
cat("Predictions made with Linear Model.\n")

# Predict with Random Forest Model (rf_model)
# If you closed R and reloaded, ensure rf_model is loaded:
# rf_model <- readRDS("rf_model.rds")
predictions_rf <- predict(rf_model_loaded, newdata = test_data)
cat("Predictions made with Random Forest Model.\n")

# Predict with XGBoost Model (xgb_model)
# If you closed R and reloaded, ensure xgb_model is loaded:
# xgb_model <- readRDS("xgb_model.rds")
predictions_xgb <- predict(xgb_model, newdata = test_data)
cat("Predictions made with XGBoost Model.\n")

# --- 2. Model Performance Evaluation on Test Data ---
cat("\n--- Evaluating Model Performance on Test Data ---\n")

# Get the actual log_PREMIUM values from the test data
actual_log_premium <- test_data$log_PREMIUM

# --- Function to calculate performance metrics ---
calculate_metrics <- function(actual, predicted, model_name) {
  rmse_val <- sqrt(mean((actual - predicted)^2))
  mae_val <- mean(abs(actual - predicted))
  # R-squared calculation
  ss_total <- sum((actual - mean(actual))^2)
  ss_residual <- sum((actual - predicted)^2)
  r_squared_val <- 1 - (ss_residual / ss_total)
  
  cat(paste0("\n--- Metrics for ", model_name, " ---\n"))
  cat(paste0("RMSE:     ", round(rmse_val, 5), "\n"))
  cat(paste0("MAE:      ", round(mae_val, 5), "\n"))
  cat(paste0("R-squared: ", round(r_squared_val, 5), "\n"))
  
  # Return as a data frame for easier comparison later
  return(data.frame(
    Model = model_name,
    RMSE = rmse_val,
    MAE = mae_val,
    Rsquared = r_squared_val
  ))
}

# Calculate metrics for each model
metrics_lm <- calculate_metrics(actual_log_premium, predictions_lm, "Linear Regression")
metrics_rf <- calculate_metrics(actual_log_premium, predictions_rf, "Random Forest")
metrics_xgb <- calculate_metrics(actual_log_premium, predictions_xgb, "XGBoost")

# --- 3. Model Selection (Comparison) ---
cat("\n--- Model Comparison ---\n")
all_metrics <- rbind(metrics_lm, metrics_rf, metrics_xgb)
print(all_metrics)

# You can also use compare_models from caret, but it's more for resampling results.
# For test set comparison, simply looking at the table is often clearest.
# Example: best_model_name <- all_metrics[which.min(all_metrics$RMSE), "Model"]
# cat(paste0("\nThe best performing model on the test set (by RMSE) is: ", best_model_name, "\n"))






# Importance Analysis for the Best Model (Random Forest)
cat("\n--- Performing Variable Importance Analysis for Random Forest Model ---\n")

# Calculate variable importance
# If you restarted R and haven't loaded rf_model, uncomment the line below:
# rf_model <- readRDS("rf_model.rds")

importance_rf <- varImp(rf_model_loaded, scale = FALSE) # scale=FALSE gives raw importance scores

# Print the importance values
print(importance_rf)

# Plot the variable importance (will appear in RStudio Plots pane)
plot(importance_rf, main = "Variable Importance for Random Forest Model")

# To get a more readable list of top N variables (e.g., top 20)
cat("\n--- Top 20 Most Important Variables (Random Forest) ---\n")
# Extracting importance values into a data frame and sorting
importance_df <- as.data.frame(importance_rf$importance)
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$Overall), ] # Sort by 'Overall' importance

# Print the top 20
print(head(importance_df, 20))



# --- Generate a Better Variable Importance Plot (Top 20) ---
cat("\n--- Generating Improved Variable Importance Plot ---\n")

# Select the top 20 most important variables for plotting
top_n_variables <- 20
importance_plot_df <- head(importance_df, top_n_variables)

# Create the ggplot2 bar plot
ggplot(importance_plot_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip coordinates to make variable names readable
  labs(title = paste0("Top ", top_n_variables, " Variable Importance for Random Forest Model"),
       x = "Variable",
       y = "Importance (Overall)") +
  theme_minimal() + # Use a clean theme
  theme(axis.text.y = element_text(size = 10), # Adjust font size for readability
        plot.title = element_text(hjust = 0.5)) # Center the plot title

# You can save this plot to a file if you like
cat(paste0("An improved Variable Importance plot (Top ", top_n_variables, " variables) has been generated in the Plots pane.\n"))

















# rf_model <- readRDS("rf_model.rds")
# predictions_rf <- predict(rf_model, newdata = test_data)
# actual_log_premium <- test_data$log_PREMIUM

# Load ggplot2 if not already loaded
# library(ggplot2)

cat("\n--- Generating Observed vs. Predicted Plot for Random Forest Model ---\n")

# Create a data frame for plotting
plot_data_rf <- data.frame(
  Actual = actual_log_premium,
  Predicted = predictions_rf
)

ggplot(plot_data_rf, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkblue") + # Add points for actual vs. predicted
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) + # Add a 45-degree line
  labs(title = "Random Forest Model: Actual vs. Predicted Log(Premium) (Test Set)",
       x = "Actual Log(Premium)",
       y = "Predicted Log(Premium)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed(ratio = 1) # Ensure aspect ratio is 1:1


cat("Observed vs. Predicted plot generated in the Plots pane.\n")

# --- Residuals Plot ---
# insightful for diagnosing model issues, though "Actual vs. Predicted" is generally more intuitive for a general audience.
cat("\n--- Generating Residuals Plot for Random Forest Model (Optional) ---\n")
residuals_rf <- actual_log_premium - predictions_rf
plot_data_residuals_rf <- data.frame(
  Predicted = predictions_rf,
  Residuals = residuals_rf
)

ggplot(plot_data_residuals_rf, aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) + # Reference line at 0
  labs(title = "Random Forest Model: Residuals vs. Predicted (Test Set)",
       x = "Predicted Log(Premium)",
       y = "Residuals (Actual - Predicted)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

cat("Residuals plot generated in the Plots pane (optional).\n")

sink("model_outputs.txt")

import pandas as pd
import random

# Specify the file path
file_path = 'specialties_data/dermatology.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Convert fields that are supposed to be numeric to numeric
numeric_fields = ['Gold Signal', 'Silver Signal', 'No Signal', 'In-State School', 'Out-Of-State School', 'MD', 'DO', 'US IMG', 'Non US IMG']
df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce')

# For example, you can print the first few rows
print(df.head())

# Print descriptive statistics for the numeric fields as a df with the numeric_fields



############################
##### Monte Carlo Simulation
############################
# Set the random seed
random.seed(42)

# Initialize lists to store the sums
gold_signal_sums = []
silver_signal_sums = []

# Run the simulation 10000 times
for _ in range(10000):
    # Pick 28 random rows
    subset = df.sample(n=28)

    # Pick 3 random rows from the subset
    random_rows = subset.sample(n=3)

    # Calculate the sum of Gold Signal for the random rows
    gold_signal_sum = random_rows['Gold Signal'].sum()

    # Calculate the sum of Silver Signal for the other 25 rows
    silver_signal_sum = subset.loc[~subset.index.isin(random_rows.index), 'Silver Signal'].sum()

    # Append the sums to the lists
    gold_signal_sums.append(gold_signal_sum)
    silver_signal_sums.append(silver_signal_sum)

# Calculate the mean of the gold_signal_sums and silver_signal_sums
mean_gold_signal_sum = sum(gold_signal_sums) / len(gold_signal_sums)
mean_silver_signal_sum = sum(silver_signal_sums) / len(silver_signal_sums)


# Calculate the 95% confidence interval for the gold_signal_sums
gold_signal_sums.sort()
lower_bound = gold_signal_sums[249]
upper_bound = gold_signal_sums[9749]
silver_signal_sums.sort()
lower_bound_silver = silver_signal_sums[249]
upper_bound_silver = silver_signal_sums[9749]

# Print the 95% confidence interval with the mean sums
print(f'Gold Signal Sum: {mean_gold_signal_sum} (95% CI: {lower_bound}-{upper_bound})')
print(f'Silver Signal Sum: {mean_silver_signal_sum} (95% CI: {lower_bound_silver}-{upper_bound_silver})')




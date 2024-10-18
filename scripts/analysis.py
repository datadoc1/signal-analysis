import random
import itertools
import pandas as pd
import os

def simulation_tiered(df, gold_count, silver_count):
    # Set the random seed
    random.seed(42)

    # Initialize lists to store the sums
    gold_signal_sums = []
    silver_signal_sums = []

    # Run the simulation 10000 times
    for _ in range(10000):
        total_count = gold_count + silver_count
        
        # Pick random rows from the dataframe equal to the total count
        subset = df.sample(n=total_count)

        # Pick `gold_count` random rows from the subset for the gold signal
        random_rows = subset.sample(n=gold_count)

        # Calculate the sum of Gold Signal for the random rows
        gold_signal_sum = random_rows['Gold Signal'].sum()

        # Calculate the sum of Silver Signal for the remaining rows
        silver_signal_sum = subset.loc[~subset.index.isin(random_rows.index), 'Silver Signal'].sum()

        # Append the sums to the lists
        gold_signal_sums.append(gold_signal_sum)
        silver_signal_sums.append(silver_signal_sum)

    # Calculate the mean of the gold_signal_sums and silver_signal_sums
    mean_gold_signal_sum = sum(gold_signal_sums) / len(gold_signal_sums)
    mean_silver_signal_sum = sum(silver_signal_sums) / len(silver_signal_sums)

    # Calculate the 95% confidence interval for the gold_signal_sums
    gold_signal_sums.sort()
    lower_bound_gold = gold_signal_sums[249]
    upper_bound_gold = gold_signal_sums[9749]

    # Calculate the 95% confidence interval for the silver_signal_sums
    silver_signal_sums.sort()
    lower_bound_silver = silver_signal_sums[249]
    upper_bound_silver = silver_signal_sums[9749]

    # Save the results
    # Round the results to 2 decimal places
    mean_gold_signal_sum = round(mean_gold_signal_sum, 2)
    mean_silver_signal_sum = round(mean_silver_signal_sum, 2)
    lower_bound_gold = round(lower_bound_gold, 2)
    upper_bound_gold = round(upper_bound_gold, 2)
    lower_bound_silver = round(lower_bound_silver, 2)
    upper_bound_silver = round(upper_bound_silver, 2)

    # Prepare the results string
    results = (
        f'Simulation Results:\n'
        f'-------------------\n'
        f'We randomly picked {gold_count} Gold Signal rows and {silver_count} Silver Signal rows from the dataset.\n\n'
        f'This was repeated 10,000 times to calculate the average number of interviews one should expect if they were a completely average applicant with completely randomly selected signals.\n\n'
        f'Gold Signal Sum ({gold_count}): {mean_gold_signal_sum} (95% CI: {lower_bound_gold}-{upper_bound_gold})\n'
        f'Silver Signal Sum ({silver_count}): {mean_silver_signal_sum} (95% CI: {lower_bound_silver}-{upper_bound_silver})\n'
    )

    # Print the results
    print(results)
    return results

def simulation(df, signal_count):
    random.seed(42)
    signal_sums = []
    for _ in range(10000):
        # Pick random rows from the dataframe equal to the signal count
        subset = df.sample(n=signal_count)
        
        # Calculate the sum of Signal for the random rows
        signal_sum = subset['Signal'].sum()
        
        # Append the sum to the list
        signal_sums.append(signal_sum)

    # Calculate the mean of the signal_sums
    mean_signal_sum = sum(signal_sums) / len(signal_sums)

    # Calculate the 95% confidence interval for the signal_sums
    signal_sums.sort()
    lower_bound_signal = signal_sums[249]
    upper_bound_signal = signal_sums[9749]

    # Round the results to 2 decimal places
    mean_signal_sum = round(mean_signal_sum, 2)
    lower_bound_signal = round(lower_bound_signal, 2)
    upper_bound_signal = round(upper_bound_signal, 2)

    # Prepare the results string
    results = (
        f'Simulation Results:\n'
        f'-------------------\n'
        f'We randomly picked {signal_count} programs from the dataset to signal.\n\n'
        f'This was repeated 10,000 times to calculate the average number of interviews one should expect if they were a completely average applicant with completely randomly selected signals.\n\n'
        f'Signal Sum ({signal_count}): {mean_signal_sum} (95% CI: {lower_bound_signal}-{upper_bound_signal})\n'
    )

    # Print the results
    print(results)
    return results

def optimal_tiered(df, gold_count, silver_count, include_no_signal=False):
    # Select the top candidates based on Gold and Silver signals
    top_gold_candidates = df.nlargest(gold_count, 'Gold Signal')
    top_silver_candidates = df.nlargest(silver_count, 'Silver Signal')
    
    print(f"Top Gold Candidates:\n{top_gold_candidates}\n")
    print(f"Top Silver Candidates:\n{top_silver_candidates}\n")

    # Collect results
    results = "Optimal Signals:\n"
    results += "Gold:\n"
    
    # Output top 5 Gold candidates
    for _, row in top_gold_candidates.iterrows():
        results += f"{row['Program']}  ({row['Gold Signal']})\n"

    results += "Silver:\n"
    
    # Output top 10 Silver candidates
    for _, row in top_silver_candidates.iterrows():
        results += f"{row['Program']} ({row['Silver Signal']})\n"

    # Calculate and print the sums of Gold and Silver
    total_gold_sum = top_gold_candidates['Gold Signal'].sum()
    total_silver_sum = top_silver_candidates['Silver Signal'].sum()
    
    results += f"\nTotal Gold Sum: {total_gold_sum}\n"
    results += f"Total Silver Sum: {total_silver_sum}\n"

    # List all rows that appear in both Gold and Silver
    overlap = top_gold_candidates[top_gold_candidates.index.isin(top_silver_candidates.index)]
    results += "\nOverlap:\n"

    # Output the overlapping rows
    for _, row in overlap.iterrows():
        results += f"{row['Program']} ({row['Gold Signal']}, {row['Silver Signal']})\n"

    # Now, select the next 10 best Gold options
    next_best_gold = df.nlargest(gold_count + 10, 'Gold Signal').iloc[gold_count:]  # Skip top 5
    next_best_silver = df.nlargest(silver_count + 10, 'Silver Signal').iloc[silver_count:]  # Skip top 10

    results += "\nNext Best Options:\n"
    
    # Output next 10 Gold candidates with values and differences
    results += "Next Best Gold Options:\n"
    for _, row in next_best_gold.iterrows():
        difference = row['Gold Signal'] - row['Silver Signal']
        results += f"{row['Program']} ({row['Gold Signal']}), Difference: {difference}\n"

    # Output next 10 Silver candidates with values and differences
    results += "Next Best Silver Options:\n"
    for _, row in next_best_silver.iterrows():
        difference = row['Gold Signal'] - row['Silver Signal']
        results += f"{row['Program']} ({row['Silver Signal']}), Difference: {difference}\n"

    return results

def optimal(df, signal_count):
    # Select the top candidates based on Signal
    top_candidates = df.nlargest(signal_count, 'Signal')
    
    print(f"Top Candidates:\n{top_candidates}\n")

    # Collect results
    results = "Optimal Signals:\n"
    results += "Top Candidates:\n"
    
    # Output top candidates
    for _, row in top_candidates.iterrows():
        results += f"{row['Program']} ({row['Signal']})\n"

    # Calculate and print the sum of Signal
    total_signal_sum = top_candidates['Signal'].sum()
    results += f"\nTotal Interviews: {total_signal_sum}\n"
    return results

def geographic_bias(df):
    # Group by 'State' and compute the average and sum of 'In-State School' and 'Out-of-State School'
    grouped = df.groupby('State')[['In-State School', 'Out-Of-State School']].agg(['mean', 'sum']).reset_index()
    
    # Flatten the MultiIndex columns
    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]
    
    # Rename columns for easier access
    grouped.rename(columns={
        'In-State School mean': 'In-State School Mean',
        'Out-Of-State School mean': 'Out-Of-State School Mean',
        'In-State School sum': 'In-State School Sum',
        'Out-Of-State School sum': 'Out-Of-State School Sum'
    }, inplace=True)
    
    # Compute the difference between In-State and Out-Of-State means and sums
    grouped['Mean Difference'] = grouped['In-State School Mean'] - grouped['Out-Of-State School Mean']
    grouped['Sum Difference'] = grouped['In-State School Sum'] - grouped['Out-Of-State School Sum']
    
    # Round the results to 2 decimal places
    grouped = grouped.round(2)
    
    return grouped

# Read the specialties file
specialties = pd.read_csv('signal_2024_match.csv')

# Filter specialties that are tiered
specialties_tiered = specialties[specialties['Tiered'] == 'Yes']
specialties_not_tiered = specialties[specialties['Tiered'] == 'No']

# Iterate over the tiered specialties
for index, row in specialties_tiered.iterrows():
    value = row["Specialty"]
    df = pd.read_csv(f'data/{value}.csv')
    
    # Convert relevant columns to numeric
    numeric_fields = ['Gold Signal', 'Silver Signal', 'No Signal', 'In-State School', 'Out-Of-State School', 'MD', 'DO', 'US IMG', 'Non US IMG']
    df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce')

    print(value)
    # Create a folder in reports named "value"
    output_folder = f'reports/{value}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate descriptive statistics for the numeric fields
    descriptive_stats = df[numeric_fields].describe()
    print(descriptive_stats)
    # Save descriptive statistics to a CSV file in the output folder
    descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_stats.csv'))
    
    # Retrieve the values from columns 4 and 5 (gold_count and silver_count)
    gold_count = int(row.iloc[3])  # Column 4: Gold Count
    silver_count = int(row.iloc[4])  # Column 5: Silver Count

    # Call simulation_tiered with the current df, gold_count, and silver_count
    with open(os.path.join(output_folder, 'random_simulation.txt'), 'w') as file:
        file.write(simulation_tiered(df, gold_count, silver_count))
    
    # Call optimal_tiered with the current df, gold_count, and silver_count
    with open(os.path.join(output_folder, 'optimal_signals.txt'), 'w') as file:
        file.write(optimal_tiered(df, gold_count, silver_count))
    
    geographic_bias_df = geographic_bias(df)
    geographic_bias_df.to_csv(os.path.join(output_folder, 'geographic_bias.csv'), index=False)

for index, row in specialties_not_tiered.iterrows():
    value = row["Specialty"]
    try:
        df = pd.read_csv(f'data/{value}.csv')
    except:
        continue

    # Convert relevant columns to numeric
    numeric_fields = ['Signal', 'No Signal', 'In-State School', 'Out-Of-State School', 'MD', 'DO', 'US IMG', 'Non US IMG']
    existing_numeric_fields = [field for field in numeric_fields if field in df.columns]
    df[existing_numeric_fields] = df[existing_numeric_fields].apply(pd.to_numeric, errors='coerce')

    print(value)
    # Create a folder in reports named "value"
    output_folder = f'reports/{value}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate descriptive statistics for the numeric fields
    descriptive_stats = df[existing_numeric_fields].describe()
    print(descriptive_stats)
    # Save descriptive statistics to a CSV file in the output folder
    descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_stats.csv'))
    
    if 'Signal' in df.columns:
        # Retrieve the values from column 3 (signal_count)
        signal_count = int(row.iloc[2])  # Column 3: Signal Count

        # Call simulation with the current df and signal_count
        with open(os.path.join(output_folder, 'random_simulation.txt'), 'w') as file:
            file.write(simulation(df, signal_count))
        
        # Call optimal with the current df and signal_count
        with open(os.path.join(output_folder, 'optimal_signals.txt'), 'w') as file:
            file.write(optimal(df, signal_count))
    
    geographic_bias_df = geographic_bias(df)
    geographic_bias_df.to_csv(os.path.join(output_folder, 'geographic_bias.csv'), index=False)



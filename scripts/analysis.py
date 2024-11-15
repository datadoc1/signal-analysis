import random
import itertools
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt

def simulation_tiered(df, gold_count, silver_count):
    # Set the random seed
    random.seed(42)

    # Initialize lists to store the sums
    gold_signal_sums = []
    silver_signal_sums = []
    total_signal_sums = []

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

        total_signal_sum = gold_signal_sum + silver_signal_sum
        # Append the sums to the lists
        gold_signal_sums.append(gold_signal_sum)
        silver_signal_sums.append(silver_signal_sum)
        total_signal_sums.append(total_signal_sum)

    # Calculate the mean of the gold_signal_sums and silver_signal_sums
    mean_gold_signal_sum = sum(gold_signal_sums) / len(gold_signal_sums)
    mean_silver_signal_sum = sum(silver_signal_sums) / len(silver_signal_sums)
    mean_total_signal_sum = sum(total_signal_sums) / len(total_signal_sums)

    # Calculate the 95% confidence interval for the gold_signal_sums
    gold_signal_sums.sort()
    lower_bound_gold = gold_signal_sums[249]
    upper_bound_gold = gold_signal_sums[9749]

    # Calculate the 95% confidence interval for the silver_signal_sums
    silver_signal_sums.sort()
    lower_bound_silver = silver_signal_sums[249]
    upper_bound_silver = silver_signal_sums[9749]

    # Calculate the 95% confidence interval for the total_signal_sums
    total_signal_sums.sort()
    lower_bound_total = total_signal_sums[249]
    upper_bound_total = total_signal_sums[9749]

    # Save the results
    # Round the results to 2 decimal places
    mean_gold_signal_sum = round(mean_gold_signal_sum, 2)
    mean_silver_signal_sum = round(mean_silver_signal_sum, 2)
    mean_total_signal_sum = round(mean_total_signal_sum, 2)
    lower_bound_gold = round(lower_bound_gold, 2)
    upper_bound_gold = round(upper_bound_gold, 2)
    lower_bound_silver = round(lower_bound_silver, 2)
    upper_bound_silver = round(upper_bound_silver, 2)
    lower_bound_total = round(lower_bound_total, 2)
    upper_bound_total = round(upper_bound_total, 2)

    # Prepare the results string
    results = (
        f'Simulation Results:\n'
        f'-------------------\n'
        f'We randomly picked {gold_count} Gold Signal rows and {silver_count} Silver Signal rows from the dataset.\n\n'
        f'This was repeated 10,000 times to calculate the average number of interviews one should expect if they were a completely average applicant with completely randomly selected signals.\n\n'
        f'Gold Signal Sum ({gold_count}): {mean_gold_signal_sum} (95% CI: {lower_bound_gold}-{upper_bound_gold})\n'
        f'Silver Signal Sum ({silver_count}): {mean_silver_signal_sum} (95% CI: {lower_bound_silver}-{upper_bound_silver})\n'
        f'Total Signal Sum ({gold_count + silver_count}): {mean_total_signal_sum} (95% CI: {lower_bound_total}-{upper_bound_total})\n'
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

def top_signals(df, column_name, count):
    return df.nlargest(count, column_name)

def generate_stacked_boxplot(df, column_names, output_file='stacked_boxplot.png'):
    fig = go.Figure()

    for column in column_names:
        fig.add_trace(go.Box(
            x=df[column],  # Changed from y to x
            name=column,
            boxpoints=False,  # Remove individual points
            whiskerwidth=0.2,
            line_width=1
        ))

    fig.update_layout(
        xaxis=dict(
            title='Interview Rate',
            zeroline=False
        ),
        boxmode='group'
    )

    # Save the plot as an image file
    pio.write_image(fig, output_file)
    return


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
    
    # Generate a stacked boxplot for the numeric fields
    generate_stacked_boxplot(df, numeric_fields, os.path.join(output_folder, 'stacked_boxplot.png'))

    # Save silver vs gold comparison stats
    with open(os.path.join(output_folder, 'silver_vs_gold.txt'), 'w') as file:
        # Count programs with equal signals
        equal_signals = len(df[df['Gold Signal'] == df['Silver Signal']])
        
        # Count programs where gold < silver
        gold_lower = len(df[df['Gold Signal'] < df['Silver Signal']])
        
        # Calculate correlation and p-value
        clean_data = df[['Gold Signal', 'Silver Signal']].dropna()
        correlation_matrix = np.corrcoef(clean_data['Gold Signal'], clean_data['Silver Signal'])
        r_squared = correlation_matrix[0,1]**2
        
        correlation, p_value = stats.pearsonr(clean_data['Gold Signal'], clean_data['Silver Signal'])
        
        # Write results
        file.write(f"Programs with equal Gold and Silver signals: {equal_signals}\n")
        file.write(f"Programs with Gold signal less than Silver signal: {gold_lower}\n")
        file.write(f"R-squared value: {r_squared:.3f}\n")
        file.write(f"P-value: {p_value:.6f}\n")
        file.write(f"Correlation: {correlation:.3f}\n")
        # Count and list programs where Gold or Silver signal is less than No Signal
        gold_below_no = df[df['Gold Signal'] < df['No Signal']]
        silver_below_no = df[df['Silver Signal'] < df['No Signal']]
        
        file.write("\nPrograms where Gold Signal < No Signal:\n")
        file.write(f"Count: {len(gold_below_no)}\n")
        for _, row in gold_below_no.iterrows():
            file.write(f"{row['Program']}: Gold={row['Gold Signal']}, No={row['No Signal']}\n")
        
        file.write("\nPrograms where Silver Signal < No Signal:\n")
        file.write(f"Count: {len(silver_below_no)}\n")
        for _, row in silver_below_no.iterrows():
            file.write(f"{row['Program']}: Silver={row['Silver Signal']}, No={row['No Signal']}\n")
    
    # Generate a scatterplot for Gold Signal and Silver Signal
    scatterplot = go.Figure()
    scatterplot.add_trace(go.Scatter
    (
        x=df['Gold Signal'],
        y=df['Silver Signal'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            opacity=0.5
        )
    ))

    scatterplot.update_layout(
        xaxis_title='Gold Signal',
        yaxis_title='Silver Signal', 
        title='Gold Signal vs Silver Signal'
    )

    # Calculate R-squared
    # Remove non-numeric values and get clean data
    clean_data = df[['Gold Signal', 'Silver Signal']].dropna()
    clean_data = clean_data[clean_data['Gold Signal'].apply(lambda x: isinstance(x, (int, float)))]
    clean_data = clean_data[clean_data['Silver Signal'].apply(lambda x: isinstance(x, (int, float)))]
    
    # Calculate coefficients and r-squared only on clean data
    coefficients = np.polyfit(clean_data['Gold Signal'], clean_data['Silver Signal'], 1)
    polynomial = np.poly1d(coefficients)
    r_squared = np.corrcoef(clean_data['Gold Signal'], clean_data['Silver Signal'])[0,1]**2

    # Add trendline
    x_range = np.linspace(df['Gold Signal'].min(), df['Gold Signal'].max(), 100)
    scatterplot.add_trace(go.Scatter(
        x=x_range,
        y=polynomial(x_range),
        mode='lines',
        name=f'Trendline (R² = {r_squared:.3f})',
        line=dict(color='red', dash='dash')
    ))

    # Save the scatterplot
    scatterplot.write_image(os.path.join(output_folder, 'signal_scatter.png'))

    # Generate descriptive statistics for the numeric fields
    descriptive_stats = df[numeric_fields].describe()
    print(descriptive_stats)
    # Save descriptive statistics to a CSV file in the output folder
    descriptive_stats.to_csv(os.path.join(output_folder, 'descriptive_stats.csv'))

    top_gold_df = top_signals(df, 'Gold Signal', 5)
    top_silver_df = top_signals(df, 'Silver Signal', 10)
    top_gold_df.to_csv(os.path.join(output_folder, 'top_gold.csv'), index=False)
    top_silver_df.to_csv(os.path.join(output_folder, 'top_silver.csv'), index=False)
    
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

    ## Generate Heatmaps
    # Load the shapefile
    shapefile_path = 'shapefiles/cb_2021_us_state_20m.shp'
    gdf = gpd.read_file(shapefile_path)

    # Merge the geographic_bias_df with the shapefile GeoDataFrame
    merged = gdf.set_index('STUSPS').join(geographic_bias_df.set_index('State'))

    # Fill NaN values with 0
    merged['Mean Difference'] = merged['Mean Difference'].fillna(0)
    merged['Sum Difference'] = merged['Sum Difference'].fillna(0)

    # Create a function to plot the map with insets
    def plot_with_insets(data, column, title, output_path):
        fig = plt.figure(figsize=(15, 10))
        
        # Main map
        ax = fig.add_axes([0, 0, 1, 1])
        continental = data[~data.index.isin(['AK', 'HI', 'PR'])]
        continental.plot(column=column, cmap='Blues', linewidth=0.8, 
                        edgecolor='0.8', legend=True, ax=ax)
        
        # Alaska inset - made larger
        ax_ak = fig.add_axes([0.02, 0.02, 0.3, 0.3])
        alaska = data[data.index == 'AK']
        alaska.plot(column=column, cmap='Blues', linewidth=0.8, 
                   edgecolor='0.8', ax=ax_ak)
        ax_ak.set_title('Alaska')
        ax_ak.axis('off')
        
        # Hawaii inset
        ax_hi = fig.add_axes([0.35, 0.05, 0.2, 0.2])
        hawaii = data[data.index == 'HI']
        hawaii.plot(column=column, cmap='Blues', linewidth=0.8, 
                   edgecolor='0.8', ax=ax_hi)
        ax_hi.set_title('Hawaii')
        ax_hi.axis('off')

        # Puerto Rico inset
        ax_pr = fig.add_axes([0.60, 0.05, 0.2, 0.2])
        puerto_rico = data[data.index == 'PR']
        puerto_rico.plot(column=column, cmap='Blues', linewidth=0.8,
                        edgecolor='0.8', ax=ax_pr)
        ax_pr.set_title('Puerto Rico')
        ax_pr.axis('off')
        
        # Main map settings
        ax.set_title(title, fontdict={'fontsize': '15', 'fontweight': '3'})
        ax.axis('off')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot both maps
    plot_with_insets(merged, 'Mean Difference',
                     'Geographic Bias Heatmap: Average Increase in Interview Odds for Attending an In-State Medical School',
                     os.path.join(output_folder, 'geographic_bias_heatmap_average.png'))
    
    plot_with_insets(merged, 'Sum Difference',
                     'Geographic Bias Heatmap: Total Increase in Expected Interviews for Attending an In-State Medical School',
                     os.path.join(output_folder, 'geographic_bias_heatmap_total.png'))
    
    

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
    top_signals_df = top_signals(df, 'Signal', 10)
    top_signals_df.to_csv(os.path.join(output_folder, 'top_signals.csv'), index=False)
    
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



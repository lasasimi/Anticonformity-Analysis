import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns



print(os.getcwd())

# Define the paths for raw and clean data
data_clean = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/Clean_files/'

# Merge the 3 DataFrames
df_1 = pd.read_csv(data_clean + 'clean_long_format_k1hhm7lf.csv')  
df_2 = pd.read_csv(data_clean + 'clean_long_format_pfu1qwfx.csv')  
df_3 = pd.read_csv(data_clean + 'clean_long_format_7agw525e.csv')  

df_merged = pd.concat([df_1, df_2, df_3], ignore_index=True)

df_merged
df_merged.shape

# Save the merged DataFrame to a CSV file
merged_filename = 'clean_long_format_merged.csv'
df_merged.to_csv(data_clean + merged_filename, index=False)
y = 1560/13 # rows / 10 + 3 rounds
y

#### Pre-survey data
"""
Pre-survey has 3 rounds
We focus on scenario_code, political_charge, emotional_charge, and response
"""

# Split scenario_code into scenario_number and variant
df_new = df_merged.copy()
df_new

# Only keep rows where scenario is not NaN
df_new = df_new[df_new['scenario'].notna()]
df_new.shape
x = 360/3 # rows / 3 rounds
x

# Extract scenario number and variant into new columns
df_new[['scenario_number', 'variant']] = df_new['scenario_code'].str.extract(r'(s\d+)_(n|p)') 
# regex note: s = literal character “s” \d+ = one or more digits (e.g., 14 or 2)(n|p) = match either 'n' or 'p'
df_new[['scenario_number', 'variant']]



# Define a color palette for each variant
response_palette = {
    "n": "#9be2d3",
    "p": "#c5a7de"
}

# List of variables
variables = [
    "response",
    "political_charge",
    "emotional_charge",
]

df = df_new.copy()


### Categorizing and ordering correctly
# Ensure scenario_number is ordered correctly
df['scenario_number'] = pd.Categorical(
    df['scenario_number'],
    categories=['s2', 's4', 's9'],
    ordered=True
)


# Ensure variant is ordered consistently
df['variant'] = pd.Categorical(
    df['variant'],
    categories=['n', 'p'],  # Define the order: 'n' first, 'p' second
    ordered=True
)
df.dtypes

# Check how many people in each scenario
df.groupby(['scenario_number']).size().reset_index(name='count')

# List of variables
variables = [
    "response",
    "political_charge",
    "emotional_charge",
]



### Creating a dictionary to store scenario titles
# Revised scenario titles
scenario_titles = {
    "s2": "The New Family in the Neighborhood",
    "s4": "The Gardening Company",
    "s9": "Repurposing the Community's Hall",
}


# Dictionary to store the grouped DataFrames
grouped_counts = {}
# Loop through each variable and group the data
for var in variables:
    # Group by scenario_number, variant, and the variable, then count
    grouped = df.groupby(['scenario_number', 'variant', var]).size().reset_index(name='count')
    
    # Calculate the total count for each scenario_number and variant
    grouped['total_per_scenario_variant'] = grouped.groupby(['scenario_number', 'variant'])['count'].transform('sum')
    
    # Calculate the proportion for each group
    grouped['proportion'] = grouped['count'] / grouped['total_per_scenario_variant']
    
    # Store in the dictionary
    grouped_counts[var] = grouped


# Combine all DataFrames in grouped_counts into a single DataFrame for inspection
grouped_counts_df = pd.concat(grouped_counts.values(), 
                              keys=grouped_counts.keys()).reset_index(level=0).rename(columns={'level_0': 'vars'})

grouped_counts_df['value'] = grouped_counts_df[['response', 'political_charge', 'emotional_charge']].bfill(axis=1).iloc[:, 0]
grouped_counts_df_long = grouped_counts_df.copy()
# drop the variable columns
grouped_counts_df_long = grouped_counts_df.drop(columns=['response', 'political_charge', 'emotional_charge'])

# Ensure scenario_number and variable are categorical and ordered
grouped_counts_df_long['scenario_number'] = pd.Categorical(
    grouped_counts_df_long['scenario_number'],
    categories=['s2', 's4', 's9'],
    ordered=True
)
grouped_counts_df_long['variable'] = pd.Categorical(
    grouped_counts_df_long['variable'],
    categories=['response', 'emotional_charge', 'political_charge'],
    ordered=True
)
grouped_counts_df_long['value'] = pd.Categorical(
    grouped_counts_df_long['value'],
    categories=[-1, 0, 1, 2, 3, 4, 5, 6],  # Ensure all values are included
    ordered=True
)

# Check if the proportion for each variable can be summed to 1 (that means it is correct)
proportion_check = grouped_counts_df_long.groupby(['scenario_number', 'variant', 'variable'])['proportion'].sum().reset_index()
proportion_check

# Revised scenario titles
scenario_titles = {
    "s2": "The New Family in the Neighborhood",
    "s4": "The Gardening Company",
    "s9": "Repurposing the Community's Hall",
}


# Prepare the figure: 3 rows (scenarios) x 3 columns (variables)
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
for row_idx, scenario in enumerate(grouped_counts_df_long['scenario_number'].cat.categories):
    for col_idx, variable in enumerate(grouped_counts_df_long['variable'].cat.categories):
        ax = axes[row_idx, col_idx]
        data = grouped_counts_df_long[
            (grouped_counts_df_long['scenario_number'] == scenario) &
            (grouped_counts_df_long['variable'] == variable)
        ]
        sns.barplot(
            data=data,
            x='value',  # Use 'value' column for x-axis
            y='proportion',
            hue='variant',
            palette=response_palette,
            ax=ax
        )
        # Set x-ticks and limits for even distribution
        if variable == 'response':
            #ax.set_xticks([-1, 0, 1,])
            ax.set_xlim(-1, 3)
        else:
            #ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(1, 7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{scenario_titles[scenario]} - {variable.replace('_', ' ').title()}")
        ax.set_xlabel(variable.replace('_', ' ').title(), labelpad=15)
        if col_idx == 0:
            ax.set_ylabel('Proportion')
        else:
            ax.set_ylabel('')
        # Custom legend with counts
        handles, labels = ax.get_legend_handles_labels()
        variant_counts = data.groupby('variant')['count'].sum()
        new_labels = [f"{variant} (n={variant_counts.get(variant, 0)})" for variant in labels]
        ax.legend(handles, new_labels, title='Variant')    
plt.tight_layout()
plt.subplots_adjust(hspace=0.4) 
plt.show()

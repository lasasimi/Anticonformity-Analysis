import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/Clean_files/presurvey_long_format.csv') # change file name as needed
df = pd.read_csv('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/Clean_files/presurvey_mock_long_format.csv') # change file name as needed
df.head()
df.shape
df["scenario"].unique()


def extract_code(s):
    if pd.isna(s):
        return None
    match = re.search(r"'code': '([^']+)'", s)
    return match.group(1) if match else None

df['scenario_code'] = df['scenario'].apply(extract_code)
df['scenario_code'].unique()


# Extract scenario number and variant into new columns
# Use scenario_code, which is already extracted from the scenario column
# This will create two columns: scenario_number (e.g., s9) and variant (e.g., n or p)
df[['scenario_number', 'variant']] = df['scenario_code'].str.extract(r'(s\d+)_(n|p)') 
#regex note: s = literal character “s” \d+ = one or more digits (e.g., 14 or 2)(n|p) = match either 'n' or 'p'

# Check the first few rows to make sure it worked
print(df[['scenario_number', 'variant']].head())

print(df["response"].value_counts(dropna=False))

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
df.columns
# Dictionary to store the grouped DataFrames
grouped_counts = {}

# Loop through each variable and group the data
for var in variables:
    grouped_counts[var] = df.groupby(['scenario_number', 'variant', var]).size().reset_index(name='count')
    grouped_counts[var]['scenario_num_int'] = grouped_counts[var]['scenario_number'].str.extract(r's(\d+)').astype(int)
    sorted_scenarios = grouped_counts[var].sort_values('scenario_num_int')['scenario_number'].unique()
    grouped_counts[var]['scenario_number'] = pd.Categorical(grouped_counts[var]['scenario_number'], categories=sorted_scenarios, ordered=True)

# Create the catplot   
for idx, var in enumerate(variables):
    g = sns.catplot(
        data=grouped_counts[var],
        x=var,
        y='count',
        hue='variant',
        palette=response_palette, 
        col='scenario_number',  # Use column name, not Series
        kind='bar',
        col_wrap=5,
        height=3,
        dodge=True,
    )
    # Set x-axis ticks for the first plot only
    if idx == 0:
        for ax in g.axes.flat:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['-1', '+1'])
    else:
        for ax in g.axes.flat:
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels([1, 2, 3, 4, 5])

    g.fig.suptitle(var, y=1.02)
    g.set(ylim=(0, 30))  # <-- Set y-axis limits here
    plt.show()

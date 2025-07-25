import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 

print(os.getcwd())

# Load and merge the page time data
file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/Clean_files/'
clean_df_1 = pd.read_csv(file_location + 'clean_long_format_k1hhm7lf.csv')
clean_df_2 = pd.read_csv(file_location + 'clean_long_format_pfu1qwfx.csv')
clean_df_3 = pd.read_csv(file_location + 'clean_long_format_7agw525e.csv')

# Merge the 3 dfs
df = pd.concat([clean_df_1, clean_df_2, clean_df_3], ignore_index=True)

participant_code = df['participant.code'].unique().tolist()
len(participant_code)
df.head()

# Filter only relevant rows in the mock part of the app; indicated by neighbors is not NaN
relevant_df = df[df['neighbors'].notna()]

# How many participants in each treatment? 
relevant_df['participant.treatment'].value_counts() / 10 # rounds

# Load the neighbor configurations
file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/'
neighbors_configs = pd.read_csv(file_location + 'neighbors_configurations.csv')

# ignore index
neighbors_configs.reset_index(drop=True, inplace=True)

# Ensure the neighbors column is a string, remove any brackets
df = relevant_df.copy()

# Ensure it does not have brackets and spaces
df['neighbors'] = df['neighbors'].astype(str).str.strip("[]").str.replace(' ', '',regex=False)

# Combine the three columns into one string column in neighbors_configs
neighbor_cols = ['neighbor_1', 'neighbor_2', 'neighbor_3']
neighbors_configs.columns = neighbor_cols
print(neighbors_configs.columns)

neighbors_configs['neighbors'] = neighbors_configs[neighbor_cols].astype(int).astype(str).agg(','.join, axis=1)


# Create a set of sorted neighbor strings from neighbors_configs for fast lookup
neighbors_set = set(
    ','.join(sorted(row.split(','), key=int))
    for row in neighbors_configs['neighbors']
)

correct_conformity = {
    '-1,-1,-1': '-1',
    '-1,-1,0': '-1',
    '-1,-1,1': '-1',
    '-1,0,1': 'None',
    '-1,0,0': '0',
    '-1,1,1': '1',
     '0,0,0': '0',
     '0,0,1': '0',
     '0,1,1': '1',
     '1,1,1': '1',
}

correct_anticonformity = {
    '-1,-1,-1': ['1','0'],
    '-1,-1,0': ['1'],
    '-1,-1,1': ['0'],
    '-1,0,1': 'None',
    '-1,0,0': ['1'],
    '-1,1,1': ['0'],
     '0,0,0': ['-1','1'],
     '0,0,1': ['-1'],
     '0,1,1': ['-1'],
     '1,1,1': ['-1','0'],
}



# Function to check if the response is correct based on the treatment type
def check_response_correct(row):
    """Input a row of the DataFrame and check if the response is correct based on the treatment type."""
    # Choose the correct dictionary based on treatment
    if str(row['participant.treatment']).startswith('C'):
        correct = row['correct_conformity']
    elif str(row['participant.treatment']).startswith('A'):
        correct = row['correct_anticonformity']
    else:
        return 'NA'  # Return 'NA' for treatments not starting with 'C' or 'A'
    
    # Convert response to string
    response_str = str(int(row['response'])) if isinstance(row['response'], float) and row['response'] == int(row['response']) else str(row['response'])
    
    # Handle 'None' cases
    if response_str == 'None' or (isinstance(correct, str) and correct == 'None'):
        return 'None'
    
    # Check correctness
    if isinstance(correct, list):
        return 'correct' if response_str in correct else 'incorrect'
    else:
        return 'correct' if response_str == correct else 'incorrect'



# Function to standardize the neighbors combinations (permutation to combination)
def which_neighbors_combinations(neighbor_input):
    # If input is a Series or list, process each element and return a list of standardized strings
    if isinstance(neighbor_input, (pd.Series, list)):
        result = []
        for neighbor_str in neighbor_input:
            nums = [int(x.strip()) for x in str(neighbor_str).split(',')]
            result.append(','.join(map(str, sorted(nums))))
        return result
    # If input is a single string
    nums = [int(x.strip()) for x in str(neighbor_input).split(',')]
    return ','.join(map(str, sorted(nums)))



# Create a toy dataset for testing
neighbors_test = ['1,1,1','0,-1,1','1,-1,1']
# Standardize the neighbors column
neighbors_test_std = which_neighbors_combinations(neighbors_test)
treatment_test = ['C_n', 'AC_n', 'NO_p']
response_test = ['1', '0', '-1'] # for 1 round

# Convert into a df with 'response' and 'neighbors' columns
test_data = pd.DataFrame({
    'neighbors': neighbors_test_std,
    'response': response_test,
    'participant.treatment': treatment_test,
})



# Create a new column for checking correctness of responses based on treatment and neighbors configs
test_data['correct_conformity'] = test_data['neighbors'].map(correct_conformity)
test_data['correct_anticonformity'] = test_data['neighbors'].map(correct_anticonformity)
# Apply to each row
test_data['correct_response_given_treatment'] = test_data.apply(check_response_correct_1, axis=1)


# Now we need to make the function works in multiple rounds
test_data2 = df.copy()
test_data2 = test_data2[0:30]   # For testing, take only the first 30 rows
test_data2.columns

# 3 steps:
# 1. Standardize the neighbors column
test_data2['neighbors'] = test_data2['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
test_data2['correct_conformity'] = test_data2['neighbors'].map(correct_conformity)
test_data2['correct_anticonformity'] = test_data2['neighbors'].map(correct_anticonformity)



# Remove unnecessary columns
test_data2 = test_data2[['participant.code', 
                         'round_no', 
                         'neighbors', 
                         'response', 
                         'participant.treatment', 
                         'correct_conformity', 'correct_anticonformity']]


# 3. Apply function to each row
test_data2['correct_response_given_treatment'] = test_data2.apply(check_response_correct, axis=1)


# Remove unnecessary columns
df = df[['participant.code', 
                         'round_no', 
                         'neighbors', 
                         'response', 
                         'participant.treatment',]]

# Get data for the round 10 only
df_round10 = df[df['round_no'] == 10]

# 3 steps:
# 1. Standardize the neighbors column
df_round10['neighbors'] = df_round10['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
df_round10['correct_conformity'] = df_round10['neighbors'].map(correct_conformity)
df_round10['correct_anticonformity'] = df_round10['neighbors'].map(correct_anticonformity)

# 3. Apply function to each row
df_round10['correct_response_given_treatment'] = df_round10.apply(check_response_correct, axis=1)

### CORRECT OR INCORRECT RESPONSE PLOT (CONFORMITY AND ANTICONFORMITY) Note: only for round 10
# TODO: plot round by round
import seaborn as sns       
# Plot barplot to count how many people get the correct response in round 10
# 2 plots: one for conformity and one for anticonformity

# Create a DataFrame for conformity responses
conformity_df = df_round10[df_round10['participant.treatment'].str.startswith('C')]
conformity_counts = conformity_df['correct_response_given_treatment'].value_counts().reset_index()
conformity_counts.columns = ['response', 'count']   
# Create a bar plot for conformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=conformity_counts, x='response', y='count', palette    ='viridis')                 
plt.title('Conformity Responses in Round 10')
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 40)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('conformity_responses_round10.png', dpi=300)
plt.show()

# Create a DataFrame for anticonformity responses
anticonformity_df = df_round10[df_round10['participant.treatment'].str.startswith('A')]
anticonformity_counts = anticonformity_df['correct_response_given_treatment'].value_counts().reset_index()
anticonformity_counts.columns = ['response', 'count']
# Create a bar plot for anticonformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=anticonformity_counts, x='response', y='count', palette='viridis')
plt.title('Anticonformity Responses in Round 10')
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 40)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anticonformity_responses_round10.png', dpi=300)
plt.show()


# Get the round 10 
df_round10 = df[df['round_no'] == 10]

# Get the round 10 and round 1 only
df_round1_10 = df[df['round_no'].isin([1, 10])]


# TODO: color change based on if round 1 response is different from round 10 response, save if different in a new column called 'response_change'
df_round1_10['response_change'] = df_round1_10.groupby('participant.code')['response'].transform(lambda x: x.iloc[0] != x.iloc[-1])

# Remove round_no == 1
df_round1_10 = df_round1_10[df_round1_10['round_no'] == 10]

## RESPONSE CHANGE PLOT
# Plot the response in round 10 but color the bars based on if the response changed from round 1 to round 10
#TODO iterate over scenario_number
plt.figure(figsize=(10, 6))
sns.countplot(data=df_round1_10, x='response', hue='response_change',
              palette={True: 'lightcoral', False: 'lightblue'},
              order=df_round1_10['response'].value_counts().index)
plt.title('Response in Round 10 with Change from Round 1')
plt.xlabel('Response')
plt.ylabel('Count')
plt.xticks(rotation=45)     
plt.legend(title='Response Change', loc='upper right', labels=['Changed', 'Unchanged'])
plt.tight_layout()
plt.show()


## Refresh df
# Ensure the neighbors column is a string, remove any brackets
df = relevant_df.copy()

# Ensure it does not have brackets and spaces
df['neighbors'] = df['neighbors'].astype(str).str.strip("[]").str.replace(' ', '',regex=False)

df_complete = df.copy()

# Create a new column for scenario code and variant
df_complete[['scenario_number', 'variant']] = df_complete['participant.scenario'].str.extract(r'(s\d+)_(\w+)')

# 3 steps:
# 1. Standardize the neighbors column
df_complete['neighbors'] = df_complete['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
df_complete['correct_conformity'] = df_complete['neighbors'].map(correct_conformity)
df_complete['correct_anticonformity'] = df_complete['neighbors'].map(correct_anticonformity)

# 3. Apply function to each row
df_complete['correct_response_given_treatment'] = df_complete.apply(check_response_correct, axis=1)

df_complete['response_change'] = df_complete.groupby('participant.code')['response'].transform(
    lambda x: x.loc[df_complete['round_no'] == 1].values[0] != x.loc[df_complete['round_no'] == 10].values[0]
)


import matplotlib.pyplot as plt
import seaborn as sns

# Combine variant and response_change for hue
df_complete['variant_change'] = df_complete['variant'] + '_' + df_complete['response_change'].astype(str) 

# Loop over each scenario_number
#TODO: sort by scenario_number and variant as a categorical variable
for scenario in df_complete['scenario_number'].unique():
    df_scenario = df_complete[df_complete['scenario_number'] == scenario]
    plt.figure(figsize=(10, 6))
    # Use hue for variant and style for response_change
    sns.countplot(
        data=df_scenario,
        x='response',
        hue='variant_change',
        order=df_scenario['response'].value_counts().index,
        palette='Set2'
    )
    plt.title(f'Response in Round 10 for {scenario} (by Variant & Response Change)')
    plt.xlabel('Response')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Variant & Changed', loc='upper right')
    plt.tight_layout()
    plt.show()
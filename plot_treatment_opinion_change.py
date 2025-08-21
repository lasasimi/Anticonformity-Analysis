import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 
import seaborn as sns       

print(os.getcwd())

# Load and merge the page time data
# windows 
file_location = '/Users/Lasmi Marbun/Documents/Git/Anticonformity-Analysis/Clean_files/'
# mac
# file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/Clean_files/'
clean_df_1 = pd.read_csv(file_location + 'clean_long_format_k1hhm7lf.csv')
clean_df_2 = pd.read_csv(file_location + 'clean_long_format_pfu1qwfx.csv')
clean_df_3 = pd.read_csv(file_location + 'clean_long_format_7agw525e.csv')

# Merge the 3 dfs
df = pd.concat([clean_df_1, clean_df_2, clean_df_3], ignore_index=True)

# Load the neighbor configurations
file_location = '/Users/Lasmi Marbun/Documents/Git/Anticonformity-Analysis/'
neighbors_configs = pd.read_csv(file_location + 'neighbors_configurations.csv')

# ignore index
neighbors_configs.reset_index(drop=True, inplace=True)


# Combine the three columns into one string column in neighbors_configs
neighbor_cols = ['neighbor_1', 'neighbor_2', 'neighbor_3']
neighbors_configs.columns = neighbor_cols
print(neighbors_configs.columns)

neighbors_configs['neighbors'] = neighbors_configs[neighbor_cols].astype(int).astype(str).agg(','.join, axis=1)



participant_code = df['participant.code'].unique().tolist()
len(participant_code)
df.head()

## PRESURVEY PART OF THE APP
## Plot which scenario has the most non-neutral 'response' from 'participant.scenario'

# Count the number of non-neutral responses for each scenario in the presurvey
# Merge the 3 dfs
df_scenario = pd.concat([clean_df_1, clean_df_2, clean_df_3], ignore_index=True)

# Select only rows with no 'neighbors' to focus on the presurvey part of the app
df_presurvey = df_scenario[df_scenario['neighbors'].isna()]
df_presurvey['scenario_code'].value_counts()

df_mock = df[df['neighbors'].notna()]

non_neutral_counts = df_presurvey[df_presurvey['response'] != 0]['scenario_code'].value_counts()
non_neutral_counts = non_neutral_counts.sort_index()

neutral_counts = df_presurvey[df_presurvey['response'] == 0]['scenario_code'].value_counts()
neutral_counts = neutral_counts.sort_index()

# Merge the counts into a single DataFrame for plotting
counts_df = pd.DataFrame({'Non-Neutral': non_neutral_counts, 'Neutral': neutral_counts}).fillna(0)  

plt.figure(figsize=(10, 6))
counts_df.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgray'])
plt.title('Counts of Non-Neutral and Neutral Responses by Scenario')    
plt.xlabel('Scenario Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Response Type',loc='upper left', bbox_to_anchor=(1.05, 1),)
plt.tight_layout()
plt.savefig('neutral_non-neutral_response_counts.png', dpi=300)
plt.show()

## MOCK PART OF THE APP
# Filter only relevant rows in the mock part of the app; indicated by neighbors is not NaN
df_mock = df[df['neighbors'].notna()]

# How many participants in each treatment? 
df_mock['participant.treatment'].value_counts() / 10 # rounds

# Ensure the neighbors column is a string, remove any brackets
df_mock = df_mock.copy()

# Ensure it does not have brackets and spaces
df_mock['neighbors'] = df_mock['neighbors'].astype(str).str.strip("[]").str.replace(' ', '',regex=False)

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

# Merge the values from correct_conformity and correct_anticonformity for each key
correct_both = {}
for key in correct_conformity:
    val1 = correct_conformity[key]
    val2 = correct_anticonformity[key]
    # Convert to list if not already
    vals = []
    if isinstance(val1, list):
        vals.extend(val1)
    elif val1 != 'None':
        vals.append(val1)
    if isinstance(val2, list):
        vals.extend(val2)
    elif val2 != 'None':
        vals.append(val2)
    # Remove duplicates and 'None'
    vals = [v for v in set(vals) if v != 'None']
    # If both are 'None', keep 'None'
    if not vals and (val1 == 'None' or val2 == 'None'):
        correct_both[key] = 'None'
    else:
        correct_both[key] = vals


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

# Function to check if the response is in correct_both, then check conformity/anticonformity
def check_response_treatment(row):
    """
    Checks if the response is in the correct_both dictionary for the given neighbors.
    If not, returns Unknown.
    If yes, checks if the response is in correct_conformity; if not, checks correct_anticonformity.
    If the value is 'None', returns 'None'.
    """
    neighbors = row['neighbors']
    response_str = str(int(row['response'])) if isinstance(row['response'], float) and row['response'] == int(row['response']) else str(row['response'])
    correct_both_val = correct_both.get(neighbors, None)
    if correct_both_val is None:
        return 'Unknown'
    if correct_both_val == 'None' or response_str == 'None':
        return 'None'
    if isinstance(correct_both_val, list):
        if response_str not in correct_both_val:
            return 'Unknown'
    else:
        if response_str != correct_both_val:
            return 'Unknown'
    # Now check conformity first
    correct_conf = correct_conformity.get(neighbors, None)
    if correct_conf == 'None':
        return 'None'
    if isinstance(correct_conf, list):
        if response_str in correct_conf:
            return 'conformity'
    else:
        if response_str == correct_conf:
            return 'conformity'
    # If not conformity, check anticonformity
    correct_anti = correct_anticonformity.get(neighbors, None)
    if correct_anti == 'None':
        return 'None'
    if isinstance(correct_anti, list):
        if response_str in correct_anti:
            return 'anticonformity'
    else:
        if response_str == correct_anti:
            return 'anticonformity'
    return False


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
test_data['correct_response_given_treatment'] = test_data.apply(check_response_correct, axis=1)


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
df_mock = df_mock[['participant.code', 
                         'round_no', 
                         'neighbors', 
                         'response', 
                         'participant.treatment',]]

# Get data for the round 10 only
df_round10 = df_mock[df_mock['round_no'] == 10]

# 3 steps:
# 1. Standardize the neighbors column
df_round10['neighbors'] = df_round10['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
df_round10['correct_conformity'] = df_round10['neighbors'].map(correct_conformity)
df_round10['correct_anticonformity'] = df_round10['neighbors'].map(correct_anticonformity)

# 3. Apply function to each row
df_round10['correct_response_given_treatment'] = df_round10.apply(check_response_correct, axis=1)

### CORRECT OR INCORRECT RESPONSE PLOT (CONFORMITY AND ANTICONFORMITY) Note: only for round 10

# Plot barplot to count how many people get the correct response in round 10
# 2 plots: one for conformity and one for anticonformity

# Create a DataFrame for conformity responses
conformity_df = df_round10[df_round10['participant.treatment'].str.startswith('C')]

# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
conformity_df['correct_response_given_treatment'] = pd.Categorical(conformity_df['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'],)
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

# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
anticonformity_df['correct_response_given_treatment'] = pd.Categorical(anticonformity_df['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'],)
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

## EXPECTED RESPONSE CHANGE PLOT
## Plot the treatment opinion change in every rounds
# Get the unique participant codes
df = df_mock.copy()
participant_codes = df['participant.code'].unique()
len(participant_codes) # N=120

# Get the unique treatment types
treatments = ['C_n', 'C_p', 'AC_n', 'AC_p', 'NO_n', 'NO_p']

# Convert to categorical type for better plotting
df['participant.treatment'] = pd.Categorical(df['participant.treatment'], categories=treatments)

# Get the unique round numbers
round_numbers = df['round_no'].unique()

# For all treatments, plot the response differentiated by expected / non-expected responses (correct/incorrect)

# 3 steps:
# 1. Standardize the neighbors column
df['neighbors'] = df['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
df['correct_conformity'] = df['neighbors'].map(correct_conformity)
df['correct_anticonformity'] = df['neighbors'].map(correct_anticonformity)



# 3. Apply function to each row
df['correct_response_given_treatment'] = df.apply(check_response_correct, axis=1)

# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
# And remove the NA from the NO treatment
df['correct_response_given_treatment'] = pd.Categorical(df['correct_response_given_treatment'],
                                                         categories=['correct', 'incorrect', 'None'])   
# Group the data to calculate counts for each round and response correctness
df_counts = df.groupby(['round_no', 'correct_response_given_treatment']).size().reset_index(name='count')

# Convert counts to proportions by dividing by 80 (40 participants * 2 treatments)
df_counts['proportion'] = df_counts['count'] / 80

# Group the data to calculate counts for each round and response correctness
df_treatment = df.copy()

# Create a new column for treatment names
df_treatment['treatment_type'] = df_treatment['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment')
)
# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
# And remove the NA from the NO treatment
df_treatment['correct_response_given_treatment'] = pd.Categorical(df_treatment['correct_response_given_treatment'],
                                                         categories=['correct', 'incorrect', 'None'])   

# Save df_treatment as .csv
df_treatment.to_csv('df_treatment.csv', index=False)

# Starts with A for Anticonformity and strats with C for Conformity
df_counts_treatment = df_treatment.groupby(['round_no', 'correct_response_given_treatment', 'treatment_type']).size().reset_index(name='count')

# Remove the rows with No Treatment
df_counts_treatment = df_counts_treatment[df_counts_treatment['treatment_type'] != 'No Treatment']

# Convert counts to proportions by dividing by 80 (40 participants * 2 treatments)
df_counts_treatment.loc[:,'proportion'] = df_counts['count'] / 80

df_counts_treatment
## COUNT PLOT
# Create subplots for Anticonformity and Conformity and Both
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=False)

# Filter data for Anticonformity and Conformity
anticonformity_data = df_counts_treatment[df_counts_treatment['treatment_type'] == 'Anticonformity']
conformity_data = df_counts_treatment[df_counts_treatment['treatment_type'] == 'Conformity']

# Plot average of both treatments
sns.lineplot(
    data=df_counts,
    x='round_no',
    y='count',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[0]
)

axes[0].set_title('Both Treatments')
axes[0].set_xlabel('Round Number')
axes[0].set_ylabel('')  # No y-axis label for the second plot
axes[0].set_xticks(range(1, 11))
axes[0].set_xticklabels(range(1, 11))
axes[0].set_ylim(0,60)

# Plot Anticonformity
sns.lineplot(
    data=anticonformity_data,
    x='round_no',
    y='count',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,
    dashes=True,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[1]
)
axes[1].set_title('Anticonformity Treatments')
axes[1].set_xlabel('Round Number')
axes[1].set_ylabel('Count')
axes[1].set_xticks(range(1, 11))
axes[1].set_xticklabels(range(1, 11))
axes[1].set_ylim(0,40)

# Plot Conformity
sns.lineplot(
    data=conformity_data,
    x='round_no',
    y='count',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,
    dashes=True,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[2]
)
axes[2].set_title('Conformity Treatments')
axes[2].set_xlabel('Round Number')
axes[2].set_ylabel('')  # No y-axis label for the second plot
axes[2].set_xticks(range(1, 11))
axes[2].set_xticklabels(range(1, 11))
axes[2].set_ylim(0,40)

# Remove legends from the first and second axes
axes[0].get_legend().remove()
axes[1].get_legend().remove()

# Add a single legend to the last axis (customized)
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(
    handles=handles,
    labels=['Expected', 'Not Expected', 'None'],  # Custom labels
    title='Expected Response',
    loc='upper left',
    bbox_to_anchor=(1.05, 1)
)

# Adjust layout
plt.tight_layout()
plt.show()
plt.savefig('All+C+AC_expected_response_change.png', dpi=300)
plt.show()


## PROPORTION PLOT
# Create subplots for Anticonformity and Conformity and Both
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

# Filter data for Anticonformity and Conformity
anticonformity_data = df_counts_treatment[df_counts_treatment['treatment_type'] == 'Anticonformity']
anticonformity_data.loc[:,'proportion'] = anticonformity_data['count'] / 40  # Convert counts to proportions
conformity_data = df_counts_treatment[df_counts_treatment['treatment_type'] == 'Conformity']
conformity_data.loc[:,'proportion'] = conformity_data['count'] / 40  # Convert counts to proportions
# Plot average of both treatments
sns.lineplot(
    data=df_counts,
    x='round_no',
    y='proportion',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[0]
)

axes[0].set_title('Both Treatments')
axes[0].set_xlabel('Round Number')
axes[0].set_ylabel('')  # No y-axis label for the second plot
axes[0].set_xticks(range(1, 11))
axes[0].set_xticklabels(range(1, 11))


# Plot Anticonformity
sns.lineplot(
    data=anticonformity_data,
    x='round_no',
    y='proportion',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,
    dashes=True,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[1]
)
axes[1].set_title('Anticonformity Treatments')
axes[1].set_xlabel('Round Number')
axes[1].set_ylabel('Count')
axes[1].set_xticks(range(1, 11))
axes[1].set_xticklabels(range(1, 11))

# Plot Conformity
sns.lineplot(
    data=conformity_data,
    x='round_no',
    y='proportion',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,
    dashes=True,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'},
    ax=axes[2]
)
axes[2].set_title('Conformity Treatments')
axes[2].set_xlabel('Round Number')
axes[2].set_ylabel('')  # No y-axis label for the second plot
axes[2].set_xticks(range(1, 11))
axes[2].set_xticklabels(range(1, 11))

# Remove legends from all but the last axis
for ax in axes[:-1]:
    ax.get_legend().remove()

# Add a single legend to the first axis (customized)
handles, labels = axes[0].get_legend_handles_labels()
axes[2].legend(
    handles=handles,
    labels=['Expected', 'Not Expected', 'None'],  # Custom labels
    title='Proportion of Expected Response',
    loc='upper left',
    bbox_to_anchor=(1.05, 1)
)

# Adjust layout
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('All+C+AC_expected_response_change_prop.png', dpi=300)
plt.show()

## SPECIFIC SCENARIO PLOT
# take from df_treatment
df = df_treatment
spec_scenario = 's9'
# Filter the DataFrame for the specific scenario
df_spec_scenario = df[df['participant.scenario'].str.startswith(spec_scenario)]

# How many participants in each specific scenario?
participant_in_scenario = df_spec_scenario['participant.code'].value_counts() / 10 # rounds
print(len(participant_in_scenario))
# s9 = 44
# s4 = 39
# s2 = 37

# Create a new column for treatment names
df_spec_scenario['treatment_type'] = df_spec_scenario['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment')
)

# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
# And remove the NA from the NO treatment
df_spec_scenario['correct_response_given_treatment'] = pd.Categorical(df_spec_scenario['correct_response_given_treatment'],
                                                         categories=['correct', 'incorrect', 'None'])   

# load df_steps_extneu
df_steps_extneu = pd.read_csv(file_location + 'df_steps_extneu.csv')
df_neutral = df_steps_extneu[df_steps_extneu['initially_neutral'] == 'Neutral']
# Filter out the participant.code that match the initially_neutral in df_neutral

df_spec_scenario_filter = df_spec_scenario[~df_spec_scenario['participant.code'].isin(df_neutral['participant.code'])]

# Starts with A for Anticonformity and strats with C for Conformity
df_counts_spec = df_spec_scenario.groupby(['round_no', 'correct_response_given_treatment', 'treatment_type']).size().reset_index(name='count')
df_counts_spec_filter = df_spec_scenario_filter.groupby(['round_no', 'correct_response_given_treatment', 'treatment_type']).size().reset_index(name='count')
# Remove the rows with No Treatment
df_counts_spec = df_counts_spec[df_counts_spec['treatment_type'] != 'No Treatment']
df_counts_spec_filter = df_counts_spec_filter[df_counts_spec_filter['treatment_type'] != 'No Treatment']
# Recount the participant after removing the No Treatment
df_spec_scenario[df_spec_scenario['treatment_type'] != 'No Treatment']['participant.code'].nunique()
df_spec_scenario_filter[df_spec_scenario_filter['treatment_type'] != 'No Treatment']['participant.code'].nunique()
# filtered s9 = 30
# filtered s2 = 11
# s9 = 35
# s4 = 25
# s2 = 20

df_counts_spec




## COUNT PLOT
# Plot for specific scenario expected response (all types) in a single plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    data=df_counts_spec,
    x='round_no',
    y='count',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    errorbar=None,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'}
)

plt.title(f'Expected Response in scenario {spec_scenario}')
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.ylim(0, 40)
plt.xticks(range(1, 11))

# Change legend labels
handles, labels = ax.get_legend_handles_labels()
custom_labels = ['Expected', 'Not Expected', 'None']
# The first legend entry is for 'hue', so skip it if present
if labels and labels[0] == 'correct_response_given_treatment':
    handles = handles[1:]
    labels = labels[1:]
ax.legend(handles=handles, labels=custom_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()



# Convert counts to proportions by dividing by counts of people in the specific scenario with treatment (excluding No Treatment)
spec_scenario_count = df_spec_scenario[df_spec_scenario['treatment_type'] != 'No Treatment']['participant.code'].nunique()
spec_scenario_count
df_counts_spec.loc[:,'proportion'] = df_counts_spec['count'] / spec_scenario_count

## PROPORTION PLOT
# Plot for specific scenario expected response (all types) in a single plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    data=df_counts_spec,
    x='round_no',
    y='proportion',
    hue='correct_response_given_treatment',
    style='correct_response_given_treatment',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    errorbar=None,
    palette={'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'}
)

plt.title(f'Expected Response in scenario {spec_scenario}')
plt.xlabel('Round Number')
plt.ylabel('Proportion')
plt.ylim(0, 1)
plt.xticks(range(1, 11))

# Change legend labels
handles, labels = ax.get_legend_handles_labels()
custom_labels = ['Expected', 'Not Expected', 'None']
# The first legend entry is for 'hue', so skip it if present
if labels and labels[0] == 'correct_response_given_treatment':
    handles = handles[1:]
    labels = labels[1:]
ax.legend(handles=handles, labels=custom_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

## PROPORTION PLOT (separated by Conformity and Anticonformity)
df_steps_extneu = pd.read_csv(file_location + 'df_steps_extneu.csv')



# Filter the ones with 
# recalculate proportion -- number of participants in each treatment type and scenario
df_spec_scenario = df_spec_scenario_filter.copy()
spec_sc_ac_count = df_spec_scenario[df_spec_scenario['treatment_type'] == 'Anticonformity']['participant.code'].nunique() 
# s9 filtered = 13
# s2 filtered = 9
# s2 = 15
# s4 = 11
# s9 = 14
spec_sc_c_count = df_spec_scenario[df_spec_scenario['treatment_type'] == 'Conformity']['participant.code'].nunique()
# s9 filtered = 17
# s2 filtered = 2
# s2 = 5
# s4 = 14
# s9 = 21

# Convert counts to proportions by dividing by counts of people in the specific scenario with treatment (excluding No Treatment)
df_counts_spec = df_counts_spec_filter.copy() # only do this if you want to use the filtered data / filtered out the initially neutral participants
df_counts_spec.loc[df_counts_spec['treatment_type'] == 'Anticonformity', 'proportion'] = df_counts_spec['count'] / spec_sc_ac_count
df_counts_spec.loc[df_counts_spec['treatment_type'] == 'Conformity', 'proportion'] = df_counts_spec['count'] / spec_sc_c_count
# Plot for specific scenario expected response (all types) in two subplots: Conformity and Anticonformity
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

treatment_types = ['Conformity', 'Anticonformity']
titles = ['Conformity', 'Anticonformity']
palette = {'correct': 'limegreen', 'incorrect': 'orange', 'None': 'gray'}
custom_labels = ['Expected', 'Not Expected', 'None']

for i, t_type in enumerate(treatment_types):
    subset = df_counts_spec[df_counts_spec['treatment_type'] == t_type]
    ax = axes[i]
    sns.lineplot(
        data=subset,
        x='round_no',
        y='proportion',
        hue='correct_response_given_treatment',
        style='correct_response_given_treatment',
        markers=True,
        dashes=True,
        errorbar=None,
        palette=palette,
        ax=ax
    )
    ax.set_title(f'Expected Response in scenario {spec_scenario} ({titles[i]})')
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 11))
    # Only show legend at the second subplot
    handles, labels = ax.get_legend_handles_labels()
    if i == 1:
        if labels and labels[0] == 'correct_response_given_treatment':
            handles = handles[1:]
            labels = labels[1:]
        ax.legend(handles=handles, labels=custom_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.show()


## PLOT THE EXPECTED X NON EXPECTED FOR INITIALLY NEUTRAL AND NON-NEUTRAL RESPONSES -- data is only from the mock round
# use df_steps_extneu from below
# Create a new column to distinguish initially neutral participants

# get the df_presurvey and df_mock
df_presurvey = df[df['neighbors'].isna()]
df_mock = df[df['neighbors'].notna()]

# add the round 0 to determine the initially neutral participants
df_merged = add_round_0(df_presurvey, df_mock)
df_merged_extneu = df_merged.copy()

# Remove round 0 data
df_merged_extneu = df_merged_extneu[df_merged_extneu['round_no'] != 0]

# Ensure neighbors does not have brackets and spaces
df_merged_extneu['neighbors'] = df_merged_extneu['neighbors'].astype(str).str.strip("[]").str.replace(' ', '',regex=False)

# Standardize the neighbors column
df_merged_extneu['neighbors'] = df_merged_extneu['neighbors'].apply(which_neighbors_combinations)

# Create column for Expected and Non-Expected responses
df_merged_extneu['correct_conformity'] = df_merged_extneu['neighbors'].map(correct_conformity)
df_merged_extneu['correct_anticonformity'] = df_merged_extneu['neighbors'].map(correct_anticonformity)

df_merged_extneu['correct_response_given_treatment'] = df_merged_extneu.apply(check_response_correct, axis=1)

df_merged_extneu['treatment_resemblance'] = df_merged_extneu.apply(check_response_treatment, axis=1)

df_merged_extneu[['response', 'correct_conformity', 'correct_anticonformity', 'treatment_resemblance']].head()

df_merged_extneu['initially_neutral'] = df_merged_extneu.groupby('participant.code')['response'].transform(
    lambda x: 'Neutral' if (x.iloc[0] == 0) else 'Non-Neutral'
)   


# How many initially non-neutral and neutral participants are there?
df_merged_extneu['initially_neutral'].value_counts()/ 10

## PLOT THE EXPECTED X NON-EXPECTED FOR INITIALLY NEUTRAL AND NON-NEUTRAL RESPONSES
# Group the data to calculate counts for each round and response correctness
# take only the rows with participant.treatment starting with C or A
df_merged_extneu_treatment = df_merged_extneu[df_merged_extneu['participant.treatment'].str.startswith(('C', 'A'))]

# create a new column for treatment names
df_merged_extneu_treatment['treatment'] = df_merged_extneu_treatment['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment')
)

# Treatment
df_merged_extneu_treatment_counts = df_merged_extneu_treatment.groupby(['round_no', 'correct_response_given_treatment', 'initially_neutral']).size().reset_index(name='count')

# Count the proportion by dividing by their respective counts of initially neutral and non-neutral participants
init_neutral_count = df_merged_extneu_treatment[df_merged_extneu_treatment['initially_neutral'] == 'Neutral']['participant.code'].nunique()
init_non_neutral_count = df_merged_extneu_treatment[df_merged_extneu_treatment['initially_neutral'] == 'Non-Neutral']['participant.code'].nunique()
df_merged_extneu_treatment_counts.loc[df_merged_extneu_treatment_counts['initially_neutral'] == 'Neutral', 'proportion'] = df_merged_extneu_treatment_counts['count'] / init_neutral_count
df_merged_extneu_treatment_counts.loc[df_merged_extneu_treatment_counts['initially_neutral'] == 'Non-Neutral', 'proportion'] = df_merged_extneu_treatment_counts['count'] / init_non_neutral_count

## COUNT PLOT
# Plot the count of expected vs non-expected responses as subplots for initially neutral and non-neutral participants
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

neutral_types = ['Neutral', 'Non-Neutral']
titles = ['Initially Neutral', 'Initially Non-Neutral']

# Custom label mapping
label_map = {'correct': 'Expected', 'incorrect': 'Not Expected', 'None': 'None'}
palette_map = {'Expected': 'limegreen', 'Not Expected': 'orange', 'None': 'gray'}

for i, init_type in enumerate(neutral_types):
    subset = df_merged_extneu_treatment_counts[df_merged_extneu_treatment_counts['initially_neutral'] == init_type].copy()
    # Map the labels for plotting
    subset['expected_label'] = subset['correct_response_given_treatment'].map(label_map)
    sns.lineplot(
        data=subset,
        x='round_no',
        y='count',
        hue='expected_label',
        style='expected_label',
        markers=True,
        dashes=True,
        palette=palette_map,
        errorbar=None,
        ax=axes[i]
    )
    axes[i].set_title(f'Expected vs Non-Expected: {titles[i]}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks(range(1, 11))
    axes[i].set_ylim(0, 40)
    if i == 1:
        axes[i].legend(title='Expected Response', loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        axes[i].get_legend().remove()

plt.tight_layout()
plt.savefig('Expected_vs_NonExpected_Responses_Initial_Neutral_Subplots.png', dpi=300)
plt.show()


## PROPORTION PLOT
# Plot the count of expected vs non-expected responses as subplots for initially neutral and non-neutral participants
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

neutral_types = ['Neutral', 'Non-Neutral']
titles = ['Initially Neutral', 'Initially Non-Neutral']

# Custom label mapping
label_map = {'correct': 'Expected', 'incorrect': 'Not Expected', 'None': 'None'}
palette_map = {'Expected': 'limegreen', 'Not Expected': 'orange', 'None': 'gray'}

for i, init_type in enumerate(neutral_types):
    subset = df_merged_extneu_treatment_counts[df_merged_extneu_treatment_counts['initially_neutral'] == init_type].copy()
    # Map the labels for plotting
    subset['expected_label'] = subset['correct_response_given_treatment'].map(label_map)
    sns.lineplot(
        data=subset,
        x='round_no',
        y='proportion',
        hue='expected_label',
        style='expected_label',
        markers=True,
        dashes=True,
        palette=palette_map,
        errorbar=None,
        ax=axes[i]
    )
    axes[i].set_title(f'Expected vs Non-Expected: {titles[i]}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks(range(1, 11))
    axes[i].set_ylim(0, 1)
    if i == 1:
        axes[i].legend(title='Expected Response', loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        axes[i].get_legend().remove()

plt.tight_layout()
plt.savefig('Expected_vs_NonExpected_Responses_Initial_Neutral_Subplots_Prop.png', dpi=300)
plt.show()


### PLOT FOR THE NO NUDGE TREATMENT
# Get the data
df = df_mock.copy()
participant_codes = df['participant.code'].unique()
len(participant_codes) # N=120

# Get the unique treatment types
treatments = ['C_n', 'C_p', 'AC_n', 'AC_p', 'NO_n', 'NO_p']

# Convert to categorical type for better plotting
df['participant.treatment'] = pd.Categorical(df['participant.treatment'], categories=treatments)

# Get the unique round numbers
round_numbers = df['round_no'].unique()

# For all treatments, plot the response differentiated by expected / non-expected responses (correct/incorrect)

# 3 steps:
# 1. Standardize the neighbors column
df['neighbors'] = df['neighbors'].apply(which_neighbors_combinations)
# 2. Create a new column for checking correctness of responses based on treatment and neighbors configs
df['correct_conformity'] = df['neighbors'].map(correct_conformity)
df['correct_anticonformity'] = df['neighbors'].map(correct_anticonformity)



# 3. Apply function to each row
df['correct_response_given_treatment'] = df.apply(check_response_correct, axis=1)
# Create a new column called 'conformity_response' for conformity responses and 'anticonformity_response' for anticonformity responses
df['treatment_resemblance'] = df.apply(check_response_treatment, axis=1)
df[['response', 'correct_conformity', 'correct_anticonformity', 'treatment_resemblance']].head()

# Convert the 'correct_response_given_treatment' to a categorical type 
df['treatment_resemblance'] = pd.Categorical(df['treatment_resemblance'],categories=['conformity', 'anticonformity', 'None', 'Unknown'])

df_notreatment = df.copy()
# Create a new column for treatment names
df_notreatment['treatment_type'] = df_notreatment['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment')
)

# Group the data to calculate counts for each round and response correctness
df_no_treatment_counts = df_notreatment.groupby(['round_no', 'treatment_resemblance', 'treatment_type']).size().reset_index(name='count')


# Remove the rows with Conformity or Anticonformity Treatment
df_no_treatment_counts = df_no_treatment_counts[df_no_treatment_counts['treatment_type'] == 'No Treatment']

# Convert counts to proportions by dividing by 40 (40 participants * 1 treatments)
df_no_treatment_counts['proportion'] = df_no_treatment_counts['count'] / 40


## COUNT PLOT
# Plot for No Treatment resemblance (all types) in a single plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_no_treatment_counts,
    x='round_no',
    y='count',
    hue='treatment_resemblance',
    style='treatment_resemblance',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    palette={'conformity': 'blue', 'anticonformity': 'red', 'None': 'gray', 'Unknown': 'purple'}
)

plt.title('No Treatment: Conformity/Anticonformity Resemblance')
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.ylim(0,40)
plt.xticks(range(1, 11))
plt.legend(title='Resemblance Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('No_Treatment_Resemblance_Count.png', dpi=300)
plt.show()


## PROPORTION PLOT
# Plot for No Treatment resemblance (all types) in a single plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_no_treatment_counts,
    x='round_no',
    y='proportion',
    hue='treatment_resemblance',
    style='treatment_resemblance',
    markers=True,  # Add markers for the dots
    dashes=True,   # Add dashed lines
    palette={'conformity': 'blue', 'anticonformity': 'red', 'None': 'gray', 'Unknown': 'purple'}
)

plt.title('No Treatment: Conformity/Anticonformity Resemblance')
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.ylim(0,1)
plt.xticks(range(1, 11))
plt.legend(title='Resemblance Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('No_Treatment_Resemblance_Proportion.png', dpi=300)
plt.show()











## PLOT THE STEPS CHANGE FROM ROUND 1 TO ROUND 10
# Get the data for both presurvey and mock part of the app
df = pd.concat([clean_df_1, clean_df_2, clean_df_3], ignore_index=True)

# Filter the presurvey and mock part of the app
df_presurvey = df[df['neighbors'].isna()]
df_mock = df[df['neighbors'].notna()]

# save as .csv
df_presurvey.to_csv('df_presurvey.csv', index=False)
df_mock.to_csv('df_mock.csv', index=False)

# Add a new row for round 0 by matching scenario_code with participant.scenario for the same participant.code
# For each participant, we will add a row with round_no = 0 with their response which has the same scenario_code as their first round in the mock part of the app
def add_round_0(df_presurvey, df_mock):

    # Create a dictionary to map participant.code to the participant.scenario
    participant_scenario_map = df_mock.set_index('participant.code')['participant.scenario'].to_dict()
    # Create a DataFrame for round 0
    round_0 = df_presurvey[['participant.code', 'scenario_code', 'response']].copy()

    # Find matching participant.code and scenario_code in the mock part of the app for the same participant using the participant_scenario_map and remove the non-matching rows
    round_0 = round_0[round_0.apply(lambda row: participant_scenario_map.get(row['participant.code'], None) == row['scenario_code'], axis=1)]

    # Add round_no = 0 to the DataFrame
    round_0['round_no'] = 0

    # Rename columns to match the original DataFrame
    round_0.rename(columns={'scenario_code': 'participant.scenario'}, inplace=True)
    
    # Merge the data based on participant.code and participant.scenario
    df_merged = pd.merge(df_mock, round_0[['participant.code', 'round_no', 'response']], 
                         on=['participant.code', 'round_no', 'response'], how='outer')   

    # Keep only the necessary columns
    df_merged = df_merged[['participant.code', 'round_no', 'participant.scenario', 'response', 'participant.treatment', 'neighbors']]

    # Sort by round_no
    df_merged = df_merged.sort_values(by=['participant.code', 'round_no']).reset_index(drop=True)

    
    return df_merged

df_merged = add_round_0(df_presurvey, df_mock)

# save as .csv
df_merged.to_csv('df_merged.csv', index=False)

# Copy values of participant.scenario from round_1 for the same participant.code
df_merged['participant.scenario'] = df_merged.groupby('participant.code')['participant.scenario'].transform(
    lambda x: x.ffill().bfill()  # Forward fill and backward fill to ensure all rounds have the same scenario_code
)

# Do the same for participant.treatment
df_merged['participant.treatment'] = df_merged.groupby('participant.code')['participant.treatment'].transform(
    lambda x: x.ffill().bfill()  # Forward fill and backward fill to ensure all rounds have the same treatment
)

df_merged['participant.code'].value_counts()
df_merged['participant.treatment'].value_counts()

# Copy df_merged 
df_steps = df_merged.copy()
# Create a new column for treatment names
df_steps['treatment_type'] = df_steps['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment'))

# Create a new column to track step changes from round 1 to round 10 round by round
# Calculate response change for each participant, round by round
df_steps['response_change'] = df_steps.groupby('participant.code')['response'].diff()
df_steps['response_delta'] = df_steps.groupby('participant.code')['response'].diff().abs()
# delete row where response_change is NaN (which is the first round for each participant)
df_steps = df_steps.dropna(subset=['response_change'])

# save as .csv
df_steps.to_csv('df_steps.csv', index=False)

# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(10, 6))
palette = {'Conformity': 'blue', 'Anticonformity': 'red', 'No Treatment': 'gray'}
sns.lineplot(
    data=df_steps,
    x='round_no',
    y='response_delta',
    hue='treatment_type',
    style='treatment_type',
    markers=True,  # Add markers for the dots
    dashes=True,   # Always use dashed lines for all treatment types
    palette=palette
)

plt.title('Average Response Change per Round by Treatment')
plt.xlabel('Round Number')
plt.ylabel('Average Delta [absolute change in response]')
plt.xticks(range(0, 11))
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('abs_response_change_per_round_by_treatment.png', dpi=300)
plt.show()


## Now pointplot with whiskers

# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(12, 6))
palette = {'Conformity': 'blue', 'Anticonformity': 'red', 'No Treatment': 'gray'}
sns.pointplot(
    data=df_steps,
    x='round_no',
    y='response_delta',
    hue='treatment_type',
    palette=palette,
    dodge=0.3,
    markers='o',
    linestyles=':',  # dots
    errorbar='se',  # Show standard error bars
    capsize=0.1
)
sns.stripplot(
    data=df_steps,
    x='round_no',
    y='response_delta',
    hue='treatment_type',
    palette=palette,
    dodge=True,
    jitter=True,
    alpha=0.1,
    marker='.',
    linewidth=0,
    legend=False  # Avoid duplicate legend
)

plt.title('Average Response Change per Round by Treatment')
plt.xlabel('Round Number')
plt.ylabel('Average Delta [absolute change in response]')
plt.xticks(range(0, 11))
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('abs_response_change_per_round_by_treatment_point.png', dpi=300)
plt.show()



# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(10, 6))
palette = {'Conformity': 'blue', 'Anticonformity': 'red', 'No Treatment': 'gray'}
sns.lineplot(
    data=df_steps,
    x='round_no',
    y='response_change',
    hue='treatment_type',
    style='treatment_type',
    markers=True,  # Add markers for the dots
    dashes=True,   # Always use dashed lines for all treatment types
    palette=palette
)
plt.title('Average Response Change per Round by Treatment')
plt.xlabel('Round Number')
plt.ylabel('Average Response Change [Round r+1 - Round r]')
plt.xticks(range(1, 11))
plt.ylim(-1,1)
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('rel_response_change_per_round_by_treatment.png', dpi=300)
plt.show()



# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(12, 6))
palette = {'Conformity': 'blue', 'Anticonformity': 'red', 'No Treatment': 'gray'}
sns.pointplot(
    data=df_steps,
    x='round_no',
    y='response_change',
    hue='treatment_type',
    palette=palette,
    dodge=0.3,
    markers='o',
    linestyles=':',  # dots
    errorbar='se',  # Show standard error bars
    capsize=0.1
)
sns.stripplot(
    data=df_steps,
    x='round_no',
    y='response_change',
    hue='treatment_type',
    palette=palette,
    dodge=True,
    jitter=True,
    alpha=0.1,
    marker='.',
    linewidth=0,
    legend=False  # Avoid duplicate legend
)

plt.title('Average Response Change per Round by Treatment')
plt.xlabel('Round Number')
plt.ylabel('Average Response Change [Round r+1 - Round r]')
plt.xticks(range(0, 11))
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('_response_change_per_round_by_treatment_point.png', dpi=300)
plt.show()





# Make the same figures but create subpanels for different participant.scenario

# Get unique scenarios
scenarios = ['s2_n', 's4_n', 's9_n', 's2_p', 's4_p', 's9_p']  

# Convert scenarios to categorical type and sort them
df_steps['participant.scenario'] = pd.Categorical(df_steps['participant.scenario'], categories=scenarios, ordered=True)
# Set up subplots: 2 rows x 3 columns (adjust if you have more/fewer scenarios)
n_scenarios = len(scenarios)
ncols = 3
nrows = int(np.ceil(n_scenarios / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True)
axes = axes.flatten()

palette = {'Conformity': 'blue', 'Anticonformity': 'red', 'No Treatment': 'gray'}

for i, scenario in enumerate(scenarios):
    scenario_data = df_steps[df_steps['participant.scenario'] == scenario]
    sns.lineplot(
        data=scenario_data,
        x='round_no',
        y='response_delta',
        hue='treatment_type',
        style='treatment_type',
        markers=True,
        dashes=True,
        palette=palette,
        ax=axes[i]
    )
    axes[i].set_title(f'Scenario {scenario}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Absolute Change')
    axes[i].set_xticks(range(1, 11))
    # Remove legend for all but the last subplot
    if i != n_scenarios - 1:
        axes[i].get_legend().remove()

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Only show legend on the last subplot
handles, labels = axes[n_scenarios - 1].get_legend_handles_labels()
axes[n_scenarios - 1].legend(handles=handles, labels=labels, title='Treatment', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('abs_response_change_per_round_by_treatment_scenario.png', dpi=300)
plt.show()

# Repeat for relative response change
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True)
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    scenario_data = df_steps[df_steps['participant.scenario'] == scenario]
    sns.lineplot(
        data=scenario_data,
        x='round_no',
        y='response_change',
        hue='treatment_type',
        style='treatment_type',
        markers=True,
        dashes=True,
        palette=palette,
        ax=axes[i]
    )
    axes[i].set_title(f'Scenario {scenario}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Response Change')
    axes[i].set_xticks(range(1, 11))
    axes[i].set_ylim(-1, 1)
    # Remove legend for all but the last subplot
    if i != n_scenarios - 1:
        axes[i].get_legend().remove()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Only show legend on the last subplot
handles, labels = axes[n_scenarios - 1].get_legend_handles_labels()
axes[n_scenarios - 1].legend(handles=handles, labels=labels, title='Treatment', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('rel_response_change_per_round_by_treatment_scenario.png', dpi=300)
plt.show()



# Distinguish initially neutral participants and non-neutral participants from round 0

df_steps_extneu = df_merged.copy()
# Create a new column to distinguish initially neutral participants
df_steps_extneu['initially_neutral'] = df_steps_extneu.groupby('participant.code')['response'].transform(
    lambda x: 'Neutral' if (x.iloc[0] == 0) else 'Non-Neutral'
)   

# Create the same figures but with the distinction of initially neutral and non-neutral participants
# Calculate response change for each participant, round by round
df_steps_extneu['response_change'] = df_steps_extneu.groupby('participant.code')['response'].diff()
df_steps_extneu['response_delta'] = df_steps_extneu.groupby('participant.code')['response'].diff().abs()
# delete row where response_change is NaN (which is the first round for each participant)
df_steps_extneu = df_steps_extneu.dropna(subset=['response_change'])

# How many initially non-neutral and neutral participants are there?
df_steps_extneu['initially_neutral'].value_counts()/ 10
# 81/120 = 67.5% of participants are initially non-neutral, 39/120 = 31.32.5% are initially neutral

# save df_steps_extneu as .csv
df_steps_extneu.to_csv('df_steps_extneu.csv', index=False)

# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(10, 6))
palette = {'Neutral': "#A88A9A", 'Non-Neutral':"#d41286" ,}
sns.lineplot(
    data=df_steps_extneu,
    x='round_no',
    y='response_delta',
    hue='initially_neutral',
    style='initially_neutral',
    markers=True,  # Add markers for the dots
    dashes=True,   # Always use dashed lines for all treatment types
    palette=palette
)
plt.title('Average Response Change per Round by Initial Opinion')
plt.xlabel('Round Number')
plt.ylabel('Average Delta [absolute change in response]')
plt.xticks(range(1, 11))
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('abs_response_change_per_round_by_ext.png', dpi=300)
plt.show()

# Plot the response change from round 1 to round 10 for all treatments, anticonformity and conformity
plt.figure(figsize=(10, 6))
palette = {'Neutral': '#A88A9A', 'Non-Neutral':'#d41286' ,}
sns.lineplot(
    data=df_steps_extneu,
    x='round_no',
    y='response_change',
    hue='initially_neutral',
    style='initially_neutral',
    markers=True,  # Add markers for the dots
    dashes=True,   # Always use dashed lines for all treatment types
    palette=palette
)
plt.title('Average Response Change per Round by Initial Opinion')
plt.xlabel('Round Number')
plt.ylabel('Average Response Change [Round r+1 - Round r]')
plt.xticks(range(1, 11))
plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig('rel_response_change_per_round_by_ext.png', dpi=300)
plt.show()

# Create a new column for treatment names
df_steps_extneu['treatment_type'] = df_steps_extneu['participant.treatment'].apply(
    lambda x: 'Conformity' if x.startswith('C') else ('Anticonformity' if x.startswith('A') else 'No Treatment'))

# Convert treatment_type to categorical type for better plotting
df_steps_extneu['treatment_type'] = pd.Categorical(df_steps_extneu['treatment_type'], 
                                                    categories=['Conformity', 'Anticonformity', 'No Treatment'],
                                                    ordered=True)


# Prepare No Treatment data for overlay
no_treatment_data = df_steps_extneu[df_steps_extneu['treatment_type'] == 'No Treatment']

palette = {'Neutral': '#A88A9A', 'Non-Neutral':'#d41286' ,}
treatment_types = ['Conformity' , 'Anticonformity' , 'No Treatment']
n_types = len(treatment_types)
fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 5), sharex=True, sharey=True)
if n_types == 1:
    axes = [axes]  # Ensure axes is always iterable

for i, t_type in enumerate(treatment_types):
    data = df_steps_extneu[df_steps_extneu['treatment_type'] == t_type]
    # Use lineplot with dashes for all lines
    sns.lineplot(
        data=data,
        x='round_no',
        y='response_delta',
        hue='initially_neutral',
        style='initially_neutral',
        palette=palette,
        markers=True,
        dashes=[(2,2), (2,2)],  # force all lines to be dashed
        ax=axes[i]
    )
    
    # Overlay No Treatment lines (same color/style for all subplots)
    if not no_treatment_data.empty:
        palette2 = {'Neutral': "#53E3E8", 'Non-Neutral':"#14908E" ,}
        for init_neu, color in palette2.items():
            nt_subset = no_treatment_data[no_treatment_data['initially_neutral'] == init_neu]
            if not nt_subset.empty:
                means = nt_subset.groupby('round_no')['response_delta'].mean()
                axes[i].plot(
                    means.index, means.values,
                    label=f'No Treatment ({init_neu})',
                    color=color,
                    linestyle='-',  # full line
                    marker='x',     # add marker here
                    linewidth=2,
                    alpha=0.7,
                    zorder=0
                )

    axes[i].set_title(f'Treatment: {t_type}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Absolute Change')
    axes[i].set_xticks(range(1, 11))
    # Only show legend for the last subplot
    if i != n_types - 1:
        axes[i].get_legend().remove()
    else:
        axes[i].legend(title='Initial Opinion', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('abs_overlay_response_change_per_round_by_ext_treatment.png', dpi=300)
#plt.savefig('abs_response_change_per_round_by_ext_treatment.png', dpi=300)
plt.show()

# Prepare No Treatment data for overlay
no_treatment_data = df_steps_extneu[df_steps_extneu['treatment_type'] == 'No Treatment']

palette = {'Neutral': '#A88A9A', 'Non-Neutral':'#d41286' ,}
fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 5), sharex=True, sharey=True)
for i, t_type in enumerate(treatment_types):
    data = df_steps_extneu[df_steps_extneu['treatment_type'] == t_type]
    sns.lineplot(
        data=data,
        x='round_no',
        y='response_change',
        hue='initially_neutral',
        style='initially_neutral',
        palette=palette,
        markers=True,
        dashes=True,
        ax=axes[i]
    )
    # Overlay No Treatment lines (same color/style for all subplots)
    if not no_treatment_data.empty:
        palette2 = {'Neutral': "#53E3E8", 'Non-Neutral':"#14908E" ,}
        for init_neu, color in palette2.items():
            nt_subset = no_treatment_data[no_treatment_data['initially_neutral'] == init_neu]
            if not nt_subset.empty:
                means = nt_subset.groupby('round_no')['response_change'].mean()
                axes[i].plot(
                    means.index, means.values,
                    label=f'No Treatment ({init_neu})',
                    color=color,
                    linestyle='-',  # full line
                    marker='x',     # add marker here
                    linewidth=2,
                    alpha=0.7,
                    zorder=0
                )

    axes[i].set_title(f'Treatment: {t_type}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Opinion Change [Round r+1 - Round r]')
    axes[i].set_xticks(range(1, 11))
    axes[i].legend(title='Initial Opinion', loc='upper left', bbox_to_anchor=(1.05, 1))
    # Only show legend for the last subplot
    if i != n_types - 1:
        axes[i].get_legend().remove()
    else:
        axes[i].legend(title='Initial Opinion', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('rel_overlay_response_change_per_round_by_ext_treatment.png', dpi=300)
#plt.savefig('rel_response_change_per_round_by_ext_treatment.png', dpi=300)
plt.show()


## POINT PLOT
# Prepare No Treatment data for overlay (average across both Neutral and Non-Neutral)
no_treatment_data = df_steps_extneu[df_steps_extneu['treatment_type'] == 'No Treatment']

palette = {'Neutral': '#A88A9A', 'Non-Neutral':'#d41286' ,}
treatment_types = ['Conformity', 'Anticonformity', 'No Treatment']
n_types = len(treatment_types)
fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 5), sharex=True, sharey=True)
if n_types == 1:
    axes = [axes]  # Ensure axes is always iterable

# Compute average No Treatment overlay for Neutral and Non-Neutral
overlay_means = no_treatment_data.groupby(['round_no'])['response_delta'].mean().reset_index()

# Ensure `round_no` is an integer and includes all rounds from 1 to 10
overlay_means['round_no'] = overlay_means['round_no'].astype(int)
overlay_means = overlay_means.set_index('round_no').reindex(range(1, 11)).reset_index()
overlay_means['response_delta'] = overlay_means['response_delta'].fillna(0)  # Fill missing values with 0

for i, t_type in enumerate(treatment_types):
    data = df_steps_extneu[df_steps_extneu['treatment_type'] == t_type]
    # Use pointplot for means and error bars
    sns.pointplot(
        data=data,
        x='round_no',
        y='response_delta',
        hue='initially_neutral',
        palette=palette,
        dodge=0.3,
        markers='o',
        linestyles=':',  # dotted lines
        errorbar='se',
        capsize=0.1,
        ax=axes[i]
    )
    # # Overlay average No Treatment lines for Neutral and Non-Neutral
    # if not overlay_means.empty:
    #     axes[i].plot(
    #         overlay_means['round_no'], overlay_means['response_delta'],
    #         label=f'No Treatment Avg',
    #         color="#53E3E8",
    #         linestyle='-',
    #         marker='x',
    #         linewidth=2,
    #         alpha=0.7,
    #         #zorder=0
    #     )


    axes[i].set_title(f'Treatment: {t_type}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Absolute Change')
    axes[i].set_xticks(range(1, 11))
    # Only show legend for the last subplot
    if i != n_types - 1:
        axes[i].get_legend().remove()
    else:
        axes[i].legend(title='Initial Opinion', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig('abs_overlay_response_change_per_round_by_ext_treatment_point.png', dpi=300)
plt.show()


## POINT PLOT: Relative Change
#TODO fix the round number for the overlay line
# Prepare No Treatment data for overlay (average across both Neutral and Non-Neutral)
no_treatment_data = df_steps_extneu[df_steps_extneu['treatment_type'] == 'No Treatment']

treatment_types = ['Conformity', 'Anticonformity', 'No Treatment']
n_types = len(treatment_types)
fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 5), sharex=True, sharey=True)
if n_types == 1:
    axes = [axes]  # Ensure axes is always iterable

# Compute average No Treatment overlay for Neutral and Non-Neutral
overlay_means = no_treatment_data.groupby(['round_no'])['response_change'].mean().reset_index()

for i, t_type in enumerate(treatment_types):
    data = df_steps_extneu[df_steps_extneu['treatment_type'] == t_type]
    # Use pointplot for means and error bars
    sns.pointplot(
        data=data,
        x='round_no',
        y='response_change',
        hue='initially_neutral',
        palette=palette,
        dodge=0.3,
        markers='o',
        linestyles=':',  # dotted lines
        errorbar='se',
        capsize=0.1,
        ax=axes[i]
    )
    # # Overlay average No Treatment lines for Neutral and Non-Neutral
    # means = overlay_means
    # if not means.empty:
    #     axes[i].plot(
    #         means['round_no'], means['response_change'],
    #         label=f'No Treatment Avg',
    #         color="#53E3E8",
    #         linestyle='-',
    #         marker='x',
    #         linewidth=2,
    #         alpha=0.7,
    #         zorder=0
    #     )

    axes[i].set_title(f'Treatment: {t_type}')
    axes[i].set_xlabel('Round Number')
    axes[i].set_ylabel('Avg. Relative Change')
    axes[i].set_xticks(range(1, 11))
    # Only show legend for the last subplot
    if i != n_types - 1:
        axes[i].get_legend().remove()
    else:
        axes[i].legend(title='Initial Opinion', loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig('rel_overlay_response_change_per_round_by_ext_treatment_point.png', dpi=300)
plt.show()
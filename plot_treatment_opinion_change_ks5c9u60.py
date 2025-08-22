import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 
import seaborn as sns       
import ast 
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

print(os.getcwd())
session_id = 'ks5c9u60'
# Load and merge the page time data
# windows 
file_location = '/Users/Lasmi Marbun/Documents/Git/Anticonformity-Analysis/'
# mac
#file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/'

# Load the three cleaned DataFrames
df = pd.read_csv(file_location + 'Clean_files/clean_long_format_' + session_id + '.csv')

participant_label = pd.DataFrame(df['participant.label'].unique(), columns=['participant.label'])
# save as csv
participant_label.to_csv(file_location + 'participant_label_e1314ij0.csv', index=False)

participant_code = df['participant.code'].unique().tolist()
len(participant_code) # 8 participants

## MOCK PART OF THE APP

# Filter only relevant rows in the mock part of the app; indicated by political_charge is NaN
df_mock = df[df['political_charge'].isna()]

def add_mock_response_column(df_mock):
    """
    Adds a 'mock.response' column to df_mock:
    - For round_no=0: value is old_response from round_no=1 for each participant.
    - For round_no=1..20: value is new_response from that round.
    Returns a new DataFrame with round_no=0 rows added.
    """
    # Ensure correct sorting
    df_mock = df_mock.sort_values(['participant.code', 'round_no']).copy()
    
    # 1. For rounds 1..20, set mock.response = new_response
    df_mock['mock.response'] = df_mock['new_response']
    
    # 2. For round 1, get old_response for each participant
    round1 = df_mock[df_mock['round_no'] == 1].copy()
    round0 = round1.copy()
    round0['round_no'] = 0
    round0['mock.response'] = round0['old_response']
    
    # Remove discussion_grp and old_response and new_response for round_no=0
    round0 = round0.drop(columns=['discussion_grp', 'old_response', 'new_response', 'forced_response'])

    # 3. Combine round 0 and the rest
    df_with_round0 = pd.concat([df_mock, round0], ignore_index=True)
    df_with_round0 = df_with_round0.sort_values(['participant.code', 'round_no']).reset_index(drop=True)
    
    return df_with_round0

# Apply to df_mock
df_mock_complete = add_mock_response_column(df_mock)

df_mock_relevant = df_mock_complete[['participant.code', 'mock.response', 'round_no', 'discussion_grp', 'forced_response', 'participant.scenario', 'participant.anticonformist', 'participant.group_size']]

def get_neighbors_responses(row):
    if pd.isna(row['discussion_grp']):
        return np.nan
    codes = ast.literal_eval(row['discussion_grp'])
    responses = []
    for code in codes:
        match = df_mock_relevant.loc[
            (df_mock_relevant['round_no'] == row['round_no']) &
            (df_mock_relevant['participant.code'] == code),
            'mock.response'
        ]
        if not match.empty:
            responses.append(match.iloc[0])
    return responses

df_mock_relevant['neighbors_response'] = df_mock_relevant.apply(get_neighbors_responses, axis=1)

## Standardize column names and values
# convert anticonformist == 1 to 'anticonformist' and 0 to 'conformist'
df_mock_relevant['participant.treatment'] = df_mock_relevant['participant.anticonformist'].replace({1: 'Anticonformist', 0: 'Conformist'})
# convert group_size == N08 to beta = 0.5 in a new column
df_mock_relevant['participant.beta'] = df_mock_relevant['participant.group_size'].replace({'N08': 0.5})
# change mock.response into response
df_mock_relevant.rename(columns={'mock.response': 'response'}, inplace=True)

# convert list into string
df_mock_relevant['neighbors'] = df_mock_relevant['neighbors_response'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else '')
# convert to string
df_mock_relevant['neighbors'] = df_mock_relevant['neighbors'].astype(str)

# save as .csv to see
df_mock_relevant.to_csv(file_location + 'mock_relevant_' + session_id + '.csv', index=False)

## Neighbor sets
# Load the neighbor configurations
neighbors_configs = pd.read_csv(file_location + 'neighbors_configurations.csv')

# ignore index
neighbors_configs.reset_index(drop=True, inplace=True)


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
    neighbors = row['neighbors_std']
    response_str = str(int(row['response'])) if isinstance(row['response'], float) and row['response'] == int(row['response']) else str(row['response'])
    correct_both_val = correct_both.get(neighbors, None)
    if correct_both_val is None:
        return 'Unknown'
    elif correct_both_val == 'None' or response_str == 'None':
        return 'None'
    elif isinstance(correct_both_val, list):
        if response_str not in correct_both_val:
            return 'Unknown'
    # else:
    #     if response_str != correct_both_val:
    #         return 'Unknown'
    # Now check conformity first
    correct_conf = correct_conformity.get(neighbors, None)
    if correct_conf == 'None':
        return 'None'
    elif isinstance(correct_conf, list):
        if response_str in correct_conf:
            return 'conformity'
    else:
        if response_str == correct_conf:
            return 'conformity'
    # If not conformity, check anticonformity
    correct_anti = correct_anticonformity.get(neighbors, None)
    if correct_anti == 'None':
        return 'None'
    elif isinstance(correct_anti, list):
        if response_str in correct_anti:
            return 'anticonformity'
    else:
        if response_str == correct_anti:
            return 'anticonformity'
    return False

# Function to convert neighbors
def normalize_neighbors(val):
    if pd.isna(val):
        return val
    # If it's already a list/tuple
    if isinstance(val, (list, tuple)):
        parts = val
    else:
        s = str(val).strip()
        # handle string-representation of a list: "['1.0', '-1.0']"
        if s.startswith('[') and s.endswith(']'):
            try:
                parts = ast.literal_eval(s)
            except Exception:
                parts = s.strip('[]').split(',')
        else:
            parts = s.split(',')
    out = []
    for p in parts:
        p = str(p).strip()
        if p == '' or p.lower() == 'nan':
            continue
        try:
            f = float(p)
            if f.is_integer():
                out.append(str(int(f)))
            else:
                # remove trailing zeros if any
                out.append(str(f).rstrip('0').rstrip('.'))
        except Exception:
            out.append(p)  # fallback: keep as-is
    return ','.join(out)

# Apply normalization to the neighbors column
df_mock_relevant['neighbors_1'] = df_mock_relevant['neighbors'].apply(normalize_neighbors)
# Function to standardize the neighbors combinations (permutation to combination)
def which_neighbors_combinations(neighbor_input):
    if pd.isna(neighbor_input) or (isinstance(neighbor_input, str) and neighbor_input.strip() == ''):
        return 'None'
    # If input is a Series or list, process each element and return a list of standardized strings
    if isinstance(neighbor_input, (pd.Series, list)):
        result = []
        for neighbor_str in neighbor_input:
            if pd.isna(neighbor_str) or (isinstance(neighbor_str, str) and neighbor_str.strip() == ''):
                result.append('None')
                continue
            try:
                nums = [int(x.strip()) for x in str(neighbor_str).split(',') if x.strip()]
                result.append(','.join(map(str, sorted(nums))))
            except ValueError:
                result.append('Invalid')  # Handle invalid entries
        return result
    # If input is a single string
    try:
        nums = [int(x.strip()) for x in str(neighbor_input).split(',') if x.strip()]
        return ','.join(map(str, sorted(nums)))
    except ValueError:
        return 'Invalid'  # Handle invalid input


## TESTING ##
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


# Apply normalization to the neighbors column
df_mock_relevant['neighbors'] = df_mock_relevant['neighbors'].apply(normalize_neighbors)
# Apply standardization 
df_mock_relevant['neighbors_std'] = df_mock_relevant['neighbors'].apply(which_neighbors_combinations)

# save as .csv to see
df_mock_relevant.to_csv(file_location + 'mock_relevant_' + session_id + '.csv', index=False)


#  read csv
df_mock_relevant = pd.read_csv('mock_relevant_'+ session_id+ '.csv')
df_mock_relevant['correct_conformity'] = df_mock_relevant['neighbors_std'].map(correct_conformity)
df_mock_relevant['correct_anticonformity'] = df_mock_relevant['neighbors_std'].map(correct_anticonformity)
df_mock_relevant['treatment_resemblance'] = df_mock_relevant.apply(check_response_treatment, axis=1)
df_mock_relevant['correct_response_given_treatment'] = df_mock_relevant.apply(check_response_correct, axis=1)
# count
df_mock_relevant['treatment_resemblance'].value_counts()


# save as .csv to include the new columns
df_mock_relevant.to_csv(file_location + 'mock_complete_' + session_id + '.csv', index=False)

### CORRECT OR INCORRECT RESPONSE PLOT (CONFORMITY AND ANTICONFORMITY) Note: only for round 10

# Plot barplot to count how many people get the correct response in round 10
# 2 plots: one for conformity and one for anticonformity

# Create a DataFrame for conformity responses
conformity_df = df_mock_relevant[df_mock_relevant['participant.treatment'].str.startswith('C')]

conformity_df_round_20 = conformity_df[conformity_df['round_no'] == 20]
# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
conformity_df['correct_response_given_treatment'] = pd.Categorical(conformity_df['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'],)
conformity_counts = conformity_df['correct_response_given_treatment'].value_counts().reset_index()
conformity_counts.columns = ['response', 'count']   


# Create a bar plot for conformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=conformity_counts, x='response', y='count', palette    ='viridis')                 
plt.title('Conformity Responses over all rounds '+ session_id)
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 200)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('conformity_responses_'+session_id +'.png', dpi=300)
plt.show()


# Visualize the last round (round 20)
conformity_df_round_20['correct_response_given_treatment'] = pd.Categorical(conformity_df_round_20['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'], ordered=True)
conformity_counts_20 = conformity_df_round_20['correct_response_given_treatment'].value_counts().reset_index()
conformity_counts_20.columns = ['response', 'count']

# Create a bar plot for conformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=conformity_counts_20, x='response', y='count', palette    ='viridis')                 
plt.title('Conformity Responses in Round 20')
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('conformity_responses_round20_' + session_id + '.png', dpi=300)
plt.show()


## Plot conformity counts over the rounds
conformity_df['round_no'] = conformity_df['round_no'].astype(float)
conformity_counts_over_rounds = conformity_df.groupby('round_no')['correct_response_given_treatment'].value_counts().unstack(fill_value=0)
# remove round 0 bcs it doesnt have neighbors yet
conformity_counts_over_rounds = conformity_counts_over_rounds.drop(index=0, errors='ignore')

## Plot conformity counts over the rounds
plt.figure(figsize=(12, 6))
markers = {'correct': 'o', 'incorrect': 'x', 'None': 's'}
for response_type in conformity_counts_over_rounds.columns:
    sns.lineplot(
        data=conformity_counts_over_rounds[response_type],
        label=response_type,
        linestyle='--',
        marker=markers.get(response_type, 'o')
    )
plt.title('Conformity Responses over Rounds ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(1, 21), rotation=45)
plt.tight_layout()
plt.savefig('conformity_responses_over_rounds_' + session_id + '.png', dpi=300)
plt.show()

### different visualizations as a stacked bar chart
conformity_df['round_no'] = conformity_df['round_no'].astype(int)
conformity_counts_over_rounds = conformity_df.groupby('round_no')['correct_response_given_treatment'].value_counts().unstack(fill_value=0)
# remove round 0 because it doesn't have neighbors yet
conformity_counts_over_rounds = conformity_counts_over_rounds.drop(index=0, errors='ignore')

# set color palette
colors = sns.color_palette("Blues", 3)
## Plot conformity counts over the rounds as a stacked bar chart
plt.figure(figsize=(12, 6))
ax = conformity_counts_over_rounds.plot(
    kind='bar',
    stacked=True,
    color=['#1f77b4', '#ff7f0e', '#2ca02c'],  # Colors for 'correct', 'incorrect', 'None'
    figsize=(12, 6)
)
plt.title('Conformity Responses over Rounds (Stacked) ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(0, 21), rotation=45)
# Move legend outside the plot
plt.legend(
    title='Response Type',
    labels=['Correct', 'Incorrect', 'None'],
    bbox_to_anchor=(1.05, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('conformity_responses_over_rounds_stacked_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()


# Check manually by getting the relevant columns 'neighbors', 'response', 'neighbors_std', 'correct_anticonformity', 'correct_conformity', 'correct_response_given_treatment', 'treatment_resemblance', 'round_no'
manual_check = conformity_df[[
    'neighbors', 'response', 'neighbors_std', 'correct_anticonformity', 'correct_conformity', 'correct_response_given_treatment', 'treatment_resemblance', 'round_no'
]]
# looks legit
# Plot the 6 participants conformist to see over the rounds
conformity_df['participant.code'].unique() # 6 participants


# make categories for correct_response_given_treatment starting with 'correct', 'None', 'incorrect'
conformity_df['correct_response_given_treatment'] = pd.Categorical(conformity_df['correct_response_given_treatment'],categories=['correct', 'None', 'incorrect'],)

# remove round 0 because there is no neighbors yet
conformity_plot = conformity_df[conformity_df['round_no'] > 0]

# Create the plot for conformists
plt.figure(figsize=(20, 4))
# set color palette
colors = sns.color_palette("husl", 6)
markers = ['o', 's', 'D', '^', 'v', 'P']  # Different markers for each participant
for i, participant in enumerate(conformity_plot['participant.code'].unique()):
    participant_data = conformity_plot[conformity_plot['participant.code'] == participant]
    sns.lineplot(
        data=participant_data,
        x='round_no',
        y='correct_response_given_treatment',
        label=f'Participant {participant}',
        linestyle='--',
        marker=markers[i % len(markers)],  # Cycle through markers
        color=colors[i]
    )
plt.title('Individual Conformists Responses over Rounds ' + session_id)
plt.xlabel('Round Number')
plt.xticks(range(1, 21))
plt.ylabel('Response')
plt.legend(
    title='Participant code',
    bbox_to_anchor=(1.02, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('conformity_individual_responses_over_rounds_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()



### Stacked bar chart but for treatment resemblance
conformity_df['round_no'] = conformity_df['round_no'].astype(int)
conformity_counts_over_rounds = conformity_df.groupby('round_no')['treatment_resemblance'].value_counts().unstack(fill_value=0)
# remove round 0 because it doesn't have neighbors yet
conformity_counts_over_rounds = conformity_counts_over_rounds.drop(index=0, errors='ignore')

conformity_df['treatment_resemblance'].value_counts()
## Plot anticonformity counts over the rounds as a stacked bar chart
plt.figure(figsize=(12, 6))
ax = conformity_counts_over_rounds.plot(
    kind='bar',
    stacked=True,
    color=[ "#8f8f8f", "#e9a5e3","#c92121", "#0551ea"],  # Colors for 'correct', 'incorrect', 'None'
    figsize=(12, 6)
)
plt.title('Conformists Resemblance over Rounds (Stacked) ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(0, 21), rotation=45)
plt.yticks(ticks=range(0, 3))
# Move legend outside the plot
plt.legend(
    title='Response Type',
    labels=['None', 'Unknown', 'anticonformity', 'conformity'],
    bbox_to_anchor=(1.05, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('conformity_resemblance_over_rounds_stacked_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()



### Anticonformity ###

# Create a DataFrame for anticonformity responses
anticonformity_df = df_mock_relevant[df_mock_relevant['participant.treatment'].str.startswith('A')]

anticonformity_df_round_20 = anticonformity_df[anticonformity_df['round_no'] == 20]
# Convert the 'correct_response_given_treatment' to a categorical type for better plotting
anticonformity_df['correct_response_given_treatment'] = pd.Categorical(anticonformity_df['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'],)
anticonformity_counts = anticonformity_df['correct_response_given_treatment'].value_counts().reset_index()
anticonformity_counts.columns = ['response', 'count']


# Create a bar plot for conformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=anticonformity_counts, x='response', y='count', palette    ='viridis')                 
plt.title('Conformity Responses over all rounds ' + session_id)
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 200)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anticonformity_responses_' + session_id + '.png', dpi=300)
plt.show()


# Visualize the last round (round 20)
anticonformity_df_round_20['correct_response_given_treatment'] = pd.Categorical(anticonformity_df_round_20['correct_response_given_treatment'],
                                                                    categories=['correct', 'incorrect', 'None'], ordered=True)
anticonformity_counts_20 = anticonformity_df_round_20['correct_response_given_treatment'].value_counts().reset_index()
anticonformity_counts_20.columns = ['response', 'count']

# Create a bar plot for conformity responses
plt.figure(figsize=(8, 6))
sns.barplot(data=anticonformity_counts_20, x='response', y='count', palette    ='viridis')                 
plt.title('Anticonformity Responses in Round 20 ' + session_id)
plt.xlabel('Response')
plt.ylabel('Count')
plt.ylim(0, 8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anticonformity_responses_round20_' + session_id + '.png', dpi=300)
plt.show()


## Plot anticonformity counts over the rounds
anticonformity_df['round_no'] = anticonformity_df['round_no'].astype(float)
anticonformity_counts_over_rounds = anticonformity_df.groupby('round_no')['correct_response_given_treatment'].value_counts().unstack(fill_value=0)
# remove round 0 bcs it doesnt have neighbors yet
anticonformity_counts_over_rounds = anticonformity_counts_over_rounds.drop(index=0, errors='ignore')

## Plot conformity counts over the rounds
plt.figure(figsize=(12, 6))
markers = {'correct': 'o', 'incorrect': 'x', 'None': 's'}
for response_type in anticonformity_counts_over_rounds.columns:
    sns.lineplot(
        data=anticonformity_counts_over_rounds[response_type],
        label=response_type,
        linestyle='--',
        marker=markers.get(response_type, 'o')
    )
plt.title('Anticonformity Responses over Rounds ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(1, 21), rotation=45)
plt.yticks(ticks=range(0, 3))
plt.tight_layout()
plt.savefig('anticonformity_responses_over_rounds_' + session_id + '.png', dpi=300)
plt.show()

### different visualizations as a stacked bar chart
anticonformity_df['round_no'] = anticonformity_df['round_no'].astype(int)
anticonformity_counts_over_rounds = anticonformity_df.groupby('round_no')['correct_response_given_treatment'].value_counts().unstack(fill_value=0)
# remove round 0 because it doesn't have neighbors yet
anticonformity_counts_over_rounds = anticonformity_counts_over_rounds.drop(index=0, errors='ignore')

## Plot anticonformity counts over the rounds as a stacked bar chart
plt.figure(figsize=(12, 6))
ax = anticonformity_counts_over_rounds.plot(
    kind='bar',
    stacked=True,
    color=['#1f77b4', '#ff7f0e', '#2ca02c'],  # Colors for 'correct', 'incorrect', 'None'
    figsize=(12, 6)
)
plt.title('Anticonformity Responses over Rounds (Stacked) ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(0, 21), rotation=45)
plt.yticks(ticks=range(0, 3))
# Move legend outside the plot
plt.legend(
    title='Response Type',
    labels=['Correct', 'Incorrect', 'None'],
    bbox_to_anchor=(1.05, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('anticonformity_responses_over_rounds_stacked_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()


# Check manually by getting the relevant columns 'neighbors', 'response', 'neighbors_std', 'correct_anticonformity', 'correct_conformity', 'correct_response_given_treatment', 'treatment_resemblance', 'round_no'
manual_check = anticonformity_df[[
    'neighbors', 'response', 'neighbors_std', 'correct_anticonformity', 'correct_conformity', 'correct_response_given_treatment', 'treatment_resemblance', 'round_no'
]]
# looks legit

## Plot the 2 participants anticonformist to see over the rounds
anticonformity_df['participant.code'].unique() # 4 participants because p = .50


# make categories for correct_response_given_treatment starting with 'correct', 'None', 'incorrect'
anticonformity_df['correct_response_given_treatment'] = pd.Categorical(anticonformity_df['correct_response_given_treatment'],
                                                                    categories=['correct', 'None', 'incorrect'],)

# Create the plot for anticonformists
plt.figure(figsize=(20, 3))
# remove round 0 because there is no neighbors yet
anticonformity_plot = anticonformity_df[anticonformity_df['round_no'] > 0]
# set color palette
colors = ['#3b1717', '#a50e0e', "#ec4848", "#D4B1C4"]
markers = ['o', 's', 'D', '^']  # 4 marker types
for i, participant in enumerate(anticonformity_plot['participant.code'].unique()):
    participant_data = anticonformity_plot[anticonformity_plot['participant.code'] == participant]
    sns.lineplot(
        data=participant_data,
        x='round_no',
        y='correct_response_given_treatment',
        label=f'Participant {participant}',
        linestyle='--',
        marker=markers[i % len(markers)],
        color=colors[i % len(colors)]
    )
plt.title('Individual Anticonformists Responses over Rounds ' + session_id)
plt.xlabel('Round Number')
plt.xticks(range(1, 21))
plt.ylabel('Response')
plt.legend(
    title='Participant code',
    bbox_to_anchor=(1.02, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('anticonformity_individual_responses_over_rounds_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()


### Stacked bar chart but for treatment resemblance
anticonformity_df['round_no'] = anticonformity_df['round_no'].astype(int)
anticonformity_counts_over_rounds = anticonformity_df.groupby('round_no')['treatment_resemblance'].value_counts().unstack(fill_value=0)
# remove round 0 because it doesn't have neighbors yet
anticonformity_counts_over_rounds = anticonformity_counts_over_rounds.drop(index=0, errors='ignore')

anticonformity_df['treatment_resemblance'].value_counts()
## Plot anticonformity counts over the rounds as a stacked bar chart
plt.figure(figsize=(12, 6))
ax = anticonformity_counts_over_rounds.plot(
    kind='bar',
    stacked=True,
    color=[ "#8f8f8f", "#e9a5e3","#c92121", "#0551ea"],  # Colors for 'correct', 'incorrect', 'None'
    figsize=(12, 6)
)
plt.title('Anticonformists Resemblance over Rounds (Stacked) ' + session_id)
plt.xlabel('Round Number')
plt.ylabel('Count')
plt.xticks(ticks=range(0, 21), rotation=45)
plt.yticks(ticks=range(0, 3))
# Move legend outside the plot
plt.legend(
    title='Response Type',
    labels=['None', 'Unknown', 'anticonformity', 'conformity'],
    bbox_to_anchor=(1.05, 1), loc='upper left'
)
plt.tight_layout()
plt.savefig('anticonformity_resemblance_over_rounds_stacked_' + session_id + '.png', dpi=300, bbox_inches='tight')
plt.show()


## PLOT OPINION CHANGE OVER ROUNDS
# each participant is represented by different line
# x axis is round_no, y-axis is -1,0,1
# color is a shade of treatment type
# convert df_mock_relevant['response'] to be numeric
df_mock_relevant['response'] = pd.to_numeric(df_mock_relevant['response'], errors='coerce')
plt.figure(figsize=(20, 6))
for i, participant in enumerate(df_mock_relevant['participant.code'].unique()):
    participant_data = df_mock_relevant[df_mock_relevant['participant.code'] == participant]
    # Plot the response for each participant
    sns.pointplot(
        data=participant_data,
        x='round_no',
        y='response',
        dodge=True,
        color=sns.color_palette()[i % len(sns.color_palette())],
        markers='o',
        linestyles="none",
        alpha=0.2  # Adjust transparency for better visibility
    )
    # Connect the dots with a dotted line
    plt.plot(
        participant_data['round_no'],
        participant_data['response'],
        linestyle=':',
        color=sns.color_palette()[i % len(sns.color_palette())],
        alpha=0.7
    )

plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Round Number')
plt.ylabel('Response')
plt.title('Individual Responses with Dodge')
plt.tight_layout()
plt.show()

### Calculate mu / polarization index ###
# convert response as integer
df_mock_relevant['response'] = pd.to_numeric(df_mock_relevant['response'], errors='coerce')
# Compute polarization index per round and store c values
mu_round = []
c_neg = []
c_pos = []
c_neu = []
# Iterate through each round and calculate the polarization index
for round_no in df_mock_relevant['round_no'].unique().tolist():
    round_data = df_mock_relevant[df_mock_relevant['round_no'] == round_no]
    response = round_data['response']
    # Calculate polarization index
    c_negative = (response == -1).sum() / len(response)
    c_positive = (response == 1).sum() / len(response)
    c_neutral = (response == 0).sum() / len(response)
    #mu = (1 - abs(c_negative - c_positive)) / 2
    mu = (1 - abs(c_negative - c_positive)) / 2  * (c_positive / (c_neutral + c_positive)   + c_negative/ (c_neutral + c_negative))
    mu_round.append(mu)
    c_neg.append(c_negative)
    c_pos.append(c_positive)
    c_neu.append(c_neutral)

c_pos 
c_neg 
c_neu
mu_round


# check color keys
import matplotlib.colors as mcolors
print(mcolors.CSS4_COLORS.keys())


# Get the sorted list of round numbers
round_numbers = sorted(df_mock_relevant['round_no'].unique().tolist())

plt.figure(figsize=(12, 6))
plt.plot(round_numbers, mu_round, marker='o', linestyle='--',label='mu (Polarization Index)', color='purple')
plt.plot(round_numbers, c_neg, marker='x',linestyle='--', label='c_negative', color='red')
plt.plot(round_numbers, c_pos, marker='^',linestyle='--', label='c_positive', color='blue')
plt.plot(round_numbers, c_neu, marker='s',linestyle='--', label='c_neutral', color='green')
plt.xlabel('Round Number')
plt.ylabel('Value')
plt.ylim(0,1.1)
plt.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, label='mu=0.5 consensus')
plt.suptitle('session ' + session_id + ' b: 0.5, p: 0.5')
plt.title('Polarization Index (mu) values per Round')
plt.xticks(round_numbers)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('polarization_index_mu_' + session_id + '.png', dpi=300)
plt.show()
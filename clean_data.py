"""
Created on Thu May 15 09:38:41 2025

@author: juncosa
"""

import os
import re
import pandas as pd
import ast
import numpy as np
import math


print(os.getcwd())

# mac
os.chdir('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/')
# windows
#os.chdir('/Users/Lasmi Marbun/Documents/Git/Prolific-Full-Experiment/')

# Define the paths for raw and clean data
data_clean = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/Clean_files/'
data_raw = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/data/Raw_otree/'

# Change the filename to the current file
filename = 'all_apps_wide_k1hhm7lf.csv' # change file name as needed
df_raw = pd.read_csv(data_raw + filename) # change file name to current file

# remove participants with column participant._current_page_name != Feedback
df_raw = df_raw[df_raw['participant._current_page_name'] == 'Feedback'].reset_index(drop=True)

# remove returned participant from session k1hhm7lf (participant.code == '06jzthd0')
df_raw = df_raw[df_raw['participant.code'] != '06jzthd0'].reset_index(drop=True)

# remove returned participant from session pfu1qwfx (participant.code == 'kfziruzq')
df_raw = df_raw[df_raw['participant.code'] != 'kfziruzq'].reset_index(drop=True)

df_raw.shape

variables = df_raw.columns.to_list()

participant_vars = [var for var in variables if re.match(r'^participant\.', var)]
presurvey_vars = [var for var in variables if re.match(r'^presurvey\.', var)]
mock_vars = [var for var in variables if re.match(r'^mock\.', var)]
pay_vars = [var for var in variables if re.match(r'^Pay\.', var)]
noPay_vars = [var for var in variables if re.match(r'^noPay\.', var)]
player_vars = presurvey_vars + mock_vars + pay_vars + noPay_vars
other_vars = [var for var in variables if var not in participant_vars+presurvey_vars+mock_vars] 

print("No. ALL variables:", len(variables))
print("No. participant variables:", len(participant_vars))
print("No. player variables in presurvey:", len(presurvey_vars)) 
print("No. player variables in mock:", len(mock_vars))
print("No. player variables in pay:", len(pay_vars))
print("No. player variables in noPay:", len(noPay_vars))
print("No. player variables:", len(player_vars)) 
print("No. other variables:", len(other_vars)) 



participant_vars_keep = [
                        'participant.code',
                        'participant.label',
                        'participant.time_started_utc',
                        'participant.gives_consent', 
                        'participant.political_affiliation', 
                        'participant.scenario_type', 
                        'participant.training_attempt', 
                        'participant.training_success', 
                        'participant.no_consent', 
                        'participant.failed_commitment',
                        'participant.treatment', 
                        'participant.scenario_order', 
                        'participant.all_responses', 
                        'participant.wait_page_arrival', 
                        'participant.failed_attention_check', 
                        'participant.active', 
                        'participant.reason', 
                        'participant.scenario', # scenario code for the Discussion
                        'participant.anticonformist', 
                        'participant.no_nudge', 
                        'participant.complete_presurvey',
                        'participant.forced_response_counter',
                        'participant.not_neutral', 
                        'participant.neighbors_configurations', 
                        'participant.neighbors']

# keep variables in session vars
other_vars_keep = [
                    'session.code',
                    'session.combined_responses', 
                    'session.AC_p', 
                    'session.AC_n', 
                    'session.C_p', 
                    'session.C_n', 
                    'session.NO_p', 
                    'session.NO_n',
                    'session.AC_Dem_p', 
                    'session.AC_Rep_p', 
                    'session.AC_Dem_n', 
                    'session.AC_Rep_n',
                    'session.C_Dem_p', 
                    'session.C_Rep_p', 
                    'session.C_Dem_n', 
                    'session.C_Rep_n',
                    'session.NO_Dem_p', 
                    'session.NO_Rep_p',
                    'session.NO_Dem_n', 
                    'session.NO_Rep_n']

player_vars_all = [
    'presurvey.{i}.player.id_in_group',
    'presurvey.{i}.player.role',
    'presurvey.{i}.player.payoff',
    'presurvey.{i}.player.gives_consent',
    'presurvey.{i}.player.scenario_code',
    'presurvey.{i}.player.test_scenario',
    'presurvey.{i}.player.dilemmatopic',
    'presurvey.{i}.player.majority',
    'presurvey.{i}.player.howmanyneighbors',
    'presurvey.{i}.player.total_correct',
    'presurvey.{i}.player.training_counter',
    'presurvey.{i}.player.attention_check',
    'presurvey.{i}.player.age',
    'presurvey.{i}.player.gender',
    'presurvey.{i}.player.education_lvl',
    'presurvey.{i}.player.neighborhood_type',
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.group.id_in_subsession',
    'presurvey.{i}.subsession.round_number',
    'mock.{i}.player.id_in_group',
    'mock.{i}.player.role',
    'mock.{i}.player.payoff',
    'mock.{i}.player.forced_response',
    'mock.{i}.player.response',
    #'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
    'mock.{i}.group.id_in_subsession',
    'mock.{i}.subsession.round_number',
    'noPay.{i}.player.id_in_group', 
    'noPay.{i}.player.id_in_group',
    'Pay.{i}.player.feedback_final',
    'Pay.{i}.player.id_in_group',]

# variables that have multiple rounds
player_vars_allR = [
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.player.scenario_code',
    'mock.{i}.player.response',
    'mock.{i}.player.forced_response',
    #'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
]
# variables that have multiple rounds in presurvey app
player_vars_presurvey_allR = [
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.player.scenario_code'
]
# variables that have multiple rounds in mock app
player_vars_mock_allR = [
    'mock.{i}.player.response',
    'mock.{i}.player.forced_response',
    #'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
]

# variables that only appear in round 1
player_vars_allR1 = [
    'presurvey.{i}.player.id_in_group',
    'presurvey.{i}.player.role',
    'presurvey.{i}.player.payoff',
    'presurvey.{i}.player.gives_consent',
    'presurvey.{i}.player.test_scenario',
    'presurvey.{i}.player.dilemmatopic',
    'presurvey.{i}.player.majority',
    'presurvey.{i}.player.howmanyneighbors',
    'presurvey.{i}.player.total_correct',
    'presurvey.{i}.player.training_counter',
    'presurvey.{i}.player.attention_check',
    'presurvey.{i}.player.political_affiliation',
    'presurvey.{i}.player.age',
    'presurvey.{i}.player.gender',
    'presurvey.{i}.player.education_lvl',
    'presurvey.{i}.player.neighborhood_type',
    'mock.{i}.player.id_in_group',
    'mock.{i}.player.role',
    'mock.{i}.player.payoff',
    'mock.{i}.group.id_in_subsession',
    'noPay.{i}.player.id_in_group',
    'noPay.{i}.player.id_in_group',
    'Pay.{i}.player.feedback_final',
    'Pay.{i}.player.id_in_group',]

# Mock app has 10 rounds whilst presurvey app have 3 rounds
# Define variables that appear in rounds 4 to 10 of the mock app
player_vars_R4_R10 = [
    'mock.{i}.player.response',
    'mock.{i}.player.forced_response',
    #'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
]


player_vars_keep = []
for r in range(1,11):
    if r == 1:
        all_vars = player_vars_allR1 + player_vars_allR
    elif r > 3:
        all_vars = player_vars_R4_R10
    else:
        all_vars = player_vars_allR 
    
    for var in all_vars:
        player_vars_keep.append(var.format(i=r))
        
all_vars_keep = participant_vars_keep + other_vars_keep + player_vars_keep  
# df_clean = df_raw[all_vars_keep]
df_clean = df_raw[[col for col in all_vars_keep if col in df_raw.columns]]
# for real data:
df_clean = df_clean[df_clean['participant.label'].notna()].reset_index(drop=True)

print('Shape clean data:', df_clean.shape)

df_clean['participant.scenario_order'] = df_clean['participant.scenario_order'].apply(ast.literal_eval)

# For the mock app (discussion over 10 rounds), we need to keep the variables for all rounds
# and then create a dictionary with the player information for each round.
# The player information will include the participant code, scenario order, and responses for each round.
# Define the player_info variables (as in your player_info function)
player_info_vars = ['session.code','participant.code',
             'participant.scenario',
             'participant.scenario_order',
             'participant.scenario_type',
             'participant.anticonformist',
             'participant.label',
             'participant.treatment',
             'participant.neighbors_configurations',
             'participant.neighbors',
            'participant.forced_response_counter',
            'presurvey.1.player.gives_consent',
            'presurvey.1.player.age',
            'presurvey.1.player.gender',
            'presurvey.1.player.education_lvl',
            'presurvey.1.player.neighborhood_type',
            'presurvey.1.player.political_affiliation']

player_vars_mock_allR = [
    'mock.{i}.player.response',
    'mock.{i}.player.forced_response',
    #'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
]
# variables that have multiple rounds in presurvey app
player_vars_presurvey_allR = [
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.player.scenario_code',
]


### PRESURVEY APP participants 
# Identify participants with incomplete presurvey answers
invalid_presurvey = df_clean[df_clean['presurvey.3.player.response'].isnull()]
len(pd.Series(invalid_presurvey['participant.code']).unique()) # 0 people
# Print the participants with incomplete presurvey answers
print("Participants with incomplete presurvey answers:")
print(invalid_presurvey[['participant.label']])

# Remove participants with incomplete presurvey answers
df_clean = df_clean[~df_clean['participant.label'].isin(invalid_presurvey['participant.label'])].reset_index(drop=True)

# Verify the remaining participants
print(f"Remaining participants: {len(df_clean['participant.label'].unique())}")
print(f"Shape of df_clean after removal: {df_clean.shape}") # 23 participants

### MOCK APP participants
# Identify participants with incomplete mock app answers
invalid_participants = df_clean[df_clean['mock.10.player.response'].isnull()]
len(pd.Series(invalid_participants['participant.code']).unique())

# Print the participants with incomplete mock app answers
print("Participants with incomplete presurvey answers:")
print(invalid_participants[['participant.label']])

# Remove participants with incomplete mock app answers
df_clean = df_clean[~df_clean['participant.label'].isin(invalid_participants['participant.label'])].reset_index(drop=True)


# Verify the remaining participants
print(f"Remaining participants: {len(df_clean['participant.label'].unique())}")
print(f"Shape of df_clean after removal: {df_clean.shape}")

df_clean.columns



# ### Prepare the long-format data for the 'presurvey app' participants
# long_data = []
# # Iterate through each row in the cleaned DataFrame and create a dictionary for each round of the presurvey app with the player information and responses.


# for idx, row in df_clean.iterrows():
#     scenario_order = row['participant.scenario_order']
#     # Presurvey variables
#     for r in range(1, 4):  # Presurvey has 3 rounds
#         round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
#         round_data['round_no'] = r
#         round_data['scenario'] = scenario_order[r-1] if len(scenario_order) >= r else None
#         # Add all presurvey round variables for this round, but drop the round number from the column name
#         for var in player_vars_presurvey_allR:
#             colname = var.format(i=r)
#             shortname = var.replace('presurvey.{i}.', '').replace('player.', '') # e.g. 'response', 'political_charge', 'emotional_charge'
#             round_data[shortname] = row.get(colname, None)
#         for var in player_vars_mock_allR:
#             colname = var.format(i=r)
#             shortname = var.replace('mock.{i}.', '').replace('player.', '')
#             round_data[shortname] = row.get(colname, None)

#         long_data.append(round_data)
#     for r in range(4, 11):  # Mock app has 10 rounds
#         round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
#         round_data['round_no'] = r
#         for var in player_vars_mock_allR:
#             colname = var.format(i=r)
#             shortname = var.replace('mock.{i}.', '').replace('player.', '')
#             round_data[shortname] = row.get(colname, None)
        
#         long_data.append(round_data)

# df_long = pd.DataFrame(long_data)
# df_long.shape # 3 rows presurvey * N (23 participants) + 10 rows mock * N = 299
# # Save the long-format DataFrame to a CSV file
# df_long.to_csv(data_clean + 'clean_long_format_k1hhm7lf.csv', index=False)




long_data = []
## test
for idx, row in df_clean.iterrows():
    scenario_order = row['participant.scenario_order']
    # Presurvey variables
    for r in range(1, 4):  # Presurvey has 3 rounds
        round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
        round_data['round_no'] = r
        round_data['scenario'] = scenario_order[r-1] if len(scenario_order) >= r else None
        # Add all presurvey round variables for this round, but drop the round number from the column name
        for var in player_vars_presurvey_allR:
            colname = var.format(i=r)
            shortname = var.replace('presurvey.{i}.', '').replace('player.', '') # e.g. 'response', 'political_charge', 'emotional_charge'
            round_data[shortname] = row.get(colname, None)
    
        long_data.append(round_data)


for idx, row in df_clean.iterrows():
    for r in range(1, 11):  # Mock app has 10 rounds
            round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
            round_data['round_no'] = r
            for var in player_vars_mock_allR:
                colname = var.format(i=r)
                shortname = var.replace('mock.{i}.', '').replace('player.', '')
                round_data[shortname] = row.get(colname, None)
            
            long_data.append(round_data)

df_long = pd.DataFrame(long_data)

# Save the long-format DataFrame to a CSV file
cleanfilename = 'clean_long_format_k1hhm7lf.csv'
df_long.to_csv(data_clean + cleanfilename, index=False)


### Only for k1hhm7lf session ###
# Handling unsaved player.neighbors in session k1hhm7lf data

import ast

# Convert the string to a real Python list of dicts
df_long['participant.neighbors_configurations'] = df_long['participant.neighbors_configurations'].apply(ast.literal_eval)

# Extract the relevant dict based on round number and convert to list of values
def get_neighbors(row):
    config_list = row['participant.neighbors_configurations']
    round_index = int(row['round_no']) - 1  # 0-based index
    if isinstance(config_list, list) and 0 <= round_index < len(config_list):
        d = config_list[round_index]
        return list(d.values())
    else:
        return None  # or [] or [None, None, None] if preferred

df_long['neighbors'] = df_long.apply(get_neighbors, axis=1)

# If scenario_code is not NaN, remove the values from neighbors column
df_long.loc[df_long['scenario_code'].notna(), 'neighbors'] = None


# Save the long-format DataFrame to a CSV file
cleanfilename = 'clean_long_format_k1hhm7lf.csv'
df_long.to_csv(data_clean + cleanfilename, index=False)

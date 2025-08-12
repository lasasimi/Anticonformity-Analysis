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
os.chdir('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/')
# windows
#os.chdir('/Users/Lasmi Marbun/Documents/Git/Anticonformity-Analysis/')

path = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/'
# Define the paths for raw and clean data
data_clean = os.path.join(path, 'Clean_files/')
data_raw = os.path.join(path, 'Raw_otree/')

# Change the filename to the current file
filename = 'all_apps_wide_e1314ij0.csv' # change file name as needed
df_raw = pd.read_csv(data_raw + filename) # change file name to current file

# remove participants with column participant._current_page_name != Feedback
df_raw = df_raw[df_raw['participant._current_page_name'] == 'Feedback'].reset_index(drop=True)

# remove returned participant from session k1hhm7lf (participant.code == '06jzthd0')
#df_raw = df_raw[df_raw['participant.code'] != '06jzthd0'].reset_index(drop=True)

# remove returned participant from session pfu1qwfx (participant.code == 'kfziruzq')
#df_raw = df_raw[df_raw['participant.code'] != 'kfziruzq'].reset_index(drop=True)

df_raw.shape # 8 participants

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
                        'participant.single_group', 
                        'participant.reason',
                        'participant.player_ids', 
                        'participant.group_size', 
                        'participant.is_group_single',
                        'participant.scenario',
                        'participant.anticonformist',
                        'participant.position',
                        'participant.own_faction',
                        'participant.other_faction',
                        'participant.discussion_grp',
                        'participant.complete_presurvey',
                        'participant.eligible_notneutral',
                        'participant.forced_response_counter',]

# keep variables in session vars
other_vars_keep = [
                    'session.code',
                    'session.combined_responses', 
                    'session.N04_p00',
                    'session.N04_p25',
                    'session.N04_p50',
                    'session.N08_p00',
                    'session.N08_p25',
                    'session.N08_p50',
                    'MAX_N04_p00',
                    'MAX_N04_p25',
                    'MAX_N04_p50',
                    'MAX_N08_p00',
                    'MAX_N08_p25',
                    'MAX_N08_p50']

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
    'mock.{i}.player.neighbors', # not valid for session k1hhm7lf
    'mock.{i}.group.id_in_subsession',
    'mock.{i}.subsession.round_number',
    'noPay.{i}.player.id_in_group', 
    'noPay.{i}.player.id_in_group',
    'Pay.{i}.player.feedback_final',
    'Pay.{i}.player.id_in_group',]

# variables that have multiple rounds in mock app
player_vars_mock_allR = [
    'mock.{i}.player.old_response',
    'mock.{i}.player.new_response',
    'mock.{i}.player.forced_response',
    'mock.{i}.player.scenario',
    'mock.{i}.player.discussion_grp',
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
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.player.scenario_code',
    'mock.{i}.player.id_in_group',
    'mock.{i}.player.role',
    'mock.{i}.player.payoff',
    'mock.{i}.group.group_size',
    'mock.{i}.group.is_group_single',
    'mock.{i}.group.beta_50',
    'mock.{i}.group.anti_prop',
    'mock.{i}.group.id_in_subsession',
    'noPay.{i}.player.id_in_group',
    'noPay.{i}.player.id_in_group',
    'Pay.{i}.player.feedback_final',
    'Pay.{i}.player.id_in_group',]

# Mock app has 20 rounds whilst presurvey app have 1 rounds
# Define variables that appear in rounds 4 to 10 of the mock app
player_vars_R2_R20 = [
    'mock.{i}.player.old_response',
    'mock.{i}.player.new_response',
    'mock.{i}.player.forced_response',
    'mock.{i}.player.scenario',
    'mock.{i}.player.discussion_grp',
]


player_vars_keep = []
for r in range(1,21):
    if r == 1:
        all_vars = player_vars_allR1 + player_vars_mock_allR
    else:
        all_vars = player_vars_R2_R20
    for var in all_vars:
        player_vars_keep.append(var.format(i=r))

col_names = df_raw.columns.to_list()
all_vars_keep = participant_vars_keep + other_vars_keep + player_vars_keep  
# df_clean = df_raw[all_vars_keep]
df_clean = df_raw[[col for col in all_vars_keep if col in df_raw.columns]]
# for real data:
df_clean = df_clean[df_clean['participant.label'].notna()].reset_index(drop=True)
df_clean['presurvey.1.player.response']
print('Shape clean data:', df_clean.shape)

# see column names in df_clean
column_names = df_clean.columns.to_list()
#TODO: get the rest into a long format
# For the mock app (discussion 20 rounds), we need to keep the variables for all rounds
# and then create a dictionary with the player information for each round.
# The player information will include the participant code, scenario order, and responses for each round.
# Define the player_info variables (as in your player_info function)



player_info_vars = ['session.code','participant.code',
             'participant.scenario',
             'participant.scenario_order',
             'participant.training_attempt', 
             'participant.anticonformist',
             'participant.own_faction',
             'participant.other_faction',
             'participant.all_responses',
             'participant.label',
             'participant.discussion_grp',
             'participant.group_size', 
             'participant.single_group', 
             'participant.is_group_single',
             'participant.complete_presurvey',
             'participant.eligible_notneutral',
             'participant.wait_page_arrival',
            'participant.forced_response_counter',
            'presurvey.1.player.gives_consent',
            'presurvey.1.player.age',
            'presurvey.1.player.gender',
            'presurvey.1.player.education_lvl',
            'presurvey.1.player.neighborhood_type',
            'mock.1.group.beta_50',
            'mock.1.group.anti_prop'
        ]

player_vars_mock_allR = [
    'mock.{i}.player.old_response',
    'mock.{i}.player.new_response',
    'mock.{i}.player.forced_response',
    'mock.{i}.player.discussion_grp',
]
# presurvey variables
player_vars_presurvey = [
    'presurvey.{i}.player.response',
    'presurvey.{i}.player.political_charge',
    'presurvey.{i}.player.emotional_charge',
    'presurvey.{i}.player.scenario_code',
]




long_data = []
## test
for idx, row in df_clean.iterrows():
    # Presurvey variables
    for r in range(1,2):  # Presurvey app has only 1 round
        round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
        round_data['round_no'] = r
        # Add all presurvey round variables for this round, but drop the round number from the column name
        for var in player_vars_presurvey:
            colname = var.format(i=r)
            shortname = var.replace('presurvey.{i}.', '').replace('player.', '') # e.g. 'response', 'political_charge', 'emotional_charge'
            round_data[shortname] = row.get(colname, None)
    
        long_data.append(round_data)


for idx, row in df_clean.iterrows():
    for r in range(1, 21):  # Mock app has 20 rounds
            round_data = {var: row[var] for var in player_info_vars} #player_info_vars is the participant unique identifier
            round_data['round_no'] = r
            for var in player_vars_mock_allR:
                colname = var.format(i=r)
                shortname = var.replace('mock.{i}.', '').replace('player.', '')
                round_data[shortname] = row.get(colname, None)
            
            long_data.append(round_data)

df_long = pd.DataFrame(long_data)
len(df_long)  # 8 people, 1 round presurvey, 20 rounds mock app = 21 * 8 = 168 rows

# Save the long-format DataFrame to a CSV file
cleanfilename = 'clean_long_format_e1314ij0.csv'
df_long.to_csv(data_clean + cleanfilename, index=False)

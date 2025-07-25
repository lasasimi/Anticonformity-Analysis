import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 

print(os.getcwd())

# Load and merge the page time data
page_time_1 = pd.read_csv('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/Raw_otree/PageTimes-k1hhm7lf.csv')
page_time_2 = pd.read_csv('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/Raw_otree/PageTimes-pfu1qwfx.csv')
page_time_3 = pd.read_csv('/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/Raw_otree/PageTimes-7agw525e.csv')

# Merge the page time data
page_time = pd.concat([page_time_1, page_time_2, page_time_3], ignore_index=True)

participant_code = page_time['participant_code'].unique().tolist()
len(participant_code)

# Filter the rows to only include those participants who finished the round_number 10
complete_participants = page_time[page_time['round_number'] == 10]['participant_code'].unique().tolist()
len(complete_participants)

# Exclude participants who were flagged by Prolific '06jzthd0' and 'kfziruzq'
complete_participants = [p for p in complete_participants if p not in ['06jzthd0', 'kfziruzq']]
len(complete_participants)

# Filter the page_time DataFrame to only include complete participants
page_time = page_time[page_time['participant_code'].isin(complete_participants)]

page_names = page_time['page_name'].unique().tolist()
page_names 





### Full app analysis (N=10) ###
# Create a dictionary to store time differences for each participant
participant_time_differences = {}

# filter to only include those who have data for the ALL app
page_time_full_app = page_time[page_time['app_name'] == 'Pay']
participant_code_full = page_time_full_app['participant_code'].unique().tolist()
page_time_full_app = page_time[page_time['participant_code'].isin(participant_code_full)]
#page_name_full_app = page_time_full_app['page_name'].unique().tolist()

page_name_full_app = ['InitializeParticipant',
 'Introduction',
 'Demographics',
 'NeighborhoodInstruction',
 'Neighborhood',
 'Training',
 'TrainingNeighbor_1',
 'TrainingNeighbor_2',
 'ExperimentInstruction',
 'Neighborhood_1',
 'Scenario',
 'Commitment',
 'FinalPage',
 'GroupingWaitPage',
 'GroupSizeWaitPage',
 'DiscussionGRPWaitPage',
 'Phase3',
 'Nudge',
 'Discussion',
 'Feedback']


for i in participant_code_full:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time_full_app, page_name_full_app, i)
    
    # Store the result in the dictionary
    participant_time_differences[i] = time_differences

# Compute the average and standard deviation across all participants
num_pages = len(page_name_full_app) - 1  # Number of page transitions
average_differences = []
std_differences = []

for i in range(num_pages):
    # Collect differences for the current page transition across all participants
    values = [differences[i] for differences in participant_time_differences.values() if differences[i] is not None]
    
    # Compute the average and standard deviation
    if values:
        average_differences.append(np.mean(values))
        std_differences.append(np.std(values))
    else:
        average_differences.append(0)
        std_differences.append(0)

# Create x-axis values (sequence of page names)
# x_values = range(1, num_pages + 1)  # Sequence of page transitions
x_values = []
for i in page_name_full_app[1:]: # Exclude the first page for x-axis
    x_values.append(i)

plt.figure(figsize=(20, 6))
# Plot the average differences with error bars
plt.errorbar(x_values, average_differences, yerr=std_differences, fmt='-o', capsize=5, label='Average Time Differences')

# Set y-axis limit
plt.ylim(0, 600)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis 

# Annotate each point with its value
for i, (x, y) in enumerate(zip(x_values, average_differences)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, ha='center', va='bottom')  # Display value with 2 decimal places

# Add labels and title
plt.xlabel('Page Time in ALL app')
plt.show()
plt.savefig('average_time_N10.png')



### Presurvey app analysis ###
presurvey_page_names = page_names[:14]
presurvey_page_names = ['InitializeParticipant',
 'Introduction',
 'Demographics',
 'NeighborhoodInstruction',
 'Neighborhood',
 'Training',
 'TrainingNeighbor_1',
 'TrainingNeighbor_2',
 'ExperimentInstruction',
 'Neighborhood_1',
 'Scenario',
 'Commitment',
 'FinalPage',]
 #'AttentionCheck']
# Create a dictionary to store time differences for each participant
participant_time_differences = {}

for participant in participant_code:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time, presurvey_page_names, participant)
    
    # Store the result in the dictionary
    participant_time_differences[participant] = time_differences

# Compute the average and standard deviation across all participants
num_pages = len(presurvey_page_names) - 1

for i in participant_code_full:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time_full_app, presurvey_page_names, i)
    
    # Store the result in the dictionary
    participant_time_differences[i] = time_differences

# Compute the average and standard deviation across all participants
num_pages = len(presurvey_page_names) - 1  # Number of page transitions
average_differences = []
std_differences = []

for i in range(num_pages):
    # Collect differences for the current page transition across all participants
    values = [differences[i] for differences in participant_time_differences.values() if differences[i] is not None]
    
    # Compute the average and standard deviation
    if values:
        average_differences.append(np.mean(values))
        std_differences.append(np.std(values))
    else:
        average_differences.append(0)
        std_differences.append(0)

# Create x-axis values (sequence of page names)
# x_values = range(1, num_pages + 1)  # Sequence of page transitions
x_values = []
for i in presurvey_page_names[1:]: # Exclude the first page for x-axis
    x_values.append(i)

plt.figure(figsize=(20, 6))
# Plot the average differences with error bars
plt.errorbar(x_values, average_differences, yerr=std_differences, fmt='-o', capsize=5, label='Average Time Differences')
plt.ylim(0, 600)  # Set y-axis limit
plt.xticks(rotation=45, ha='right')  # Rotate x-axis 

# Annotate each point with its value
for i, (x, y) in enumerate(zip(x_values, average_differences)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, ha='center', va='bottom')  # Display value with 2 decimal places

# Add labels and title
plt.xlabel('Page Time in presurvey app')
plt.show()
plt.savefig('average_time_differences_presurvey.png')






### Those who waited ###
# Filter otree_data to find matching participant.code
participant_label_wait = pd.read_csv('participantid_waitingbonus.csv')['Participant id'].unique().tolist()
otree_data = pd.read_csv('all_apps_wide_b0qkdyda.csv')


participant_code_wait = otree_data[otree_data['participant.label'].isin(participant_label_wait)]['participant.code'].unique().tolist()
len(participant_code_wait) # 15 people who waited
page_time_wait = page_time[page_time['participant_code'].isin(participant_code_wait)]

page_name_wait = ['InitializeParticipant',
 'Introduction',
 'Demographics',
 'NeighborhoodInstruction',
 'Neighborhood',
 'Training',
 'TrainingNeighbor_1',
 'TrainingNeighbor_2',
 'ExperimentInstruction',
 'Neighborhood_1',
 'Scenario',
 'Commitment',
 'FinalPage',
# 'AttentionCheck',
 'GroupingWaitPage',
 'GroupSizeWaitPage',
 'DiscussionGRPWaitPage']


# Create a dictionary to store time differences for each participant
participant_time_differences = {}
for i in participant_code_wait:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time_wait, page_name_wait, i)
    
    # Store the result in the dictionary
    participant_time_differences[i] = time_differences

### Plotting the individual time differences for those who waited ###
plt.figure(figsize=(20, 6))
for participant, time_diffs in participant_time_differences.items():
    x = page_name_wait[1:]  # Exclude the first page for x-axis
    y = [diff if diff is not None else 0 for diff in time_diffs][0:15]  # Replace None with 0
    plt.plot(x, y, marker='o', label=participant, alpha=0.5)  # Set alpha for transparency
    for idx, (xi, yi) in enumerate(zip(x, y)):
        # Add a small random jitter to the x-position for readability
        jitter = random.uniform(-0.2, 0.2)  # Jitter for categorical x-axis
        plt.text(idx + jitter, yi, f'{yi}', fontsize=8, ha='center', va='bottom', rotation=0)

plt.xticks(ticks=range(len(x)), labels=x, rotation=45, ha='right')
plt.xlabel('Page Name')
plt.ylabel('Time Difference')
plt.title('Time Differences per Participant (Wait)')
plt.legend(title='Participant Code', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('time_differences_per_participant_wait.png')
plt.show()

### Plotting the average time differences for those who waited ###
# Compute the average and standard deviation across all participants
num_pages = len(page_name_wait) - 1  # Number of page transitions
average_differences = []
std_differences = []

for i in range(num_pages):
    # Collect differences for the current page transition across all participants
    values = [differences[i] for differences in participant_time_differences.values() if differences[i] is not None]
    
    # Compute the average and standard deviation
    if values:
        average_differences.append(np.mean(values))
        std_differences.append(np.std(values))
    else:
        average_differences.append(0)
        std_differences.append(0)

# Create x-axis values (sequence of page names)
# x_values = range(1, num_pages + 1)  # Sequence of page transitions
x_values = []
for i in page_name_wait[1:]: # Exclude the first page for x-axis
    x_values.append(i)

plt.figure(figsize=(20, 6))
# Plot the average differences with error bars
plt.errorbar(x_values, average_differences, yerr=std_differences, fmt='-o', capsize=5, label='Average Time Differences')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis 

# Annotate each point with its value
for i, (x, y) in enumerate(zip(x_values, average_differences)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, ha='center', va='bottom')  # Display value with 2 decimal places

# Add labels and title
plt.xlabel('Page Time for those who waited')
plt.show()
plt.savefig('average_time_differences_wait.png')


### Find arrival time for those who waited ###
# Load participant_code_wait and otree_data


# Filter otree_data to find matching participant.code
participant_label_wait = pd.read_csv('participantid_waitingbonus.csv')['Participant id'].unique().tolist()
otree_data = pd.read_csv('all_apps_wide_b0qkdyda.csv')
participant_code_wait = otree_data[otree_data['participant.label'].isin(participant_label_wait)]['participant.code'].unique().tolist()
len(participant_code_wait) # 15 people who waited


page_time_wait = page_time[page_time['participant_code'].isin(participant_code_wait)]

page_name_wait = ['InitializeParticipant',
 'Introduction',
 'Demographics',
 'NeighborhoodInstruction',
 'Neighborhood',
 'Training',
 'TrainingNeighbor_1',
 'TrainingNeighbor_2',
 'ExperimentInstruction',
 'Neighborhood_1',
 'Scenario',
 'Commitment',
 'FinalPage',
# 'AttentionCheck',
 'GroupingWaitPage',
 'GroupSizeWaitPage',
 'DiscussionGRPWaitPage']

# Filter rows where app_name is "Pay" for those who only waited (not continue)
pay_epoch_times = page_time_wait[page_time_wait['app_name'] == 'Pay'][['epoch_time_completed', 'page_name']]

pay_epoch_times = pay_epoch_times.loc[pay_epoch_times['page_name'] == 'FinalPage', 'epoch_time_completed']

# those who needed trianing neighbor 2
# everyone
test = page_time.loc[page_time['page_name'] == 'TrainingNeighbor_2', 'epoch_time_completed']
len(test)
# those who waited but not matched
test = page_time_wait.loc[page_time_wait['page_name'] == 'TrainingNeighbor_2', 'epoch_time_completed']
len(test)
# those who waited and matched
test = page_time_full_app.loc[page_time_full_app['page_name'] == 'TrainingNeighbor_2', 'epoch_time_completed']
len(test)

from datetime import datetime, timezone, timedelta

# Convert epoch times to date and time in GMT+2
pay_epoch_times_converted = pay_epoch_times.apply(
    lambda x: datetime.fromtimestamp(x, tz=timezone(timedelta(hours=2)))
)

pay_epoch_times_converted
# Save the converted epoch times to a CSV file
pay_epoch_times_converted.to_csv('waitingonly_FinalPage_time.csv', index=False)

# Filter rows where app_name is "Pay" for those who only waited (not continue)
pay_full_epoch_times = page_time_full_app[page_time_full_app['app_name'] == 'presurvey'][['epoch_time_completed', 'page_name']]

pay_full_epoch_times = pay_full_epoch_times.loc[pay_full_epoch_times['page_name'] == 'FinalPage', 'epoch_time_completed']

test = pay_full_epoch_times.loc[pay_full_epoch_times['page_name'] == 'TrainingNeighbor_2', 'epoch_time_completed']

# Convert epoch times to date and time in GMT+2
pay_full_epoch_times_converted = pay_full_epoch_times.apply(
    lambda x: datetime.fromtimestamp(x, tz=timezone(timedelta(hours=2)))
)

pay_full_epoch_times_converted
# Save the converted epoch times to a CSV file
pay_full_epoch_times_converted.to_csv('N10allapp_FinalPage_time.csv', index=False)


##### Filter particpants who don't have the finish code (no waiting bonus and no full app bonus; N = 12 -- but 2 are not detected in otree) ####
participant_label_nowait = pd.read_csv('participantid_nowait.csv')['Participant id'].unique().tolist()
otree_data = pd.read_csv('Raw_otree/all_apps_wide_b0qkdyda.csv')
participant_code_nowait = otree_data[otree_data['participant.label'].isin(participant_label_nowait)]['participant.code'].unique().tolist()
len(participant_code_nowait) # 10 people who didn't get the waiting bonus and exist in otree data
participant_label_nowait_otree = otree_data[otree_data['participant.label'].isin(participant_label_nowait)]['participant.label'].unique().tolist()
len(participant_label_nowait_otree)

pd.Series(participant_label_nowait).isin(participant_label_nowait_otree)

page_time_nowait = page_time[page_time['participant_code'].isin(participant_code_nowait)]

page_name_nowait = ['InitializeParticipant',
 'Introduction',
 'Demographics',
 'NeighborhoodInstruction',
 'Neighborhood',
 'Training',
 'TrainingNeighbor_1',
 'TrainingNeighbor_2',
 'ExperimentInstruction',
 'Neighborhood_1',
 'Scenario',
 'Commitment',
 'FinalPage',]
# 'AttentionCheck',
# 'GroupingWaitPage',
# 'GroupSizeWaitPage',
# 'DiscussionGRPWaitPage']

len(page_name_nowait)

participant_time_differences = {}
for i in participant_code_nowait:
    print(i)
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time_nowait, page_name_nowait, i)
    
    # Store the result in the dictionary
    participant_time_differences[i] = time_differences

plt.figure(figsize=(20, 6))
for participant, time_diffs in participant_time_differences.items():
    x = page_name_nowait[1:]
    y = [diff if diff is not None else 0 for diff in time_diffs][0:12]  # Replace None with 0
    plt.plot(x, y, marker='o', label=participant, alpha=0.5)  # Set alpha for transparency
    for idx, (xi, yi) in enumerate(zip(x, y)):
        # Add a small random jitter to the x-position for readability
        jitter = random.uniform(-0.2, 0.2)  # Jitter for categorical x-axis
        plt.text(idx + jitter, yi, f'{yi}', fontsize=8, ha='center', va='bottom', rotation=0)

    '''for idx, (xi, yi) in enumerate(zip(x, y)):

        plt.text(idx + jitter, yi, f'{yi}', fontsize=8, ha='center', va='bottom', rotation=0)'''
plt.xticks(ticks=range(len(x)), labels=x, rotation=45, ha='right')
plt.xticks(ticks=range(len(x)), labels=x, rotation=45, ha='right')
plt.ylim(0, 800)  # Set y-axis limit
plt.xlabel('Page Name')
plt.ylabel('Time Difference')
plt.title('Time Differences per Participant (No Wait)')
plt.legend(title='Participant Code', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('time_differences_per_participant_nowait.png')
plt.show()


#### Create a different plot: x-axis is a timeline of 10s intervals, y axis show different participants, nodes show the pagename ####
## Group participants based on completion of the full app
# Load otree data
otree_data = pd.read_csv('all_apps_wide_b0qkdyda.csv')

# Assign completion status for each participant
# Default to 'unknown'
otree_data['completion'] = 'unknown'
# If 'presurvey.4.player.emotional_charge' is not null, set to 'presurvey'
otree_data.loc[otree_data['presurvey.4.player.emotional_charge'].notnull(), 'completion'] = 'presurvey'
# If 'mock.5.group.is_group_single' is True, set to 'waiting'
otree_data.loc[otree_data['mock.1.group.is_group_single'] == True, 'completion'] = 'waiting'
# If 'mock.5.player.new_response' is not null, set to 'fullapp'
otree_data.loc[otree_data['mock.5.player.new_response'].notnull(), 'completion'] = 'fullapp'
otree_data['completion'].value_counts()


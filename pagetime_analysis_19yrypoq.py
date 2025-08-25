import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 

print(os.getcwd())

# Load and merge the page time data
# windows
file_location = '/Users/Lasmi Marbun/Documents/Git/Anticonformity-Analysis/'
# mac
#file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/'
session_id = '19yrypoq' # change this to the session id you want to analyze
page_time = pd.read_csv(file_location + 'Raw_otree/PageTimes-' + session_id + '.csv')

# Merge the page time data
# page_time = pd.concat([page_time_1, page_time_2, page_time_3], ignore_index=True)

participant_code = page_time['participant_code'].unique().tolist()
len(participant_code) # 37 total participants

# get unique page names
page_time['page_name'].unique().tolist() 

# Filter the rows to only include those participants who are commited and waiting (Page Name is GroupSizeWaitPage)
waiting_label = ['60c969c12749ced10866921d',
                 '640d43c9451ce148162a7643',
                 '5d5b67ab9ad5a300181398dc',
                 '672c1b00e4ed661368e89749',
                 '67509e842a08ee314f43c29f',]
                 

base_pay_label = [
'63e55f08844f6fb08115851f',
'66451ed7d46e28ec596d2514',
'67195ce0322d3935df36e8be',
'5dc5995e5d81cd4153e00138',
'5ba8e80089390100019064ed',
'67e069199f5b036960b0cc5f',
'663e5cf112f5d047cf5c9fad',
'676cf85a261dedbddea3e240',
'5e7b6486b704ac034e3f01c8',
'5ee94f66395da52f305b3ca7',
'676325864a29c405dab887a0',
'6317afadedfd0a63eb5d0522',
'5cd062b7c24ebe0001f9fdb6',
'67db3dd91c05b7278a0aad44',
'6678a6336d90a3a9973585c3',]



full_pay_label = ['5e39d872793a082024d9cf18',
                  '672d127c0844b35fabc0dae8',
                  '66846905cfa45a35dcdb7288',
                  '5cb9bc10e50fe5001746c43b',
                  '65cb86a7e667e637599a3af3',
                  '5eb02a40b17a8c0868685d32']

timed_out_label = ['5edcfe625cd39c9efd5cc2ae',
                   '655fa6bf1cfd3bd5343e7778']

# match the participant code with Prolific participant label
prolific_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_' + session_id + '.csv')
prolific_data = prolific_data.loc[:, ['participant.code', 'participant.label']]

# Remove nan values
prolific_data = prolific_data.dropna(subset=['participant.label'])
len(prolific_data['participant.label'].unique().tolist()) # 42 participants (still includes those who returned)
prolific_data['participant.label'].unique().tolist() 

# Get the participant code and label for base_pay and complete participants
base_pay_participants_code = prolific_data[prolific_data['participant.label'].isin(base_pay_label)]
waiting_participants_code = prolific_data[prolific_data['participant.label'].isin(waiting_label)]
full_participants_code = prolific_data[prolific_data['participant.label'].isin(full_pay_label)]
timed_out_participants_code = prolific_data[prolific_data['participant.label'].isin(timed_out_label)]
# Get the last page name of each participant in the waiting participants code
page_time['last_page_name'] = page_time.groupby('participant_code')['page_name'].transform('last')
base_pay_participants_last_page = page_time[page_time['participant_code'].isin(base_pay_participants_code['participant.code'])][['participant_code', 'last_page_name']]
waiting_participants_last_page = page_time[page_time['participant_code'].isin(waiting_participants_code['participant.code'])][['participant_code', 'last_page_name']]
full_participants_last_page = page_time[page_time['participant_code'].isin(full_participants_code['participant.code'])][['participant_code', 'last_page_name']]
timed_out_participants_last_page = page_time[page_time['participant_code'].isin(timed_out_participants_code['participant.code'])][['participant_code', 'last_page_name']]
# Filter only the unique participant codes
base_pay_participants_last_page = base_pay_participants_last_page.drop_duplicates(subset=['participant_code'])
waiting_participants_last_page = waiting_participants_last_page.drop_duplicates(subset=['participant_code'])
full_participants_last_page = full_participants_last_page.drop_duplicates(subset=['participant_code'])
timed_out_participants_last_page = timed_out_participants_last_page.drop_duplicates(subset=['participant_code'])
# Combine with participant label from the waiting_participants_code
base_pay_participants_last_page = base_pay_participants_last_page.merge(base_pay_participants_code, left_on='participant_code', right_on='participant.code', how='left')
waiting_participants_last_page = waiting_participants_last_page.merge(waiting_participants_code, left_on='participant_code', right_on='participant.code', how='left')
full_participants_last_page = full_participants_last_page.merge(full_participants_code, left_on='participant_code', right_on='participant.code', how='left')
timed_out_participants_last_page = timed_out_participants_last_page.merge(timed_out_participants_code, left_on='participant_code', right_on='participant.code', how='left') 
### Waiting Participants Analysis ###
# Filter the page_time DataFrame to only include waiting participants
waitpart_page_time = page_time[page_time['participant_code'].isin(waiting_participants_last_page['participant_code'])]

page_names = page_time['page_name'].unique().tolist()
page_names 

# If round_number is >1, append the round number to the page name and create a new row for that participant
page_time['page_name'] = page_time.apply(
    lambda row: f"{row['page_name']}_{row['round_number']}" if row['round_number'] > 1 else row['page_name'],
    axis=1
)

all_page_names = page_time['page_name'].unique()
all_page_names_presurvey = ['InitializeParticipant', 
                            'Introduction', 
                            'Demographics',
                            'NeighborhoodInstruction', 
                            'Training', 
                            'TrainingNeighbor_1',
                            #'TrainingNeighbor_2',
                            'ExperimentInstruction',
                            'Scenario',
                            'Commitment',
                            'AttentionCheck', 
                            #'TrainingNeighbor_3',
                            'GroupingWaitPage', 
                            'GroupSizeWaitPage', 
                            'DiscussionGRPWaitPage',]

# Create a function to compute time differences between consecutive pages for a given participant
def compute_page_time_differences(page_time, page_names, participant):
    """
    Compute differences in epoch_time_completed for consecutive pages for a given participant.

    Args:
        page_time (DataFrame): The DataFrame containing page time data.
        page_names (list): List of mock page names to compute differences for.
        participant (str): The participant code.

    Returns:
        list: A list of differences in epoch_time_completed for consecutive pages.
    """
    differences = []
    for i in range(1, len(page_names)):
        # Filter data for the current and previous page
        page_prev = page_time[
            (page_time['page_name'] == page_names[i - 1]) &
            (page_time['participant_code'] == participant)
        ]
        page_curr = page_time[
            (page_time['page_name'] == page_names[i]) &
            (page_time['participant_code'] == participant)
        ]
        
        # Check if both pages have data for the participant
        if not page_prev.empty and not page_curr.empty:
            # Compute the difference for the given participant
            time_diff = page_curr['epoch_time_completed'].values[0] - page_prev['epoch_time_completed'].values[0]
            differences.append(int(time_diff))
        else:
            # Append None if data is missing
            differences.append(None)
    
    return differences


# Example usage:
participant = '4gam1eus'
time_differences = compute_page_time_differences(page_time, all_page_names_presurvey, participant)
print(f"Time differences for participant {participant}: {time_differences}")
all_page_names_presurvey = ['InitializeParticipant', 
                            'Introduction', 
                            'Demographics',
                            'NeighborhoodInstruction', 
                            'Training', 
                            'TrainingNeighbor_1',
                            #'TrainingNeighbor_2',
                            'ExperimentInstruction',
                            'Scenario',
                            'Commitment',
                            'AttentionCheck', 
                            #'TrainingNeighbor_3',
                            'GroupingWaitPage', 
                            'GroupSizeWaitPage', 
                            'DiscussionGRPWaitPage',]

# Get all the page names except the excluded ones
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
relevant_page_names = [page for page in all_page_names if page not in excluded_pages]


# Create a dictionary to store time differences for each participant
participant_time_differences = {}

for participant in participant_code:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(waitpart_page_time, all_page_names_presurvey, participant)
    
    # Store the result in the dictionary
    participant_time_differences[participant] = time_differences

# Compute the average and standard deviation across all participants
num_pages = len(all_page_names_presurvey) - 1  # Number of page transitions
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
for i in all_page_names_presurvey[1:]: # Exclude the first page for x-axis
    x_values.append(i)

# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'average_time_differences_all_pages_{session_id}.png')

plt.figure(figsize=(20, 6))
# Plot the average differences with error bars
plt.errorbar(x_values, average_differences, yerr=std_differences, fmt='-o', capsize=5, label='Average Time Differences')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis 

# Annotate each point with its value
for i, (x, y) in enumerate(zip(x_values, average_differences)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, ha='center', va='bottom')  # Display value with 2 decimal places

# Add labels and title
plt.xlabel('Page Time in app')
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

###### Plotting the page transitions over time for each participant ######
# Filter the page_time DataFrame to only include waiting participants
len(waitpart_page_time['participant_code'].unique().tolist()) # 5 participants
waitpart_page_time
all_page_names_presurvey

# Exclude Training_Neighbor_2 and Training_Neighbor_3 and AttentionCheck pages from waitpart_page_time
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
waitpart_page_time = waitpart_page_time[~waitpart_page_time['page_name'].isin(excluded_pages)]


min_time = waitpart_page_time['epoch_time_completed'].min()
waitpart_page_time['time_from_start'] = waitpart_page_time['epoch_time_completed'] - min_time


# Bin time into 10s intervals
waitpart_page_time['time_bin'] = (waitpart_page_time['time_from_start'] // 10) * 10

# Get unique participant codes and page names
waiting_participants = waitpart_page_time['participant_code'].unique() 
len(waiting_participants) # 8 participants

# Remove unnecessary pages from all_page_names
waiting_page_names = waitpart_page_time['page_name'].unique().tolist()   

# Check each participant's answers in Scenario
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_' + session_id + '.csv')
# Filter the otree_data to only include relevant participants
waiting_otree_data = otree_data[otree_data['participant.code'].isin(waiting_participants)]
# Get the participant codes and their answers in Scenario
waiting_scenario_answers = waiting_otree_data[['participant.code','presurvey.1.player.response']]

# check column names from otree data
otree_data.columns.to_list()
### Note: still doesn't work, need to fix that it only captures the relevant pages
# Assign a color to each page_name
colors = plt.cm.get_cmap('tab20', len(waiting_page_names))
page_name_to_color = {name: colors(i) for i, name in enumerate(waiting_page_names)}

# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'participant_page_transitions_over_time_{session_id}.png')

plt.figure(figsize=(20, 6))
for i, participant in enumerate(waiting_participants):
    df_part = waitpart_page_time[waitpart_page_time['participant_code'] == participant]
    # Get the scenario answer for this participant
    scenario_row = waiting_scenario_answers[waiting_scenario_answers['participant.code'] == participant]
    if not scenario_row.empty:
        answer = scenario_row['presurvey.1.player.response'].values[0]
        if answer == -1:
            line_color = '#097969'
        elif answer == 1:
            line_color = '#400080'
        else:
            line_color = '#808080'
    else:
        line_color = '#808080'
    # Draw horizontal line for this participant, colored by scenario answer
    if not df_part.empty:
        min_time = df_part['time_bin'].min()
        max_time = df_part['time_bin'].max()
        plt.hlines(i, min_time, max_time, color=line_color, linewidth=4, alpha=0.5, zorder=1)
    # Scatter plot for nodes
    plt.scatter(
        df_part['time_bin'],
        np.full_like(df_part['time_bin'], i),
        c=[page_name_to_color[name] for name in df_part['page_name']],
        label=participant if i == 0 else "",  # Only label first for legend
        s=60,
        edgecolor='k',
        zorder=2
    )

# Set x-ticks to show every 2 minutes (120s)
min_bin = int(waitpart_page_time['time_bin'].min())
max_bin = int(waitpart_page_time['time_bin'].max())
step = 120  # 2 minutes in seconds
xticks = np.arange(min_bin, max_bin + step, step)
xtick_labels = [str(int(x // 60)) for x in xticks]  # show as minutes
plt.xticks(xticks, xtick_labels, rotation=45)
plt.xlabel('Time (minutes, binned every 2 min)')
# Set y-ticks to participant codes
plt.yticks(range(len(waiting_participants)), waiting_participants)
plt.ylabel('Participant Code')
plt.title('Waiting Participant Page Transitions Over Time')
# Create a legend for page_name colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in waiting_page_names]
# add label for line colors
line_labels = ['Scenario Answer: -1 Against', 'Scenario Answer: 1 For']
line_colors = ['#097969', '#400080'] 
handles += [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(line_colors, line_labels)]
plt.legend(handles=handles, title='Page Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



### Visualize all participants page sequence ###
waitpart_page_time
basepay_page_time = page_time[page_time['participant_code'].isin(base_pay_participants_last_page['participant_code'])]
full_pay_page_time = page_time[page_time['participant_code'].isin(full_participants_last_page['participant_code'])]
timed_out_page_time = page_time[page_time['participant_code'].isin(timed_out_participants_last_page['participant_code'])]
# Merge the four page times
all_page_time = pd.concat([full_pay_page_time, timed_out_page_time, waitpart_page_time, basepay_page_time], ignore_index=True)


min_time = all_page_time['epoch_time_completed'].min()
all_page_time['time_from_start'] = all_page_time['epoch_time_completed'] - min_time


# Bin time into 10s intervals
all_page_time['time_bin'] = (all_page_time['time_from_start'] // 10) * 10

# Get unique participant codes and page names
participants = all_page_time['participant_code'].unique() 
len(participants) # 29 participants
page_names = all_page_time['page_name'].unique()
len(page_names) # 56 pages

# Presurvey pages only
# Remove unnecessary pages from all_page_names
all_page_names_presurvey
excluded_pages = list(set(all_page_names) - set(all_page_names_presurvey))

# Remove where pagename is in excluded_pages
all_page_time = all_page_time[~all_page_time['page_name'].isin(excluded_pages)]


# Check each participant's answers in Scenario
# get data from the all_apps_wide_gneiwiau.csv file
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_' + session_id + '.csv')
# Filter the otree_data to only include relevant participants
all_otree_data = otree_data[otree_data['participant.code'].isin(participants)]
# Get the participant codes and their answers in Scenario
scenario_answers = all_otree_data[['participant.code','presurvey.1.player.response']]

# check column names from otree data
otree_data.columns.to_list()
### Note: still doesn't work, need to fix that it only captures the relevant pages
# Assign a color to each page_name
colors = plt.cm.get_cmap('tab20', len(all_page_names_presurvey))
page_name_to_color = {name: colors(i) for i, name in enumerate(all_page_names_presurvey)}

## Optionally, sort the all_page_time based on first time_from_start of each participant
sorted_participants = all_page_time.groupby('participant_code')['time_from_start'].min().sort_values().index.tolist()
# Sort all_page_time by participant_code order in sorted_participants
all_page_time['participant_code'] = pd.Categorical(all_page_time['participant_code'], categories=sorted_participants, ordered=True)
sorted_all_page_time = all_page_time.sort_values(['participant_code', 'time_from_start']).reset_index(drop=True)

# remove data of sorted_all_page_time when page_name is in excluded_pages
excluded_pages
sorted_all_page_time = sorted_all_page_time[~sorted_all_page_time['page_name'].isin(excluded_pages)]

sorted_all_page_time['page_name'].unique().tolist()

# Exclude attentionCheck from all_page_names_presurvey
all_page_names_presurvey = [page for page in all_page_names_presurvey if page != 'AttentionCheck']
len(all_page_names_presurvey) # 12 pages
len(sorted_all_page_time['page_name'].unique().tolist()) # 13 pages

# remove attention check from sorted_all_page_time
sorted_all_page_time = sorted_all_page_time[sorted_all_page_time['page_name'] != 'AttentionCheck']
len(sorted_all_page_time['page_name'].unique().tolist()) # 12 pages


# Plot
# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'participant_page_transitions_over_time_all_sorted{session_id}.png')

plt.figure(figsize=(20, 6))
for i, participant in enumerate(participants):
    df_part = sorted_all_page_time[sorted_all_page_time['participant_code'] == participant]
    # Get the scenario answer for this participant
    scenario_row = scenario_answers[scenario_answers['participant.code'] == participant]
    if not scenario_row.empty:
        answer = scenario_row['presurvey.1.player.response'].values[0]
        if answer == -1:
            line_color = '#097969'
        elif answer == 1:
            line_color = '#400080'
        else:
            line_color = '#808080'
    else:
        line_color = '#808080'
    # Draw horizontal line for this participant, colored by scenario answer
    if not df_part.empty:
        min_time = df_part['time_bin'].min()
        max_time = df_part['time_bin'].max()
        plt.hlines(i, min_time, max_time, color=line_color, linewidth=4, alpha=0.5, zorder=1)
    # Scatter plot for nodes
    plt.scatter(
        df_part['time_bin'],
        np.full_like(df_part['time_bin'], i),
        c=[page_name_to_color[name] for name in df_part['page_name']],
        label=participant if i == 0 else "",  # Only label first for legend
        s=60,
        edgecolor='k',
        zorder=2
    )


# Set x-ticks to show every 2 minutes (120s)
min_bin = int(sorted_all_page_time['time_bin'].min())
max_bin = int(sorted_all_page_time['time_bin'].max())
step = 120  # 2 minutes in seconds
xticks = np.arange(min_bin, max_bin + step, step)
xtick_labels = [str(int(x // 60)) for x in xticks]  # show as minutes
plt.xticks(xticks, xtick_labels, rotation=45)
plt.xlabel('Time (minutes, binned every 2 min)')
# Set y-ticks to participant codes
plt.yticks(range(len(participants)), participants)
plt.ylabel('Participant Code')
plt.title('All Participant Page Transitions Over Time (sorted) ' + session_id)
# Create a legend for page_name colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in all_page_names_presurvey]
# add label for line colors
line_labels = ['Scenario Answer: -1 Against', 'Scenario Answer: 1 For']
line_colors = ['#097969', '#400080'] 
handles += [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(line_colors, line_labels)]
plt.legend(handles=handles, title='Page Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()




# Plot
# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'participant_page_transitions_over_time_all_{session_id}.png')

plt.figure(figsize=(20, 6))
for i, participant in enumerate(participants):
    df_part = all_page_time[all_page_time['participant_code'] == participant]
    # Get the scenario answer for this participant
    scenario_row = scenario_answers[scenario_answers['participant.code'] == participant]
    if not scenario_row.empty:
        answer = scenario_row['presurvey.1.player.response'].values[0]
        if answer == -1:
            line_color = '#097969'
        elif answer == 1:
            line_color = '#400080'
        else:
            line_color = '#808080'
    else:
        line_color = '#808080'
    # Draw horizontal line for this participant, colored by scenario answer
    if not df_part.empty:
        min_time = df_part['time_bin'].min()
        max_time = df_part['time_bin'].max()
        plt.hlines(i, min_time, max_time, color=line_color, linewidth=4, alpha=0.5, zorder=1)
    # Scatter plot for nodes
    plt.scatter(
        df_part['time_bin'],
        np.full_like(df_part['time_bin'], i),
        c=[page_name_to_color[name] for name in df_part['page_name']],
        label=participant if i == 0 else "",  # Only label first for legend
        s=60,
        edgecolor='k',
        zorder=2
    )

# Set x-ticks to show every 2 minutes (120s)
min_bin = int(all_page_time['time_bin'].min())
max_bin = int(all_page_time['time_bin'].max())
step = 120  # 2 minutes in seconds
xticks = np.arange(min_bin, max_bin + step, step)
xtick_labels = [str(int(x // 60)) for x in xticks]  # show as minutes
plt.xticks(xticks, xtick_labels, rotation=45)
plt.xlabel('Time (minutes, binned every 2 min)')
# Set y-ticks to participant codes
plt.yticks(range(len(participants)), participants)
plt.ylabel('Participant Code')
plt.title('All Participant Page Transitions Over Time ' + session_id)
# Create a legend for page_name colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in all_page_names_presurvey]
# add label for line colors
line_labels = ['Scenario Answer: -1 Against', 'Scenario Answer: 1 For']
line_colors = ['#097969', '#400080'] 
handles += [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(line_colors, line_labels)]
plt.legend(handles=handles, title='Page Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



### Check the base pay participants responses in Scenario ###
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_' + session_id + '.csv')
# Filter the otree_data to only include relevant participants
base_pay_otree_data = otree_data[otree_data['participant.code'].isin(base_pay_participants_last_page['participant_code'])]
full_pay_otree_data = otree_data[otree_data['participant.code'].isin(full_participants_last_page['participant_code'])]
timed_out_otree_data = otree_data[otree_data['participant.code'].isin(timed_out_participants_last_page['participant_code'])]
# Get the participant codes and their answers in Scenario
base_pay_scenario_answers = base_pay_otree_data[['participant.code','presurvey.1.player.response']]
waiting_scenario_answers = waiting_otree_data[['participant.code','presurvey.1.player.response']]
full_pay_scenario_answers = full_pay_otree_data[['participant.code','presurvey.1.player.response']]
timed_out_scenario_answers = timed_out_otree_data[['participant.code','presurvey.1.player.response']]
# # Note: there is one nocode participant who answered the scenario question but is not in the base pay participants
# nocode_participant = '6734c699d11234d19b0be0ec'
# nocode_data = otree_data[otree_data['participant.label'] == nocode_participant]
# nocode_scenario_answer = nocode_data['presurvey.1.player.response']


# Prepare scenario answer counts for each group
base_pay_counts = base_pay_scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)
waiting_counts = waiting_scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)
full_pay_counts = full_pay_scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)
timed_out_counts = timed_out_scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)

# Bar positions
x = np.array([-1, 0, 1])

# Colors for answers
answer_colors = { -1: '#097969', 0: '#808080', 1: '#400080' }
bar_colors = [answer_colors[i] for i in x]

# Colors for groups
group_colors = {
    'Base Pay': 'tab:blue',
    'Waiting': 'tab:orange',
    'Nocode': 'tab:green'
}

plt.figure(figsize=(10, 6))

# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'scenario_answers_distribution_stacked_{session_id}.png')

# Plot stacked bars
bp = plt.bar(x, base_pay_counts.values, color=bar_colors, label='Base Pay', edgecolor=group_colors['Base Pay'],alpha=0.5,linewidth=3)
wp = plt.bar(x, waiting_counts.values, bottom=base_pay_counts.values, color=bar_colors, alpha=0.5, label='Waiting', edgecolor=group_colors['Waiting'],linewidth=3)
nc = plt.bar(x, full_pay_counts.values, bottom=base_pay_counts.values + waiting_counts.values, color=bar_colors, alpha=0.5, label='Nocode', 
edgecolor=group_colors['Nocode'],linewidth=3)
to = plt.bar(x, timed_out_counts.values, bottom=base_pay_counts.values + waiting_counts.values + full_pay_counts.values, color=bar_colors, alpha=0.5, label='Timed Out', edgecolor='black',linewidth=3)

plt.xlabel('Scenario Answer')
plt.ylabel('Count')
plt.ylim(0,20)
plt.title('Scenario Answers Distribution (Stacked)')
plt.xticks(x, ['Against', 'Neutral', 'For'])
plt.legend()
plt.tight_layout()
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



### Page sequence for full pay participants ###
full_pay_page_time

min_time = full_pay_page_time['epoch_time_completed'].min()
full_pay_page_time['time_from_start'] = full_pay_page_time['epoch_time_completed'] - min_time


# Bin time into 10s intervals
full_pay_page_time['time_bin'] = (full_pay_page_time['time_from_start'] // 10) * 10

# Get unique participant codes and page names
full_pay_participants = full_pay_page_time['participant_code'].unique() 
len(full_pay_participants) # 8 participants
full_pay_page_names = full_pay_page_time['page_name'].unique()
len(full_pay_page_names) # 56 pages


# Get all the page names except the excluded ones
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
relevant_page_names = [page for page in all_page_names if page not in excluded_pages]
full_pay_page_time_clean = full_pay_page_time[full_pay_page_time['page_name'].isin(relevant_page_names)]
len(full_pay_page_time_clean['participant_code'].unique().tolist()) # 8 participants



# Check each participant's answers in Scenario
# get data from the all_apps_wide_gneiwiau.csv file
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_' + session_id + '.csv')
# Filter the otree_data to only include relevant participants
full_pay_otree_data = otree_data[otree_data['participant.code'].isin(full_pay_participants)]
# Get the participant codes and their answers in Scenario
full_pay_scenario_answers = full_pay_otree_data[['participant.code','presurvey.1.player.response']]

# check column names from otree data
full_pay_otree_data.columns.to_list()
### Note: still doesn't work, need to fix that it only captures the relevant pages
# Assign a color to each page_name
colors = plt.cm.get_cmap('tab20', len(relevant_page_names))
page_name_to_color = {name: colors(i) for i, name in enumerate(relevant_page_names)}


# ensure output folder exists and save without cutting
output_dir = os.path.join(file_location, session_id)
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, f'participant_page_transitions_over_time_all_{session_id}.png')


plt.figure(figsize=(20, 6))
for i, participant in enumerate(full_pay_participants):
    df_part = full_pay_page_time_clean[full_pay_page_time_clean['participant_code'] == participant]
    # Get the scenario answer for this participant
    scenario_row = full_pay_scenario_answers[full_pay_scenario_answers['participant.code'] == participant]
    if not scenario_row.empty:
        answer = scenario_row['presurvey.1.player.response'].values[0]
        if answer == -1:
            line_color = '#097969'
        elif answer == 1:
            line_color = '#400080'
        else:
            line_color = '#808080'
    else:
        line_color = '#808080'
    # Draw horizontal line for this participant, colored by scenario answer
    if not df_part.empty:
        min_time = df_part['time_bin'].min()
        max_time = df_part['time_bin'].max()
        plt.hlines(i, min_time, max_time, color=line_color, linewidth=4, alpha=0.5, zorder=1)
    # Scatter plot for nodes
    plt.scatter(
        df_part['time_bin'],
        np.full_like(df_part['time_bin'], i),
        c=[page_name_to_color[name] for name in df_part['page_name']],
        label=participant if i == 0 else "",  # Only label first for legend
        s=60,
        edgecolor='k',
        zorder=2
    )

# Set x-ticks to show every 2 minutes (120s)
min_bin = int(full_pay_page_time_clean['time_bin'].min())
max_bin = int(full_pay_page_time_clean['time_bin'].max())
step = 120  # 2 minutes in seconds
xticks = np.arange(min_bin, max_bin + step, step)
xtick_labels = [str(int(x // 60)) for x in xticks]  # show as minutes
plt.xticks(xticks, xtick_labels, rotation=45)
plt.xlabel('Time (minutes, binned every 2 min)')
# Set y-ticks to participant codes
plt.yticks(range(len(full_pay_participants)), full_pay_participants)
plt.ylabel('Participant Code')
plt.title('All Participant Page Transitions Over Time ' + session_id)
# Create a legend for page_name colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in relevant_page_names]
# add label for line colors
line_labels = ['Scenario Answer: -1 Against', 'Scenario Answer: 1 For']
line_colors = ['#097969', '#400080'] 
handles += [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(line_colors, line_labels)]
plt.tight_layout(rect=[0, 0.15, 1, 1])  # leave space at bottom for legend
plt.legend(
    handles=handles,
    title='Page Name',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.5),
    ncol=10,
    frameon=True,
    fontsize=12,
    title_fontsize=14
)
# standard save that preserves outside legend/labels
plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

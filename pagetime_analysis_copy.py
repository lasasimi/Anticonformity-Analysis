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
page_time = pd.read_csv(file_location + 'Raw_otree/PageTimes-gneiwiau.csv')

# Merge the page time data
# page_time = pd.concat([page_time_1, page_time_2, page_time_3], ignore_index=True)

participant_code = page_time['participant_code'].unique().tolist()
len(participant_code) # 20 total participants

# get unique page names
page_time['page_name'].unique().tolist() 

# Filter the rows to only include those participants who are commited and waiting (Page Name is GroupSizeWaitPage)
waiting_label = ['670fe98d7026b251962d9024',
                  '66644b493ad8e4efaadd70ef',
                  '617646d656a0115ca27ec794',
                  '5bc9f8b677740000016a08b5',
                  '66294001494b75def82c92c7',
                  '6734c699d11234d19b0be0ec',]
                 

base_pay_label = ['5c8db07eb1f89a0016777fe0',
                  '66a8f367f59edf05be5b8ed2',
                  '607668ac450fc6e3025a131d',
                  '65ee224fc6146bbd065a8ae0',
                  '66463e3ccc2fcde4d3fabf53',
                  '67d55f1a8d7ba521684d602e',
                  '56bfcce79f7a1e0005fdca9e',
                  '650b03136ab3d4c832d98b71']                



# match the participant code with Prolific participant label
prolific_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_gneiwiau.csv')
prolific_data = prolific_data.loc[:, ['participant.code', 'participant.label']]

prolific_data['participant.label'].unique().tolist() # 20 participants  
# Get the participant code and label for base_pay and complete participants
base_pay_participants_code = prolific_data[prolific_data['participant.label'].isin(base_pay_label)]
waiting_participants_code = prolific_data[prolific_data['participant.label'].isin(waiting_label)]

# Get the last page name of each participant in the waiting participants code
page_time['last_page_name'] = page_time.groupby('participant_code')['page_name'].transform('last')
base_pay_participants_last_page = page_time[page_time['participant_code'].isin(base_pay_participants_code['participant.code'])][['participant_code', 'last_page_name']]
waiting_participants_last_page = page_time[page_time['participant_code'].isin(waiting_participants_code['participant.code'])][['participant_code', 'last_page_name']]

# Filter only the unique participant codes
base_pay_participants_last_page = base_pay_participants_last_page.drop_duplicates(subset=['participant_code'])
waiting_participants_last_page = waiting_participants_last_page.drop_duplicates(subset=['participant_code'])

# Combine with participant label from the waiting_participants_code
base_pay_participants_last_page = base_pay_participants_last_page.merge(base_pay_participants_code, left_on='participant_code', right_on='participant.code', how='left')
waiting_participants_last_page = waiting_participants_last_page.merge(waiting_participants_code, left_on='participant_code', right_on='participant.code', how='left')


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
all_page_names

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
participant = '9qs759en'
time_differences = compute_page_time_differences(page_time, all_page_names, participant)
print(f"Time differences for participant {participant}: {time_differences}")

# Exclude Training_Neighbor_2 and Training_Neighbor_3 and AttentionCheck pages
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
analysed_page_names = [page for page in all_page_names if page not in excluded_pages]

# Create a dictionary to store time differences for each participant
participant_time_differences = {}

for participant in participant_code:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(waitpart_page_time, analysed_page_names, participant)
    
    # Store the result in the dictionary
    participant_time_differences[participant] = time_differences

# Compute the average and standard deviation across all participants
num_pages = len(analysed_page_names) - 1  # Number of page transitions
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
for i in analysed_page_names[1:]: # Exclude the first page for x-axis
    x_values.append(i)

plt.figure(figsize=(20, 6))
# Plot the average differences with error bars
plt.errorbar(x_values, average_differences, yerr=std_differences, fmt='-o', capsize=5, label='Average Time Differences')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis 

# Annotate each point with its value
for i, (x, y) in enumerate(zip(x_values, average_differences)):
    plt.text(x, y, f'{y:.2f}', fontsize=10, ha='center', va='bottom')  # Display value with 2 decimal places

# Add labels and title
plt.xlabel('Page Time in app')
plt.show()
plt.savefig('average_time_differences_all_pages_gneiwiau.png')
plt.close()



###### Plotting the page transitions over time for each participant ######
# Filter the page_time DataFrame to only include waiting participants
waitpart_page_time
analysed_page_names

min_time = waitpart_page_time['epoch_time_completed'].min()
waitpart_page_time['time_from_start'] = waitpart_page_time['epoch_time_completed'] - min_time


# Bin time into 10s intervals
waitpart_page_time['time_bin'] = (waitpart_page_time['time_from_start'] // 10) * 10

# Get unique participant codes and page names
participants = waitpart_page_time['participant_code'].unique() 
len(participants) # 6 participants
page_names = page_time['page_name'].unique()
len(page_names) # 12 pages

# Remove unnecessary pages from all_page_names
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
relevant_page_names = [page for page in page_names if page not in excluded_pages]

# Check each participant's answers in Scenario
# get data from the all_apps_wide_gneiwiau.csv file
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_gneiwiau.csv')
# Filter the otree_data to only include relevant participants
waiting_otree_data = otree_data[otree_data['participant.code'].isin(participants)]
# Get the participant codes and their answers in Scenario
scenario_answers = waiting_otree_data[['participant.code','presurvey.1.player.response']]

# check column names from otree data
otree_data.columns.to_list()
### Note: still doesn't work, need to fix that it only captures the relevant pages
# Assign a color to each page_name
colors = plt.cm.get_cmap('tab20', len(relevant_page_names))
page_name_to_color = {name: colors(i) for i, name in enumerate(relevant_page_names)}


plt.figure(figsize=(20, 6))
for i, participant in enumerate(participants):
    df_part = waitpart_page_time[waitpart_page_time['participant_code'] == participant]
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
min_bin = int(waitpart_page_time['time_bin'].min())
max_bin = int(waitpart_page_time['time_bin'].max())
step = 120  # 2 minutes in seconds
xticks = np.arange(min_bin, max_bin + step, step)
xtick_labels = [str(int(x // 60)) for x in xticks]  # show as minutes
plt.xticks(xticks, xtick_labels, rotation=45)
plt.xlabel('Time (minutes, binned every 2 min)')
# Set y-ticks to participant codes
plt.yticks(range(len(participants)), participants)
plt.ylabel('Participant Code')
plt.title('Participant Page Transitions Over Time')
# Create a legend for page_name colors
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in relevant_page_names]
# add label for line colors
line_labels = ['Scenario Answer: -1 Against', 'Scenario Answer: 1 For']
line_colors = ['#097969', '#400080'] 
handles += [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(line_colors, line_labels)]
plt.legend(handles=handles, title='Page Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.savefig('participant_page_transitions_over_time_gneiwiau.png')
plt.show()


### Check the base pay participants responses in Scenario ###
# get data from the all_apps_wide_gneiwiau.csv file
otree_data = pd.read_csv(file_location + 'Raw_otree/all_apps_wide_gneiwiau.csv')
# Filter the otree_data to only include relevant participants
base_pay_otree_data = otree_data[otree_data['participant.code'].isin(base_pay_participants_last_page['participant_code'])]
# Get the participant codes and their answers in Scenario
base_pay_scenario_answers = base_pay_otree_data[['participant.code','presurvey.1.player.response']]

# Note: there is one nocode participant who answered the scenario question but is not in the base pay participants
nocode_participant = '6734c699d11234d19b0be0ec'
nocode_data = otree_data[otree_data['participant.label'] == nocode_participant]
nocode_scenario_answer = nocode_data['presurvey.1.player.response']


# Prepare scenario answer counts for each group
base_pay_counts = base_pay_scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)
waiting_counts = scenario_answers['presurvey.1.player.response'].value_counts().reindex([-1, 0, 1], fill_value=0)
nocode_counts = nocode_scenario_answer.value_counts().reindex([-1, 0, 1], fill_value=0)
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

# Plot stacked bars
bp = plt.bar(x, base_pay_counts.values, color=bar_colors, label='Base Pay', edgecolor=group_colors['Base Pay'],alpha=0.5,linewidth=3)
wp = plt.bar(x, waiting_counts.values, bottom=base_pay_counts.values, color=bar_colors, alpha=0.5, label='Waiting', edgecolor=group_colors['Waiting'],linewidth=3)
nc = plt.bar(x, nocode_counts.values, bottom=base_pay_counts.values + waiting_counts.values, color=bar_colors, alpha=0.5, label='Nocode', 
edgecolor=group_colors['Nocode'],linewidth=3)

plt.xlabel('Scenario Answer')
plt.ylabel('Count')
plt.ylim(0,10)
plt.title('Scenario Answers Distribution (Stacked)')
plt.xticks(x, ['Against', 'Neutral', 'For'])
plt.legend()
plt.tight_layout()
plt.show()
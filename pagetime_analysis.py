import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os 

print(os.getcwd())

# Load and merge the page time data
file_location = '/Users/lasmimarbun/Documents/Git/Anticonformity-Analysis/'
page_time_1 = pd.read_csv(file_location + 'Raw_otree/PageTimes-k1hhm7lf.csv')
page_time_2 = pd.read_csv(file_location + 'Raw_otree/PageTimes-pfu1qwfx.csv')
page_time_3 = pd.read_csv(file_location + 'Raw_otree/PageTimes-7agw525e.csv')

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
participant = 'gvjieukk'
time_differences = compute_page_time_differences(page_time, all_page_names, participant)
print(f"Time differences for participant {participant}: {time_differences}")

# Exclude Training_Neighbor_2 and Training_Neighbor_3 and AttentionCheck pages
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
analysed_page_names = [page for page in all_page_names if page not in excluded_pages]

# Create a dictionary to store time differences for each participant
participant_time_differences = {}

for participant in participant_code:
    # Compute time differences for the current participant
    time_differences = compute_page_time_differences(page_time, analysed_page_names, participant)
    
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
plt.savefig('average_time_differences_all_pages.png')
plt.close()



###### Plotting the page transitions over time for each participant ######
# Load page time data

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

# If round_number is >1, append the round number to the page name and create a new row for that participant
page_time['page_name'] = page_time.apply(
    lambda row: f"{row['page_name']}_{row['round_number']}" if row['round_number'] > 1 else row['page_name'],
    axis=1
)

all_page_names = page_time['page_name'].unique()
all_page_names

# Slice the page_time to 3 based on the session code
page_time_1 = page_time[page_time['session_code'] == 'k1hhm7lf']
page_time_2 = page_time[page_time['session_code'] == 'pfu1qwfx']
page_time_3 = page_time[page_time['session_code'] == '7agw525e']

# Normalize time to start at 0 for different sessions
min_time_1 = page_time_1['epoch_time_completed'].min()
min_time_2 = page_time_2['epoch_time_completed'].min()
min_time_3 = page_time_3['epoch_time_completed'].min()
page_time_1['time_from_start'] = page_time_1['epoch_time_completed'] - min_time_1
page_time_2['time_from_start'] = page_time_2['epoch_time_completed'] - min_time_2
page_time_3['time_from_start'] = page_time_3['epoch_time_completed'] - min_time_3

min_time = page_time['epoch_time_completed'].min()
page_time['time_from_start'] = page_time['epoch_time_completed'] - min_time


# Bin time into 10s intervals
page_time['time_bin'] = (page_time['time_from_start'] // 10) * 10

# Get unique participant codes and page names
participants = page_time['participant_code'].unique() 
len(participants) # 121 participants (should be 120!)
page_names = page_time['page_name'].unique()
len(page_names) # 35 pages

# Merge participants with their treatment type 
otree_data = pd.read_csv(file_location + 'Clean_files/clean_long_format_merged.csv')
sliced_data = otree_data.loc[:, ['participant.code', 'participant.treatment']]

# Match participants with their treatment type
otree_data_map = dict(zip(sliced_data['participant.code'], sliced_data['participant.treatment']))

# Define a color for each treatment type
treatment_types = list(set(otree_data_map.values()))
# Add 'Unknown' to handle participants not in the treatment data
treatment_types.append('Unknown')
treatment_colors = plt.cm.get_cmap('Set1', len(treatment_types))
treatment_to_colors = {treatment: treatment_colors(i) for i, treatment in enumerate(treatment_types)}

# Remove unnecessary pages from all_page_names
excluded_pages = ['TrainingNeighbor_2', 'TrainingNeighbor_3', 'AttentionCheck']
relevant_page_names = [page for page in all_page_names if page not in excluded_pages]

### Note: still doesn't work, need to fix that it only captures the relevant pages
# Assign a color to each page_name
colors = plt.cm.get_cmap('tab20', len(relevant_page_names))
page_name_to_color = {name: colors(i) for i, name in enumerate(relevant_page_names)}


plt.figure(figsize=(20, 8))

for i, participant in enumerate(participants):
    df_part = page_time_1[page_time_1['participant_code'] == participant]
    # Draw horizontal line for this participant, colored by completion status
    if not df_part.empty:
        min_time = df_part['time_bin'].min()
        max_time = df_part['time_bin'].max()
        treatment = otree_data_map.get(participant, 'Unknown')
        plt.hlines(i, min_time, max_time, color=treatment_to_colors[treatment], linewidth=4, alpha=0.5, zorder=1)
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
min_bin = int(page_time_1['time_bin'].min())
max_bin = int(page_time_1['time_bin'].max())
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
handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=page_name_to_color[name], markersize=10) for name in page_names]
legend1 = plt.legend(handles=handles, title='Page Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().add_artist(legend1)
# Add a legend for completion status
handles2 = [plt.Line2D([0], [0], color=treatment_to_colors[status], lw=4, label=status) for status in treatment_types]
plt.legend(handles=handles2, title='Treatment Type', bbox_to_anchor=(1.05, 0.15), loc='lower left')
plt.tight_layout()
# plt.savefig('participant_page_transitions_over_time.png')
plt.show()

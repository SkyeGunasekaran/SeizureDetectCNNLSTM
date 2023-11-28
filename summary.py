# Define a dictionary to store channels and their corresponding status (changed or not)
channel_status = {}

# Function to process channel information in each file
def process_channels(channel_list, file_name):
    global channel_status
    for channel in channel_list:
        if channel not in channel_status:
            channel_status[channel] = set()
        channel_status[channel].add(file_name)

# Function to print intersecting channels
def print_intersecting_channels():
    global channel_status
    intersecting_channels = [channel for channel, files in channel_status.items() if len(files) > 1]
    print("Intersecting Channels:")
    print(intersecting_channels)

# Read the seizure summary file
with open("Dataset/chb12/chb12-summary.txt", "r") as file:
    current_file = None
    for line in file:
        line = line.strip()
        if line.startswith("File Name:"):
            # Extract file name
            current_file = line.split(":")[1].strip()
        elif line.startswith("Channels changed:"):
            # Extract and process changed channels
            changed_channels = [channel.strip() for channel in line.split(":")[1].split()]
            process_channels(changed_channels, current_file)

# Print intersecting channels
print_intersecting_channels()

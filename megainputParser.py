import os
import re

# Directory where the .txt files are stored
input_dir = "./data/FullDiscordDMS/"

# Directory where the output file will be stored
output_dir = "./output/"

# Name of the output file
output_file = "samples.txt"

# Regex pattern to match the message lines in the input files
message_pattern = re.compile(r'\[\d{2}-\w{3}-\d{2} \d{2}:\d{2} (?:AM|PM)\] .+#\d{4}\n(.*)')

# Function to clean up a message
def clean_message(message):
    # Add your cleaning steps here. For example, you might want to remove certain words or symbols.
    # For now, we just strip leading/trailing whitespace
    return message.strip()

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the output file
with open(os.path.join(output_dir, output_file), 'w') as out_file:
    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            # Open the current file
            with open(os.path.join(input_dir, filename), 'r') as in_file:
                # Read the file line by line
                for line in in_file:
                    # Check if the line matches the message pattern
                    match = message_pattern.match(line)
                    if match:
                        # If it does, clean the message and write it to the output file
                        cleaned_message = clean_message(match.group(1))
                        out_file.write(cleaned_message + '\n')

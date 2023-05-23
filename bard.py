import re
import os

def parse_discord_messages(file_name):
  """Parses the Discord messages in a file and returns a list of messages.

  Args:
    file_name: The name of the file to parse.

  Returns:
    A list of messages. Each message is a string.
  """

  messages = []
  filepath = os.path.join("./data/FullDiscordDMS/", file_name)
  with open(filepath, "r") as f:
    for line in f:
      if re.match(r"^\[(.*?)\] (.*?)$", line):
        messages.append(line.split()[1])

  return messages

def clean_messages(messages):
  """Cleans the Discord messages by removing the date and username.

  Args:
    messages: A list of messages. Each message is a string.

  Returns:
    A list of cleaned messages.
  """

  cleaned_messages = []
  for message in messages:
    cleaned_messages.append(re.sub(r"^\[(.*?)\] (.*?)$", r"\2", message))

  return cleaned_messages

def write_messages_to_file(messages, file_name):
  """Writes the cleaned messages to a file.

  Args:
    messages: A list of cleaned messages.
    file_name: The name of the file to write to.
  """

  with open(file_name, "w") as f:
    for message in messages:
      f.write(message + "\n")

def main():
  input_file_names = []
  for file_name in os.listdir("./data/FullDiscordDMS/"):
    if file_name.endswith(".txt"):
      input_file_names.append(file_name)

  messages = []
  for input_file_name in input_file_names:
    messages += parse_discord_messages(input_file_name)

  cleaned_messages = clean_messages(messages)
  write_messages_to_file(cleaned_messages, "output.txt")

if __name__ == "__main__":
  main()

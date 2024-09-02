import subprocess

# Get the list of installed packages
result = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)

# Write the output to requirements.txt
with open('requirements.txt', 'w') as f:
    f.write(result.stdout)

print("requirements.txt file has been generated.")
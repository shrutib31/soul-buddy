import requests

response = requests.get("http://srv940034.hstgr.cloud:11434/v1")  # paste your actual URL
print(response.json())
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
filename = "winequality-red.csv"

response = requests.get(url)
with open(filename, 'wb') as f:
    f.write(response.content)

print(f"✅ Đã tải xong file: {filename}")

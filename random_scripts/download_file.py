import urllib.request

url = "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt"
output_path = "DB100K_test.txt"  # change if you want a different filename or path

urllib.request.urlretrieve(url, output_path)

print(f"Downloaded to {output_path}")

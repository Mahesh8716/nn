import requests

# URL of your local server (or public URL if hosted)
url = "http://127.0.0.1:4000/predict"

# Path to a sample image to test (you can use any image of a 3 or 7)
image_path = "mnist_sample/train/3/7.png"  # Replace with actual image path

# Open the image file in binary mode
with open(image_path, 'rb') as img_file:
    files = {'file': img_file}
    
    # Send the POST request to the /predict endpoint
    response = requests.post(url, files=files)

# Print the server's response
if response.status_code == 200:
    print("Prediction Response:", response.json())
else:
    print(f"Failed to get prediction. Status Code: {response.status_code}")
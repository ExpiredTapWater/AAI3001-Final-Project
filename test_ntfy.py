import requests

try:
    topic = "AAI3001-FinalProj"
    title = "Notification Test"
    message = "Go and look at your printer please!"

    response = requests.post(
        f"https://ntfy.sh/{topic}",  # Topic directly in the URL
        headers={"Title": title},  # Notification title
        data=message  # Notification message
    )

    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification: {response.status_code}")
except Exception as e:
    print(f"Error sending notification: {e}")

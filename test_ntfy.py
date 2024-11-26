import ntfy

# Test the NTFY notification functionality
try:
    ntfy.notify(
        topic="AAI3001-FinalProj",  # Use the exact topic name as in the app
        title="Test Notification",
        message="This is a test notification.",
        priority="high"
    )
    print("Notification sent successfully!")
except Exception as e:
    print(f"Failed to send notification: {e}")

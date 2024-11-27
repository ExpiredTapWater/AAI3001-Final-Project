import streamlit as st

# Title for the Streamlit app
st.title("MJPEG Stream in Streamlit")

# Embed the MJPEG stream using an iframe
mjpeg_stream_url = "http://127.0.0.1:3001/"
iframe_code = f"""
<iframe src="{mjpeg_stream_url}" width="640" height="480" frameborder="0" allowfullscreen></iframe>
"""
st.components.v1.html(iframe_code, height=480)

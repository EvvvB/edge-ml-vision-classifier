# Copy this file to wifi_config.py on the Nicla/OpenMV filesystem and
# fill in your local test network settings.
#
# Important: API_URL must use your Mac's LAN/Wi-Fi IP address, not 127.0.0.1.
# Find it on macOS with:
#
#   ipconfig getifaddr en0
#
# Then run the API on your Mac with --host 0.0.0.0 so other devices on the
# same Wi-Fi can reach it.

WIFI_SSID = "your-wifi-name"
WIFI_PASSWORD = "your-wifi-password"

API_URL = "http://192.168.1.50:8000/detections"

DEVICE_ID = "nicla-vision-01"

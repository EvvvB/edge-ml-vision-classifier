# Minimal Nicla/OpenMV Wi-Fi test.
#
# Save this file to the Nicla as main.py temporarily, alongside wifi_config.py.
# It only tests joining Wi-Fi and opening a TCP connection to API_URL.

import network
import socket
import time

from wifi_config import WIFI_SSID, WIFI_PASSWORD, API_URL


WIFI_CONNECT_TIMEOUT_MS = 20000
SOCKET_TIMEOUT_SECONDS = 5


def parse_http_url(url):
    if not url.startswith("http://"):
        raise ValueError("Only http:// API_URL values are supported")

    remainder = url[len("http://"):]
    slash_index = remainder.find("/")

    if slash_index < 0:
        host_port = remainder
        path = "/"
    else:
        host_port = remainder[:slash_index]
        path = remainder[slash_index:] or "/"

    if ":" in host_port:
        host, port_text = host_port.rsplit(":", 1)
        port = int(port_text)
    else:
        host = host_port
        port = 80

    return host, port, path


def connect_wifi():
    wlan = network.WLAN(network.WLAN.IF_STA)
    wlan.active(True)

    print("Connecting Wi-Fi:", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    started_at = time.ticks_ms()

    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), started_at) > WIFI_CONNECT_TIMEOUT_MS:
            print("Wi-Fi connection timed out.")
            return None

        time.sleep_ms(250)

    print("Wi-Fi connected:", wlan.ifconfig())
    return wlan


def test_api_socket():
    host, port, path = parse_http_url(API_URL)
    print("Testing API socket:", host, port, path)

    address = socket.getaddrinfo(host, port)[0][-1]
    sock = socket.socket()
    sock.settimeout(SOCKET_TIMEOUT_SECONDS)

    try:
        sock.connect(address)
        print("API TCP connection succeeded.")

    finally:
        sock.close()


wlan = connect_wifi()

if wlan:
    test_api_socket()

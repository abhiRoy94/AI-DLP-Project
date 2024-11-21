import scapy.all as scapy
from scapy.all import get_if_list
import re
import os
from plyer import notification

# Define a function to detect sensitive information
def detect_sensitive_data(packet):
    # Example regex pattern for PII (e.g., email addresses)
    sensitive_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    
    if packet.haslayer(scapy.Raw):
        payload = packet[scapy.Raw].load.decode(errors='ignore')
        
        # Check for sensitive data using regex
        if sensitive_pattern.search(payload):
            print(f"Sensitive data detected: {payload}")
            notify_user("Sensitive data detected!")
            block_network_traffic()
            return True

    return False

# Function to block network transmission (dummy implementation)
def block_network_traffic():
    # In a real-world scenario, you might add firewall rules or terminate the connection
    os.system("netsh advfirewall firewall add rule name=\"BlockSensitiveData\" dir=out action=block")

# Function to notify user of the detected sensitive data
def notify_user(message):
    notification.notify(
        title="Alert: Sensitive Data Detected",
        message=message,
        timeout=5
    )

# Sniff network traffic and check for sensitive data
def sniff_packets(interface):
    scapy.sniff(iface=interface, store=False, prn=detect_sensitive_data)


# Run packet sniffer on the network interface (e.g., 'Wi-Fi' or 'Ethernet')
sniff_packets("Wi-Fi")

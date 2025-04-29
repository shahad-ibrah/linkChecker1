# utils.py (Helper functions)
import re
from urllib.parse import urlparse
import ipaddress
import os

def get_tld(url):
    """Extracts the Top-Level Domain (TLD) from a URL."""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname:
            parts = hostname.split('.')
            if len(parts) > 1 and parts[-1]:
                return parts[-1]
        return "" 
    except Exception:
        return ""

def is_ip_address(url):
    """Checks if the hostname part of a URL is an IP address."""
    try:
        hostname = urlparse(url).hostname
        if hostname:
            ipaddress.ip_address(hostname)
            return 1
        return 0
    except ValueError:
        return 0
    except Exception:
        return 0

def extract_url_features(url):
    """Extracts URLLength, IsDomainIP, and TLD from a URL."""
    url_length = len(url)
    is_domain_ip = is_ip_address(url)
    tld = get_tld(url) if is_domain_ip == 0 else "" 
    return {
        'URL': url,
        'URLLength': url_length,
        'IsDomainIP': is_domain_ip,
        'TLD': tld
    }
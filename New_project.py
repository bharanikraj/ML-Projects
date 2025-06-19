import pandas as pd
import csv
print(pd.__file__)


def detect_delimiter(file_path, sample_size=1024):
    """Automatically detect the delimiter in a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        sample = file.read(sample_size)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter
    
detect_delimiter(f"C:\Users\bharani\Downloads\bank+marketing\bank\bank.csv",sample_size=1024)
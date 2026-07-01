# Preprocessing script for network traffic dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(filepath):
    df = pd.read_csv(filepath)
    le_protocol = LabelEncoder()
    df['Protocol_enc'] = le_protocol.fit_transform(df['Protocol'].fillna('Unknown'))
    le_src = LabelEncoder()
    le_dst = LabelEncoder()
    df['Source_enc'] = le_src.fit_transform(df['Source'].fillna('Unknown'))
    df['Destination_enc'] = le_dst.fit_transform(df['Destination'].fillna('Unknown'))
    df['src_port'] = df['Info'].str.extract(r'(\d+)\s*>').astype(float).fillna(0)
    df['dst_port'] = df['Info'].str.extract(r'>\s*(\d+)').astype(float).fillna(0)
    X = df[['No.','Time','Length','Protocol_enc','Source_enc','Destination_enc','src_port','dst_port']]
    y = le_protocol.fit_transform(df['Protocol'].fillna('Unknown'))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

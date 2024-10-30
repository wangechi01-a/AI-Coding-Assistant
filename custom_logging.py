import pandas as pd
import os
from datetime import datetime
from config import LOG_FILE

def log_interaction(query, response, feedback=None):
    # Ensure the log file exists
    if not os.path.isfile(LOG_FILE):
        # Create a new DataFrame and save it if the file doesn't exist
        df = pd.DataFrame(columns=["timestamp", "query", "response", "feedback"])
        df.to_csv(LOG_FILE, index=False)
    
    # Prepare the data to log
    timestamp = datetime.now().isoformat()
    data = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "feedback": feedback
    }
    
    # Append the interaction to the CSV
    df = pd.read_csv(LOG_FILE)
    df = df.append(data, ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

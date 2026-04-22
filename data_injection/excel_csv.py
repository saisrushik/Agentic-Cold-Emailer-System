import numpy as np
import pandas as pd

def read_csv(file_path):
    try:
        with open(file_path) as file:
            df = pd.read_csv(file) 
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def get_hr_names(df):
    """
    Extracts the HR team name from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The HR team name if found, otherwise None.
    """
    try:
        hr_team_name = df['HR Name']
        return hr_team_name.tolist()
    except Exception as e:
        print(f"Error extracting HR team name: {e}")
        return None

def get_emails(df):
    """
    Extracts the HR team name from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The HR team name if found, otherwise None.
    """
    try:
        hr_team_name = df['Email']
        return hr_team_name.tolist()
    except Exception as e:
        print(f"Error extracting HR team name: {e}")
        return None 

def get_companies(df):
    """
    Extracts the HR team name from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The HR team name if found, otherwise None.
    """
    try:
        company_name = df['Company']
        return company_name.tolist()
    except Exception as e:
        print(f"Error extracting company name: {e}")
        return None 

def get_hiring_roles(df):
    """
    Extracts the HR team name from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The HR team name if found, otherwise None.
    """
    try:
        hr_team_name = df['Hiring Role']
        return hr_team_name.tolist()
    except Exception as e:
        print(f"Error extracting HR team name: {e}")
        return None 

def get_last_email_sent_dates(df):
    """
    Extracts the HR team name from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The HR team name if found, otherwise None.
    """
    try:
        hr_team_name = df['Last Email Sent Date']
        return hr_team_name.tolist()
    except Exception as e:
        print(f"Error extracting HR team name: {e}")
        return None     

def get_callback_status(df):
    try:
        callback_status = df['Received Callback']
        return callback_status.tolist()
    except Exception as e:
        print(f"Error extracting callback status: {e}")
        return None 
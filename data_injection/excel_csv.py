import numpy as np
import pandas as pd

class ExcelCsvReader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_csv(self):
        try:
            with open(self.file_path) as file:
                df = pd.read_csv(file) 
            return df
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def get_hr_names(self):
        """
        Extracts the HR team name from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        str: The HR team name if found, otherwise None.
        """
        try:
            hr_team_name = self.df['HR Name']
            return hr_team_name.tolist()
        except Exception as e:
            print(f"Error extracting HR team name: {e}")
            return None

    def get_emails(self):
        """
        Extracts the HR team name from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        str: The HR team name if found, otherwise None.
        """
        try:
            hr_team_name = self.df['Email']
            return hr_team_name.tolist()
        except Exception as e:
            print(f"Error extracting HR team name: {e}")
            return None 

    def get_companies(self):
        """
        Extracts the HR team name from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        str: The HR team name if found, otherwise None.
        """
        try:
            company_name = self.df['Company']
            return company_name.tolist()
        except Exception as e:
            print(f"Error extracting company name: {e}")
            return None 

    def get_hiring_roles(self):
        """
        Extracts the HR team name from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        str: The HR team name if found, otherwise None.
        """
        try:
            hr_team_name = self.df['Hiring Role']
            return hr_team_name.tolist()
        except Exception as e:
            print(f"Error extracting HR team name: {e}")
            return None 

    def get_last_email_sent_dates(self):
        """
        Extracts the HR team name from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
        str: The HR team name if found, otherwise None.
        """
        try:
            hr_team_name = self.df['Last Email Sent Date']
            return hr_team_name.tolist()
        except Exception as e:
            print(f"Error extracting HR team name: {e}")
            return None     

    def get_callback_status(self):
        try:
            callback_status = self.df['Received Callback']
            return callback_status.tolist()
        except Exception as e:
            print(f"Error extracting callback status: {e}")
            return None 
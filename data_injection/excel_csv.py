import numpy as np
import pandas as pd


class ExcelCsvReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def read_csv(self):
        try:
            self.df = pd.read_csv(self.file_path)
            return self.df
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
            return None
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def get_hr_names(self):
        try:
            return self.df['HR Name/Team'].tolist()
        except Exception as e:
            print(f"Error extracting HR names: {e}")
            return None

    def get_emails(self):
        try:
            return self.df['Email'].tolist()
        except Exception as e:
            print(f"Error extracting emails: {e}")
            return None

    def get_companies(self):
        try:
            return self.df['Company'].tolist()
        except Exception as e:
            print(f"Error extracting companies: {e}")
            return None

    def get_company_types(self):
        try:
            return self.df['Company_Type'].tolist()
        except Exception as e:
            print(f"Error extracting company types: {e}")
            return None

    def get_hiring_roles(self):
        try:
            return self.df['Hiring Role'].tolist()
        except Exception as e:
            print(f"Error extracting hiring roles: {e}")
            return None

    def get_last_email_sent_dates(self):
        try:
            return self.df['Last Email Sent Date'].tolist()
        except Exception as e:
            print(f"Error extracting last email sent dates: {e}")
            return None

    def get_callback_status(self):
        try:
            return self.df['Received Callback'].tolist()
        except Exception as e:
            print(f"Error extracting callback status: {e}")
            return None

    def get_all_records(self):
        try:
            return self.df.to_dict(orient='records')
        except Exception as e:
            print(f"Error getting all records: {e}")
            return None
from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf as fitz  # PyMuPDF >= 1.24
import os
import re
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"


class ResumeReader:
    def __init__(self):
        pass

    def read_resume(self, file_path: str):
        """
        Load resume PDF using PyMuPDFLoader (LangChain) and enrich each
        page's content with the hyperlinks found on that page.
        """
        try:
            # --- LangChain loader: returns one Document per page ---
            loader = PyMuPDFLoader(file_path)
            resume_data = loader.load()

            # --- PyMuPDF: extract links per page ---
            pdf_doc = fitz.open(file_path)

            for page_index, page in enumerate(pdf_doc):
                links = page.get_links()  # list of dicts with 'uri' key for hyperlinks
                urls = [link["uri"] for link in links if link.get("uri")]

                if urls and page_index < len(resume_data):
                    hyperlink_section = (
                        "\n\nHyperlinks on this page:\n"
                        + "\n".join(f"- {url}" for url in urls)
                    )
                    resume_data[page_index].page_content += hyperlink_section
                    resume_data[page_index].metadata["hyperlinks"] = urls

            pdf_doc.close()
            return resume_data

        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            return None
        except Exception as e:
            print(f"Error reading resume file: {e}")
            return None
    
    def extract_email(self, mailto_links):
        for link in mailto_links:
            decoded = unquote(link)                      # decode %40 -> @, etc.
            match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', decoded)
            if match:
                return match.group()
        return None

    def extract_phone(self, mailto_links):
        for link in mailto_links:
            decoded = unquote(link)                      # decode %20 -> space, etc.
            match = re.search(r'[\+\d][\d\s\-]{8,14}\d', decoded)
            if match:
                return re.sub(r'\s+', '', match.group())
        return None


    def extract_contact_info(self, resume_data):
        web_links = [link for link in resume_data[0].metadata['hyperlinks'] if link.startswith("http")]
        mail_links = [l for l in resume_data[0].metadata['hyperlinks'] if l.startswith("mailto:")]

        links_dict = {
            "email":    self.extract_email(mail_links),
            "phone":    self.extract_phone(mail_links),
            "linkedin": next((l for l in web_links if "linkedin.com" in l), None),
            "github":   next((l for l in web_links if "github.com" in l), None),
            "portfolio": next((l for l in web_links if "sites.google.com" in l), None),
            "scholar":  next((l for l in web_links if "scholar.google.com" in l), None),
        }

        return links_dict


    
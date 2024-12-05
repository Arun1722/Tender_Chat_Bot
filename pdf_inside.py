import fitz  # PyMuPDF to extract text from the PDF
import requests
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import io
import re
from fpdf import FPDF

# Step 1: Extract Links from the Original PDF
def extract_links_from_pdf(pdf_path):
    links = []
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        link_dict = page.get_links()
        for link in link_dict:
            if 'uri' in link:
                # Only extract PDF links
                if link['uri'].endswith('.pdf'):
                    links.append(link['uri'])
    return links

# Step 2: Process Each PDF Link
def process_pdf_links(links):
    extracted_pdfs = []
    for link in links:
        pdf_data = extract_pdf_from_url(link)
        if pdf_data:  # Only add if valid content is extracted
            extracted_pdfs.append(pdf_data)
    return extracted_pdfs

# Function to extract PDF data from a URL
def extract_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content  # Return the raw bytes of the PDF
    else:
        return None

# Step 3: Append Extracted Data to the Original PDF
def append_data_to_pdf(original_pdf, extracted_pdfs, output_pdf):
    merger = PdfMerger()

    # Get filtered pages as a PDF stream
    filtered_pdf_stream = filter_pdf_pages_to_stream(original_pdf)
    merger.append(filtered_pdf_stream)

    # For each extracted PDF, append it to the original PDF
    for pdf_content in extracted_pdfs:
        if pdf_content:  # Ensure content is not None
            content_stream = io.BytesIO(pdf_content)  # Treat PDF data as a PDF file stream
            merger.append(content_stream)

    # Write the output PDF with appended content
    with open(output_pdf, "wb") as f_out:
        merger.write(f_out)
        merger.close()

# Function to filter meaningful pages and write them to a BytesIO stream
def filter_pdf_pages_to_stream(input_pdf_path, min_word_count=50, keyword_list=None):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    doc = fitz.open(input_pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Check if the page is meaningful
        if is_meaningful_page(text, min_word_count=min_word_count, keyword_list=keyword_list):
            writer.add_page(reader.pages[page_num])  # Add meaningful page to writer

    # Save the filtered PDF to a BytesIO stream
    filtered_pdf_stream = io.BytesIO()
    writer.write(filtered_pdf_stream)
    filtered_pdf_stream.seek(0)  # Reset stream position to the beginning
    return filtered_pdf_stream

# Function to determine if a page is meaningful based on text length and keyword presence
def is_meaningful_page(text, min_word_count=50, keyword_list=None):
    cleaned_text = re.sub(r'[^\x20-\x7E\n\r]', '', text).strip()
    word_count = len(cleaned_text.split())

    # Check for minimum word count
    if word_count >= min_word_count:
        return True

    # Check for presence of important keywords
    if keyword_list:
        for keyword in keyword_list:
            if keyword.lower() in cleaned_text.lower():
                return True

    return False

# Example Execution
original_pdf = 'GEM_files/GeM-Bidding-6429565.pdf'  # Path to your original PDF
output_pdf = 'GEM_files/GEM_inside.pdf'  # Path for the filtered PDF

# Extract links from the original PDF
links = extract_links_from_pdf(original_pdf)

# Process the links to extract the required PDF data
extracted_pdfs = process_pdf_links(links)

# Append the extracted PDF data to the original PDF with automatic filtering
append_data_to_pdf(original_pdf, extracted_pdfs, output_pdf)

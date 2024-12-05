import os
import asyncio
from llama_index.core import Document, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from qdrant_client import QdrantClient, models
from llama_parse import LlamaParse
import re
import fitz  # PyMuPDF to extract text from the PDF
import requests
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import io

# Initialize embedding model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Set environment variables
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-cL4g9SKHoDyJ3o4IBd72aB7MtYnXybSE6f4dMoYSk2LXaSdv"

# Set the global embedding model in Settings
Settings.embed = embed_model  # This sets the global embedding model

# Function to remove Hindi characters
def remove_hindi_characters(text):
    hindi_pattern = re.compile(r'[\u0900-\u097F]+')
    return hindi_pattern.sub('', text)

# Synchronously parse PDF with LlamaParse
def parse_pdf_with_llamaparse(file_path):
    document_with_instruction = LlamaParse(
        result_type="markdown",
        parsing_instruction="""Your parsing instructions here"""
    )
    return asyncio.run(document_with_instruction.aload_data(file_path))

# Function to extract links from the PDF
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

# Function to process each PDF link
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

# Function to append extracted data to the original PDF
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

# Process document content
def process_document(document_content):
    if isinstance(document_content, list):
        document_content = "\n".join([doc.text for doc in document_content])
    cleaned_content = remove_hindi_characters(document_content)
    return cleaned_content

# Store chunks in Qdrant
def store_chunks_in_qdrant(nodes, collection_name="user_documents"):
    print("Connecting to Qdrant...")
    client = QdrantClient(
        url="https://fd4ba2d3-e45f-44d6-9b20-b0c5d1cabb9d.us-east4-0.gcp.cloud.qdrant.io",
        api_key="your_qdrant_api_key_here",  # Use your own Qdrant API key
    )
    print("Connected to Qdrant.")

    # Check if collection exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists.")
    except Exception:
        print(f"Collection '{collection_name}' does not exist. Creating it.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

    # Prepare points to upsert
    points = []
    for doc_id, node in enumerate(nodes):
        embedding = node.embedding
        metadata = node.metadata

        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            print(f"Invalid embedding for node {doc_id}. Skipping.")
            continue

        point = models.PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"text": node.get_content(), "metadata": metadata}
        )
        points.append(point)

    # Upsert points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Stored {len(points)} chunks in Qdrant.")
    else:
        print("No valid points to upsert.")

# Process uploaded PDF file
def process_uploaded_file(file_path):
    print("Processing uploaded file...")

    # Step 1: Extract links from the original PDF
    links = extract_links_from_pdf(file_path)
    print(f"Extracted {len(links)} links from the PDF.")

    # Step 2: Process each PDF link
    extracted_pdfs = process_pdf_links(links)
    print(f"Downloaded {len(extracted_pdfs)} linked PDFs.")

    # Step 3: Append extracted data to the original PDF
    merged_pdf_path = file_path + "_merged.pdf"
    append_data_to_pdf(file_path, extracted_pdfs, merged_pdf_path)
    print(f"Merged PDF saved to {merged_pdf_path}")
    
    # Now, use the merged PDF for further processing
    # Parse the merged PDF
    parsed_content = parse_pdf_with_llamaparse(merged_pdf_path)
    print("Parsing completed.")

    # Process content
    processed_content = process_document(parsed_content)
    print("Content processed.")

    # Create document and parse nodes
    document = Document(text=processed_content, metadata={'filename': os.path.basename(merged_pdf_path)})
    parser = SimpleNodeParser()
    parsed_nodes = parser.get_nodes_from_documents([document])
    print(f"Parsed {len(parsed_nodes)} nodes.")

    # Generate embeddings
    for node in parsed_nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    print("Embeddings generated.")

    # Store in Qdrant
    store_chunks_in_qdrant(parsed_nodes)
    print("File processing completed.")
    return len(parsed_nodes)


# This function allows the user to input a PDF file from the local system
def main():
    file_path = input("Enter the path to the PDF file you want to process: ")
    if os.path.exists(file_path) and file_path.lower().endswith(".pdf"):
        num_nodes = process_uploaded_file(file_path)
        print(f"Processed {num_nodes} nodes.")
    else:
        print("Invalid file. Please provide a valid PDF file.")

if __name__ == "__main__":
    main()

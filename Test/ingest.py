import os
from engine.rag_retriever import add_ncert_pdf, rebuild_index

def start_ingestion():
    # 1. Define the path to your file in the data folder
    pdf_file = "data/ncert_chunks/math10.pdf" 
    
    if not os.path.exists(pdf_file):
        print(f"❌ File not found at {pdf_file}. Did you move it to the data folder?")
        return

    # 2. Run the ingestion (Extracts text and saves to data/ncert_chunks/)
    print("🚀 Starting PDF Ingestion...")
    add_ncert_pdf(
        pdf_path=pdf_file,
        subject="Mathematics",
        grade=9
    )

    # 3. Rebuild the FAISS index so the AI can actually 'find' the new data
    print("\n🔄 Updating the search index...")
    rebuild_index()
    
    print("\n✨ Ingestion complete! SYRA now knows this textbook.")

if __name__ == "__main__":
    start_ingestion()
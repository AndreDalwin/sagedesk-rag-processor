import pymupdf
import os
import io
import statistics # For calculating average char width/line height if needed
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    print("Error: pymupdf4llm package not found. Please install it: pip install pymupdf4llm")
    PYMUPDF4LLM_AVAILABLE = False
    # Exit if the required library is missing
    import sys
    sys.exit(1)

# --- Configuration ---
INPUT_PDF = "PHY 6.10 - Vision Physiology v2.pdf"
# OCR_LANGUAGE is no longer needed here as pymupdf4llm handles it internally if required
# OCR_ZOOM_MATRIX is no longer needed
# INVALID_UNICODE check is handled by pymupdf4llm

# --- Helper Functions ---
def get_output_md_filename(pdf_path):
    """Generates the output MD filename based on the input PDF."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return f"{base_name}_extracted_content.md" # Changed extension to .md

# --- Main Processing ---
output_filename = get_output_md_filename(INPUT_PDF)
# No longer need to open the doc explicitly here, pymupdf4llm handles it.
doc = None

if not PYMUPDF4LLM_AVAILABLE:
    # This check is slightly redundant due to sys.exit above, but safe
    print("Exiting due to missing pymupdf4llm library.")
else:
    print(f"Processing '{INPUT_PDF}' using pymupdf4llm...")
    try:
        # Call pymupdf4llm to convert the whole PDF to Markdown
        md_text = pymupdf4llm.to_markdown(INPUT_PDF)

        # Write the resulting Markdown text to the output file
        print(f"Writing Markdown output to '{output_filename}'...")
        with open(output_filename, "w", encoding="utf-8") as md_out:
            md_out.write(md_text)

        print(f"\nProcessing complete. Markdown content saved to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: Input PDF file not found at '{INPUT_PDF}'")
    except pymupdf.errors.PasswordError:
         print(f"Error: PDF file '{INPUT_PDF}' requires a password.")
    # Catch potential exceptions during pymupdf4llm processing
    except Exception as e:
        print(f"An unexpected error occurred during Markdown conversion: {e}")

# The finally block for doc.close() is removed as we are not opening the doc directly.
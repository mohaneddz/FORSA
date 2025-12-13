from pdf2image import convert_from_path
import requests
from langdetect import detect
from PIL import Image
import os
import fitz  # PyMuPDF
from services.llm_service import LLMService
import logging

API_KEY = "K85435650588957"

logger = logging.getLogger(__name__)

def ocr_space_auto_image(image, api_key=API_KEY):
    """Send a PIL image to OCR.space and return extracted text"""
    import io
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    url = "https://api.ocr.space/parse/image"
    try:
        response = requests.post(
            url,
            files={"filename": buf},
            data={"apikey": api_key, "language": "eng"},
            timeout=30
        )
        result = response.json()
    except requests.exceptions.Timeout:
        print(f"Timeout error during OCR")
        return ""
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

    if result.get("IsErroredOnProcessing"):
        print(f"OCR Error: {result.get('ErrorMessage')}")
        return ""

    text = ""
    for parsed in result.get("ParsedResults", []):
        text += parsed.get("ParsedText", "")

    return text

def extract_images_from_pdf(pdf_path, output_folder="images"):
    """Extract images from PDF and save to output folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"{base_name}_page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            images.append(image_path)
            print(f"Extracted: {image_filename}")
    
    doc.close()
    return images

def correct_ocr_with_deepseek(raw_text, base_name, file_path=""):
    """Use DeepSeek to correct OCR errors, remove useless data, and structure the document"""
    
    llm_service = LLMService()
    
    system_prompt = """You are an expert in correcting OCR errors for Algérie Télécom documents.

Context: This document is from Algérie Télécom and may contain information about:
- Internet offers (ADSL, Fibre, 4G LTE, FTTH)
- Mobile plans and tariffs (Idoom, Mobilis)
- ONT (Optical Network Terminal) equipment specifications
- Residential and business contracts (conventions)
- Pricing structures and promotional offers
- Technical specifications (Mbps, Gbps, DA)
- Customer service procedures
- NGBSS system procedures and guides

Task: Correct all OCR errors and improve structure:
1. Fix spelling mistakes, especially French and Arabic technical terms
2. Correct misread numbers, prices (DA), and speeds (Mbps/Gbps)
3. Properly format sections with Markdown headers (##, ###)
4. Ensure proper spacing and punctuation
5. Maintain the original meaning and content
6. Use proper Markdown tables for tariffs and specifications
7. Format procedures as numbered lists (1. 2. 3.)
8. Preserve both French and Arabic content
9. Remove USELESS data:
   - Table headers/cells that are navigation elements (like "نقطة حضور", "جندل", "جليدة")
   - Repetitive footer/header text (like "La Division Marketing ,Communication et Commerciale  - Mars 2019  -")
   - Page numbers and division names when they clutter the content
   - Decorative table borders that don't contain data
10. Focus on the MAIN CONTENT:
    - Keep procedure steps clear and sequential
    - Keep tariff tables with actual pricing data
    - Keep Q/A pairs complete
    - Keep technical specifications

Output: Clean, well-structured Markdown with <---CHUNK---> delimiters between logical sections.
DO NOT output JSON. DO NOT use code fences. Output MARKDOWN ONLY."""

    user_prompt = f"""Document: {base_name}
File Path: {file_path}

Raw OCR Text:
{raw_text}

Provide corrected, well-structured Markdown with <---CHUNK---> delimiters separating logical sections.
Remove useless decorative elements and focus on the main content."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        logger.info(f"Calling DeepSeek to correct OCR for {base_name}...")
        print(f"Calling DeepSeek API to correct OCR for {base_name}...")
        
        corrected_text = llm_service.chat_completion(
            messages=messages,
            temperature=0.1,
            json_mode=False,  # We want Markdown, not JSON
            max_retries=2
        )
        
        # Remove markdown code fences if present
        corrected_text = corrected_text.strip()
        if corrected_text.startswith("```markdown"):
            corrected_text = corrected_text[11:]
        if corrected_text.startswith("```"):
            corrected_text = corrected_text[3:]
        if corrected_text.endswith("```"):
            corrected_text = corrected_text[:-3]
        
        corrected_text = corrected_text.strip()
        
        logger.info(f"✓ DeepSeek correction complete ({len(corrected_text)} chars)")
        print(f"✓ DeepSeek correction complete ({len(corrected_text)} chars)")
        return corrected_text
        
    except Exception as e:
        logger.error(f"⚠ DeepSeek API error: {e} - using raw text")
        print(f"⚠ DeepSeek API error: {e} - using raw text")
        return raw_text

def process_file_to_md(file_path):
    """Process PDF or image file, extract images, OCR them, and combine into markdown"""
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    file_ext = os.path.splitext(file_path)[1].lower()
    
    all_text = ""
    
    if file_ext == '.pdf':
        # Extract images from PDF
        print(f"Extracting images from {file_path}...")
        image_paths = extract_images_from_pdf(file_path, output_folder)
        
        # If no embedded images, convert pages to images
        if not image_paths:
            print("No embedded images found. Converting pages to images...")
            images = convert_from_path(file_path)
            for i, image in enumerate(images, start=1):
                image_filename = f"{base_name}_page_{i}.png"
                image_path = os.path.join(output_folder, image_filename)
                image.save(image_path)
                image_paths.append(image_path)
                print(f"Saved page as: {image_filename}")
        
        # OCR all images
        for i, img_path in enumerate(image_paths, start=1):
            print(f"Processing image {i}/{len(image_paths)}...")
            image = Image.open(img_path)
            text = ocr_space_auto_image(image)
            if text.strip():
                all_text += f"\n--- Page/Image {i} ---\n\n{text}\n"
    
    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        # Process single image
        print(f"Processing image {file_path}...")
        image_filename = f"{base_name}{file_ext}"
        image_path = os.path.join(output_folder, image_filename)
        
        # Copy to images folder
        image = Image.open(file_path)
        image.save(image_path)
        print(f"Copied to: {image_filename}")
        
        # OCR the image
        text = ocr_space_auto_image(image)
        if text.strip():
            all_text = text
    
    # Correct OCR errors and structure with DeepSeek
    if all_text.strip():
        corrected_text = correct_ocr_with_deepseek(all_text, base_name, file_path)
    else:
        print("⚠ No text extracted from OCR")
        corrected_text = all_text
    
    # Save combined markdown
    md_filename = f"{base_name}.md"
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(corrected_text)
    print(f"Saved corrected markdown: {md_filename}")
    
    # Chunk and save to gen folder using DeepSeek delimiters
    chunk_to_gen_folder_with_delimiters(corrected_text, base_name, file_path)
    
    return md_filename

def chunk_to_gen_folder_with_delimiters(text, base_name, file_path):
    """Split text by DeepSeek <---CHUNK---> delimiters and save to gen folder"""
    gen_folder = "gen"
    os.makedirs(gen_folder, exist_ok=True)
    
    # Split by chunk delimiter
    delimiter = "<---CHUNK--->"
    chunks = [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]
    
    # If no delimiters found, use basic chunking
    if len(chunks) <= 1:
        print("⚠ No chunk delimiters found, using basic chunking")
        chunk_to_gen_folder(text, base_name, chunk_size=2000)
        return
    
    # Save chunks with SOURCE_FILE header
    for i, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{base_name}__Chunk{i:03d}.md"
        chunk_path = os.path.join(gen_folder, chunk_filename)
        
        # Add SOURCE_FILE header (matching preprocess.py format)
        chunk_content = f"# SOURCE_FILE: {file_path}\n\n{chunk}"
        
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk_content)
        print(f"Saved chunk: {chunk_filename}")
    
    print(f"Created {len(chunks)} chunks in {gen_folder}/")

def chunk_to_gen_folder(text, base_name, chunk_size=2000):
    """Chunk text and save to gen folder (fallback basic chunking)"""
    gen_folder = "gen"
    os.makedirs(gen_folder, exist_ok=True)
    
    # Split text into chunks
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Save chunks
    for i, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{base_name}_chunk_{i}.md"
        chunk_path = os.path.join(gen_folder, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"Saved chunk: {chunk_filename}")
    
    print(f"Created {len(chunks)} chunks in {gen_folder}/")

# Usage
if __name__ == "__main__":
    # Example usage
    # process_file_to_md("your_file.pdf")
    # process_file_to_md("your_image.png")
    pass
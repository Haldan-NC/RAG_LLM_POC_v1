import fitz
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from shapely.geometry import box
from shapely.ops import unary_union


def render_pdf_to_images(pdf_path: str, zoom: float = 2.0) -> list:
    """
    Renders a PDF file to images, one for each page.
    Args:
        pdf_path (str): Path to the PDF file.
        zoom (float): Zoom factor for rendering.
    Returns:
        list: List of dictionaries containing page number and image data.
    """

    cwd = os.getcwd()
    pdf_path = os.path.join(cwd, pdf_path)

    doc = fitz.open(pdf_path)
    images = []
    for i,page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom))
        img = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
        images.append({'page_number':i+1,'image':img})
    return images



def detect_image_regions(page_image: Image.Image, buffer:int=0, min_size:int=70, threshold:int=240) -> list:
    """
    Finds bounding-boxes of image-like regions. Used before cropping.
    """
    gray = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2GRAY)
    _,th = cv2.threshold(gray, threshold,255,cv2.THRESH_BINARY_INV)
    conts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions=[]
    for c in conts:
        x,y,w,h = cv2.boundingRect(c)
        if w>min_size and h>min_size:
            regions.append([x-buffer,y-buffer,x+w+buffer,y+h+buffer])
    return regions


def merge_overlapping_regions(regions: list, buffer: int =0) -> list:
    """
    Merges intersecting boxes via Shapely. Improves crop consistency.
    """
    boxes = [box(x1-buffer,y1-buffer,x2+buffer,y2+buffer) for x1,y1,x2,y2 in regions]
    merged = unary_union(boxes)
    merged = [merged] if merged.geom_type=='Polygon' else list(merged.geoms)
    return [[int(b.bounds[0]),int(b.bounds[1]),int(b.bounds[2]),int(b.bounds[3])] for b in merged]


def crop_regions_from_image(page_image: Image.Image, regions: list, output_dir: str, page_num:int, manual_id:int, metadata:dict) -> dict:
    """
    Crops & saves each region; updates `metadata`. Used in extract_images_from_pdf.
    """
    os.makedirs(output_dir,exist_ok=True)
    for i,coords in enumerate(regions):
        x1,y1,x2,y2 = map(int,coords)
        cropped = page_image.crop((x1,y1,x2,y2))
        path = os.path.join(output_dir, f"doc_{manual_id}_page_{page_num}_img_{i}.png")
        cropped.save(path)
        metadata.setdefault(page_num, {})[i] = {'page':page_num,'image_path':path,'coords':coords}
    return metadata


def extract_images_from_pdf(pdf_path: str, manual_id: int, output_dir: str, verbose: int=0) -> dict:
    """
    High-level orchestration: render -> detect -> merge -> crop.
    Returns metadata mapping pages->region metadata. Used in database setup.
    """
    pages = render_pdf_to_images(pdf_path)
    meta = {}
    for pg in pages:
        regs = detect_image_regions(pg['image'], buffer=2)
        merged = merge_overlapping_regions(regs)
        meta = crop_regions_from_image(pg['image'], merged,
                                       os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf','')),
                                       pg['page_number'], manual_id, meta)
    return meta


def extract_page_number_from_filename(filename: str) -> str:
    return filename.split("_")[3] if "_" in filename else None


def generate_image_table(documents_df: pd.DataFrame, sections_df: pd.DataFrame, image_dir: str, all_manuals_metadata: dict) -> pd.DataFrame:
    image_records = []

    # Loop over all subdirectories in image_dir
    for subfolder in os.listdir(image_dir):
        subfolder_path = os.path.join(image_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue  # skip files
        
        # Match to document by DOCUMENT_NAME (strip extension if needed)
        matching_docs = documents_df[documents_df['DOCUMENT_NAME'].str.contains(subfolder, case=False)]
        if matching_docs.empty:
            print(f"No matching document for subfolder: {subfolder}")
            continue
        
        document_id = matching_docs.iloc[0]['DOCUMENT_ID']
        document_name = matching_docs.iloc[0]['DOCUMENT_NAME']
        
        # List all image files in subdirectory
        for image_file in os.listdir(subfolder_path):
            if not image_file.lower().endswith((".png")):
                continue
            
            image_path = os.path.join(subfolder_path, image_file)
            page_number = extract_page_number_from_filename(image_file)
            order_number = image_file.split("img_")[-1].strip(".png")

            image_size = os.path.getsize(image_path)
            image_width, image_height = Image.open(image_path).size
            
            # Try to match to a section (same document, closest PAGE <= image page)
            section_match = None
            if page_number is not None:
                matching_sections = sections_df[
                    (sections_df['DOCUMENT_ID'] == document_id) & 
                    (sections_df['PAGE'].astype(str) <= str(page_number))
                ]
                if not matching_sections.empty:
                    section_match = matching_sections.sort_values("PAGE", ascending=False).iloc[0]
            
            image_records.append({
                "DOCUMENT_ID": document_id,
                "SECTION_ID": section_match["SECTION_ID"] if section_match is not None else None,
                "SECTION_NUMBER": section_match["SECTION_NUMBER"] if section_match is not None else None,
                "PAGE": page_number,
                "IMG_ORDER": order_number,
                "IMAGE_FILE": image_file,
                "IMAGE_PATH": image_path,
                "IMAGE_SIZE": image_size,
                "IMAGE_WIDTH": image_width,
                "IMAGE_HEIGHT": image_height,
                "IMAGE_X1": all_manuals_metadata[document_id][int(page_number)][int(order_number)]["coords"][0],
                "IMAGE_Y1": all_manuals_metadata[document_id][int(page_number)][int(order_number)]["coords"][1],
                "IMAGE_X2": all_manuals_metadata[document_id][int(page_number)][int(order_number)]["coords"][2],
                "IMAGE_Y2": all_manuals_metadata[document_id][int(page_number)][int(order_number)]["coords"][3],
                "DESCRIPTION": ""  # Placeholder for image description
            })

    image_df = pd.DataFrame(image_records)
    image_df.dropna(inplace=True)
    image_df.reset_index(drop=True, inplace=True)
    return image_df


if __name__ == "__main__":
    print("")
    # Test whether this script runs
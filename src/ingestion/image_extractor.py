import fitz
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL import UnidentifiedImageError
import io
import cv2
from shapely.geometry import box
from shapely.ops import unary_union
import sys
import pytesseract
import pdfplumber
from tqdm import tqdm

sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 
from src.utils.utils import convert_to_abs_path, SuppressStderr, log



def render_pdf_to_images(pdf_path: str, zoom: float = 2.0) -> list:
    """
    Extracts all images from a given PDF page along with their positions and surrounding text,
    saves the images to disk, and returns structured metadata for each image.

    Args:
        pdf_path (str): Path to the PDF file.
        zoom (float): Zoom factor for rendering.
    Returns:
        list: List of dictionaries containing page number and image data.
    """

    pdf_path = convert_to_abs_path(pdf_path)

    doc = fitz.open(pdf_path)
    images = []
    for i,page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom))
        img = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
        images.append({'page_number':i+1,'image':img})
    return images


def render_pdf_page_to_image(file_path: str, real_page_num: int, zoom: float = 2.0) -> Image.Image:
    """
    Renders a specific page of a PDF file to an image.
    Args:
        pdf_path (str): Path to the PDF file.
        real_page_num (int): Page number to render.
        zoom (float): Zoom factor for rendering.
    Returns:
        Image.Image: Rendered image of the specified page.
    """
    doc = fitz.open(file_path)
    page = doc.load_page(real_page_num-1)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom,zoom))
    img = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
    return img


def crop_and_rotate_bottom_right(image: Image.Image) -> Image.Image:
    """
    Crop bottom-right 5% x 50% of the image and rotate it 90° clockwise.
    This is used in the document metadata extraction process. 
    Could be improved to be more genereal.
    """
    w, h = image.size
    crop_box = (int(w * 0.95), int(h * 0.5), w, h)
    return image.crop(crop_box).transpose(Image.ROTATE_270)


def ocr_on_side_label_vestas_page(file_path: str) -> str:
    """
    Performs OCR on the side label of a Vestas PDF page.
    """
    image = render_pdf_page_to_image(file_path = file_path, real_page_num = 1)
    cropped = crop_and_rotate_bottom_right(image)
    text = pytesseract.image_to_string(cropped, lang="eng")

    return text


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



def extract_images_from_page(page: fitz.Page, page_num: int, image_path: str) -> list:
    """
    Extracts all images from a given PDF page and saves them to disk.

    Args:
        page (fitz.Page): The page object from which to extract images.
        page_num (int): The page number (used in image file naming).
        image_path (str): The base path where extracted images should be saved.

    Returns:
        list: A list of file paths to the extracted and saved images.
    """
    images = page.images
    images_path_list = []
    
    image_path = image_path.replace(".pdf", "").replace("Documents\\VGA_guides", "Images")    
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)

    for image_index, image in enumerate(images):
        images_path_list.append(extract_image_from_imageobj(image, image_path, image_index, page_num))
    return images_path_list


# -------------------- FUNCTION FOR SAVING IMAGE -------------------- 
def extract_image_from_imageobj(image_obj, image_path, image_index, page_num) -> str:
    """
    Extracts image data from a PDF image object, saves it as a PNG file, and returns the filename.

    Args:
        image_obj (dict): The image object extracted from a PDF page (typically from `page.images`),
                          containing the raw image stream and metadata.
        image_path (str): The destination folder path where the image file should be saved.
        image_index (int): The index of the image on the page (used in naming the file).
        page_num (int): The page number the image was extracted from (used in naming the file).

    Returns:
        str: The name of the saved image file. Returns "NaN" if saving the image fails.
    """
    stream = image_obj["stream"]
    image_bytes = stream.get_data()  # raw image bytes
    image_name = f"page_{page_num}_image_{image_index}.png"
    image_dest = os.path.join(image_path, image_name)
    
    try:
        # Trying to save image converted with BytesIO
        image = Image.open(io.BytesIO(image_bytes))
        image.save(image_dest)
    except Exception as e:
        # Trying to save image converted with frombytes method
        try:
            Height = stream["Height"]
            Width = stream["Width"]
            imag = Image.frombytes("RGB", (Width, Height), image_bytes)
            imag.save(image_dest)
        except Exception as e:
            log(f"Cannot open or save image \n {image_obj}, \nerror: {e}\n", level=2)
            image_dest = "NaN"
    return image_name



# -------------------- HELPING FUNCTION FOR TEXT LOCATION -------------------- 
def is_above(word_bbox, image_bbox, vertical_margin=30):
    """
    Checks if a word is located above the image within a vertical margin.

    Args:
        word_bbox (dict): Bounding box of the word with 'top' and 'bottom'.
        image_bbox (dict): Bounding box of the image with 'top'.
        vertical_margin (int): Max allowed vertical distance to consider the word 'above'.

    Returns:
        bool: True if word is above the image and within the margin.
    """
    return (
        word_bbox['bottom'] <= image_bbox['top'] and
        abs(word_bbox['bottom'] - image_bbox['top']) <= vertical_margin
    )
    
def is_below(word_bbox, image_bbox, vertical_margin=30):
    """
    Checks if a word is located below the image within a vertical margin.

    Args:
        word_bbox (dict): Bounding box of the word with 'top' and 'bottom'.
        image_bbox (dict): Bounding box of the image with 'bottom'.
        vertical_margin (int): Max allowed vertical distance to consider the word 'below'.

    Returns:
        bool: True if word is below the image and within the margin.
    """
    return (
        word_bbox['top'] >= image_bbox['bottom'] and
        abs(word_bbox['top'] - image_bbox['bottom']) <= vertical_margin
    )

def is_left(word_bbox, image_bbox):
    """
    Checks if a word is to the left of the image and vertically overlaps.

    Args:
        word_bbox (dict): Bounding box of the word with 'x1', 'top', 'bottom'.
        image_bbox (dict): Bounding box of the image with 'x0', 'top', 'bottom'.

    Returns:
        bool: True if the word is left of the image and vertically aligned.
    """
    vertical_overlap = not (
        word_bbox['top'] > image_bbox['bottom'] or word_bbox['bottom'] < image_bbox['top']
    )
    return (
        word_bbox['x1'] <= image_bbox['x0'] and
        vertical_overlap
    )

def is_right(word_bbox, image_bbox):
    """
    Checks if a word is to the right of the image and vertically overlaps.

    Args:
        word_bbox (dict): Bounding box of the word with 'x0', 'top', 'bottom'.
        image_bbox (dict): Bounding box of the image with 'x1', 'top', 'bottom'.

    Returns:
        bool: True if the word is right of the image and vertically aligned.
    """
    vertical_overlap = not (
        word_bbox['top'] > image_bbox['bottom'] or word_bbox['bottom'] < image_bbox['top']
    )
    return (
        word_bbox['x0'] >= image_bbox['x1'] and
        vertical_overlap
    )


# -------------------- HELPING FUNCTION FOR WORD MAPPING -------------------- 
def word_list_to_sorted_sentence(related_words: list[tuple]) -> str:
    """Takes in list of tuples where each instance in the list is a tuple on the format (x0,y0, word). 
        The words in the tuples are joined toger based on x0 and y0 location to each other.

    Args:
        related_words (list): list of words to be joined together into a sentence sorted based on placement

    Returns:
        str: joied string of all words in list.  
    """
    related_words.sort()
    sentence = ' '.join([w[2] for w in related_words])
    return sentence


def extract_image_data_from_page(page: fitz.Page, page_num: int, document_name: str, document_id: int, image_folder: str) -> list[dict]:
    """
    Extracts all images from a given PDF page along with their positions and surrounding text,
    saves the images to disk, and returns structured metadata for each image.

    Args:
        page (fitz.Page): The PDF page object to extract images from.
        page_num (int): The current page number in the document.
        document_name (str): The name of the PDF document (used to name folders).
        document_id (int): Unique identifier for the document (used in metadata).
        image_folder (str): Base directory path where images should be saved.

    Returns:
        list[dict]: A list of dictionaries, each containing metadata for one image, including:
            - file path, file name, position (x0, y0, x1, y1),
            - image size (width, height),
            - page number,
            - text located above, below, left, and right of the image.
    """
    
    # Get images and words on page
    words = page.extract_words()  # returns list of word dicts with positions
    images = page.images
    destination_path = os.path.join(image_folder, document_name.replace(".pdf", ""))
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
    
    image_records = []
    for img_index, image in enumerate(images):
        image_name = extract_image_from_imageobj(image, destination_path, img_index, page_num) # Saving image
        image_bbox = {
            'x0': image['x0'],
            'x1': image['x1'],
            'top': image['top'],
            'bottom': image['bottom']
        }
        related_words_above = []
        related_words_below = []
        related_words_left = []
        related_words_right = []
        
        for word in words:
            word_bbox = {
                'x0': float(word['x0']),
                'x1': float(word['x1']),
                'top': float(word['top']),
                'bottom': float(word['bottom'])
            }

            if is_left(word_bbox, image_bbox):
                related_words_left.append((word['top'], word['x0'], word['text']))
            elif is_above(word_bbox, image_bbox):
                related_words_above.append((word['top'], word['x0'], word['text']))  # retain 'top' and 'left' for sorting
            elif is_right(word_bbox, image_bbox):
                related_words_right.append((word['top'], word['x0'], word['text']))
            elif is_below(word_bbox, image_bbox):
                related_words_below.append((word['top'], word['x0'], word['text']))

            sentence_above = word_list_to_sorted_sentence(related_words_above) 
            sentence_below = word_list_to_sorted_sentence(related_words_below) 
            sentence_left = word_list_to_sorted_sentence(related_words_left) 
            sentence_right = word_list_to_sorted_sentence(related_words_right) 
        image_records.append({
            "IMAGE_PATH": destination_path, # Locally installed file destination
            "DOCUMENT_ID": document_id, # Document ID as defined in snowflake
            "IMAGE_FILE": image_name, # Image file name
            "PAGE": page_num, # Page number, image was found on
            "IMG_ORDER": img_index, # Image index as extracted on page
            "IMAGE_WIDTH": image['width'], # Image width
            "IMAGE_HEIGHT": image['height'], # Image number
            "IMAGE_X0": image['x0'], # Distance of left side of the image from left side of page.
            "IMAGE_Y0": image['y0'], # Distance of right side of the image from left side of page.
            "IMAGE_X1": image['x1'], # Distance of bottom of the image from bottom of page.
            "IMAGE_Y1": image['y1'], # Distance of top of the image from bottom of page.
            "TEXT_ABOVE": sentence_above, # Text above image
            "TEXT_BELOW": sentence_below, # Text under image
            "TEXT_LEFT": sentence_left, # Text to the left of image
            "TEXT_RIGHT": sentence_right # Text to the right of image
        })
    return image_records





def delete_all_local_images() -> None:
    """
    Deletes all images in the specified directory.
    Args:
        image_dir (str): Directory containing images to delete.
    """
    image_dir = "data\\Vestas_RTP\\Images"
    for root, dirs, files in os.walk(image_dir):
        for sub_dirs in dirs:
            # remove subdirectories
            sub_dir_path = os.path.join(root, sub_dirs)
            if os.path.exists(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    file_path = os.path.join(sub_dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(sub_dir_path)

    log(f"Deleted all images in {image_dir}", level=1)


def delete_all_local_images() -> None:
    """
    Deletes all images in the specified directory.
    Args:
        image_dir (str): Directory containing images to delete.
    """
    image_dir = "data\\Vestas_RTP\\Images"
    for root, dirs, files in os.walk(image_dir):
        for sub_dirs in dirs:
            # remove subdirectories
            sub_dir_path = os.path.join(root, sub_dirs)
            if os.path.exists(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    file_path = os.path.join(sub_dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(sub_dir_path)

    log(f"Deleted all images in {image_dir}", level=1)


if __name__ == "__main__":
    pass   
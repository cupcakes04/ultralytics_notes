import xml.etree.ElementTree as ET
import os
from pathlib import Path
import yaml

def xml_to_yolo(xml_path, class_names):
    """
    Convert Pascal VOC XML annotation to YOLO format.
    
    Args:
        xml_path: Path to the XML file
        class_names: List of class names in order (e.g., ['pothole', 'crack'])
        
    Returns:
        str: YOLO format string with one line per object
             Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Process each object
    yolo_lines = []
    for obj in root.findall('object'):
        # Get class name
        class_name = obj.find('name').text
        
        # Get class ID (index in class_names list)
        if class_name not in class_names:
            print(f"Warning: Class '{class_name}' not in class_names list. Skipping.")
            continue
        class_id = class_names.index(class_name)
        
        # Get bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Create YOLO format line
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return '\n'.join(yolo_lines)


def convert_xml_folder(xml_folder, output_folder, class_names):
    """
    Convert all XML files in a folder to YOLO format txt files.
    
    Args:
        xml_folder: Folder containing XML files
        output_folder: Folder to save YOLO txt files
        class_names: List of class names in order
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each XML file
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        
        # Convert to YOLO format
        yolo_str = xml_to_yolo(xml_path, class_names)
        
        # Save to txt file (same name as XML but with .txt extension)
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write(yolo_str)
        
        print(f"Converted: {xml_file} -> {txt_filename}")
    
    print(f"\nTotal files converted: {len(xml_files)}")

from ultralytics.data.split import autosplit

if __name__ == "__main__":
    print("XML to YOLO Converter")

    # Get the root path for dataset
    scrip_dir = Path(__file__).resolve().parent
    with open(scrip_dir / 'road.yaml', 'r') as file:
        data = yaml.safe_load(file)
    root_path = Path(data['path'])

    # Automatically split the dataset
    autosplit(
        path= root_path/ 'images', # Directory containing all your image files
        weights=(0.8, 0.1, 0.1),     # The desired split ratio (Train, Val, Test)
        annotated_only=True          # Only use images that have a corresponding label file
    )
    class_names = ['pothole']  # Define your classes here
    
    # Convert entire folder
    xml_folder = root_path / "annotations"
    output_folder = root_path / "labels"
    convert_xml_folder(xml_folder, output_folder, class_names)

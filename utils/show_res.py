import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from pathlib import Path

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def review_results(result, print_res=True, draw_boxes=True, labels=None, axis=False, mask_alpha=0.5, linewidth=2):
    """self practice to parse result data

    Args:
        result (_type_): _description_
        print_res (bool, optional): _description_. Defaults to True.
        draw_boxes (bool, optional): _description_. Defaults to True.
        labels (_type_, optional): _description_. Defaults to None.
        axis (bool, optional): _description_. Defaults to False.
        mask_alpha (float, optional): _description_. Defaults to 0.5.
    """

    # Check if masks are available (segmentation task)
    has_masks = result.masks is not None
    
    # Extract data from results
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy()     # Class IDs
    class_names = result.names                      # Class name mapping
    
    if has_masks:
        masks_xyn = result.masks.xyn  # Normalized polygon coordinates
        mask_orig_shape = result.masks.orig_shape  # (H, W)
    
    # Print extracted data
    if print_res == True:
        print(f"Number of detections: {len(boxes_xyxy)}")
        print(f"Task type: {'Segmentation' if has_masks else 'Detection'}")
        print("\nDetailed detection info:")
        for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
            x1, y1, x2, y2 = box
            class_name = class_names[int(cls_id)]
            print(f"Detection {i+1}:")
            print(f"  Class: {class_name} (ID: {int(cls_id)})")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            if has_masks:
                print(f"  Polygon points: {len(masks_xyn[i])}")
            print()

    # Visualize with matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 8))
    img_rgb = bgr2rgb(result.orig_img)
    ax.imshow(img_rgb)
    
    img_height, img_width = result.orig_shape
    
    # Draw ground truth labels if provided
    if labels is not None:
        # labels can be a file path (str) or list of YOLO format lines
        if isinstance(labels, str):
            with open(labels, 'r') as f:
                label_lines = f.readlines()
        else:
            label_lines = labels
        
        for label_line in label_lines:
            parts = label_line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to xyxy format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Draw ground truth box in green
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add GT label
            gt_label = f"GT: {class_names.get(cls_id, f'Class {cls_id}')}"
            ax.text(
                x1, y1 - 20,
                gt_label,
                color='white',
                fontsize=9,
                bbox=dict(facecolor='green', alpha=0.7, edgecolor='none', pad=2)
            )

    # Draw segmentation masks if available
    if has_masks:
        # Create a color map for different instances
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks_xyn)))
        
        # Get original image dimensions from mask
        mask_height, mask_width = mask_orig_shape
        
        for i, (polygon_norm, cls_id) in enumerate(zip(masks_xyn, class_ids)):
            if len(polygon_norm) == 0:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            polygon = polygon_norm.copy()
            polygon[:, 0] *= mask_width   # x coordinates
            polygon[:, 1] *= mask_height  # y coordinates
            
            # Create colored mask overlay with RGBA using polygon
            color = colors[i][:3]  # RGB values
            
            # Create a binary mask from polygon
            mask_img = np.zeros((mask_height, mask_width), dtype=np.uint8)
            polygon_int = polygon.astype(np.int32)
            cv2.fillPoly(mask_img, [polygon_int], 1)
            
            # Create RGBA overlay
            colored_mask = np.zeros((mask_height, mask_width, 4))  # RGBA
            for c in range(3):
                colored_mask[:, :, c] = mask_img * color[c]
            # Set alpha channel: 0 for background, mask_alpha for object pixels
            colored_mask[:, :, 3] = mask_img * mask_alpha
            
            # Overlay mask on image (only object pixels will be visible)
            ax.imshow(colored_mask)
            
            # Draw polygon contour
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth)

            # Close the polygon
            ax.plot([polygon[-1, 0], polygon[0, 0]], 
                   [polygon[-1, 1], polygon[0, 1]], color=color, linewidth=linewidth)
    
    # Draw prediction bounding boxes
    if draw_boxes:
        for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch (predictions in red)
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with class name and confidence
            label = f"{class_names[int(cls_id)]} {conf:.2f}"
            if has_masks:
                label += " [SEG]"
            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
            )
    
    if axis is False:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def obtain_attr(results):
    for n in dir(results):
        print(n, ":", getattr(results, n, None))
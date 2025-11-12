try:
    from .show_res import bgr2rgb
except ImportError:
    from show_res import bgr2rgb
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

def draw_boxes_gui(image_path):
    """
    Interactive GUI to draw bounding boxes on an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        list: List of bounding boxes in format [(x1, y1, x2, y2), ...]
        
    Usage:
        Click and drag to draw boxes. Press 'Enter' when done, 'u' to undo last box.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img_rgb = bgr2rgb(img)
    
    # Storage for boxes
    boxes = []
    current_box = {'start': None, 'rect': None}
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title("Click and drag to draw boxes | Press 'Enter' to finish | Press 'u' to undo")
    
    def on_press(event):
        """Mouse press event - start drawing box"""
        if event.inaxes != ax:
            return
        current_box['start'] = (event.xdata, event.ydata)
    
    def on_motion(event):
        """Mouse motion event - update box preview"""
        if event.inaxes != ax or current_box['start'] is None:
            return
        
        # Remove previous preview rectangle
        if current_box['rect'] is not None:
            current_box['rect'].remove()
        
        # Draw new preview rectangle
        x1, y1 = current_box['start']
        x2, y2 = event.xdata, event.ydata
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        current_box['rect'] = rect
        fig.canvas.draw_idle()
    
    def on_release(event):
        """Mouse release event - finalize box"""
        if event.inaxes != ax or current_box['start'] is None:
            return
        
        x1, y1 = current_box['start']
        x2, y2 = event.xdata, event.ydata
        
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Only add if box has area
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))
            
            # Remove preview rectangle
            if current_box['rect'] is not None:
                current_box['rect'].remove()
            
            # Draw final rectangle
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add box number label
            ax.text(
                x1, y1 - 5,
                f"Box {len(boxes)}",
                color='white',
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
            )
            
            print(f"Box {len(boxes)}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Center: ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f})")
        
        # Reset current box
        current_box['start'] = None
        current_box['rect'] = None
        fig.canvas.draw_idle()
    
    def on_key(event):
        """Key press event - handle undo and finish"""
        if event.key == 'u' and len(boxes) > 0:
            # Undo last box
            boxes.pop()
            # Redraw everything
            ax.clear()
            ax.imshow(img_rgb)
            ax.set_title("Click and drag to draw boxes | Press 'Enter' to finish | Press 'u' to undo")
            
            # Redraw all remaining boxes
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5,
                    f"Box {i+1}",
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
                )
            
            fig.canvas.draw_idle()
            print(f"Undone. {len(boxes)} boxes remaining.")
        
        elif event.key == 'enter':
            # Finish and close
            plt.close(fig)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinished! Total boxes drawn: {len(boxes)}")
    return boxes

if __name__ == "__main__":
    # Check if a command-line argument was provided
    if len(sys.argv) > 1:
        # If an argument is provided, use it as the image path
        # sys.argv[0] is the script name, sys.argv[1] is the first argument
        image_path_from_arg = sys.argv[1]
        print(f"Using argument-provided path: {image_path_from_arg}")
        draw_boxes_gui(image_path_from_arg)
    else:
        print('errored, please enter 1 str only')
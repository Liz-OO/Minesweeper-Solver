import cv2
import numpy as np

import cv2
import numpy as np

def detect_minesweeper_grid(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binary threshold to separate the board from background
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect the board outline
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the game board
    board_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle of the board
    x, y, w, h = cv2.boundingRect(board_contour)
    
    # Create a mask for the game board region
    mask = np.zeros_like(gray)
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Detect edges within the board area
    edges = cv2.Canny(masked_img, 50, 150)
    
    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=w/10, maxLineGap=10)
    
    # Create a clean image for the grid
    grid_img = np.zeros_like(img)
    
    # Filter and classify lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle to classify as horizontal or vertical
            if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                horizontal_lines.append((y1 + y2) // 2)  # Store y-coordinate
            else:  # Vertical line
                vertical_lines.append((x1 + x2) // 2)  # Store x-coordinate
    
    # Remove duplicates by clustering nearby lines
    def cluster_lines(lines, max_gap=10):
        if not lines:
            return []
        
        # Sort lines
        sorted_lines = sorted(lines)
        clusters = []
        current_cluster = [sorted_lines[0]]
        
        for i in range(1, len(sorted_lines)):
            if sorted_lines[i] - sorted_lines[i-1] <= max_gap:
                current_cluster.append(sorted_lines[i])
            else:
                clusters.append(int(sum(current_cluster) / len(current_cluster)))
                current_cluster = [sorted_lines[i]]
                
        if current_cluster:
            clusters.append(int(sum(current_cluster) / len(current_cluster)))
            
        return clusters
    
    # Cluster the horizontal and vertical lines
    horizontal_clusters = cluster_lines(horizontal_lines)
    vertical_clusters = cluster_lines(vertical_lines)
    
    # Detect grid size (number of cells)
    grid_rows = len(horizontal_clusters) - 1
    grid_cols = len(vertical_clusters) - 1
    
    print(f"Detected grid size: {grid_rows}x{grid_cols}")
    
    # Draw the grid
    for y_line in horizontal_clusters:
        cv2.line(grid_img, (x, y_line), (x + w, y_line), (0, 255, 0), 1)
        
    for x_line in vertical_clusters:
        cv2.line(grid_img, (x_line, y), (x_line, y + h), (0, 255, 0), 1)
    
    # Create a grid of cells based on the lines
    cells = []
    for i in range(len(horizontal_clusters) - 1):
        row = []
        for j in range(len(vertical_clusters) - 1):
            x1 = vertical_clusters[j]
            y1 = horizontal_clusters[i]
            x2 = vertical_clusters[j + 1]
            y2 = horizontal_clusters[i + 1]
            
            cell_contour = np.array([
                [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
            ])
            row.append(cell_contour)
        cells.append(row)
    
    # Draw cells on the original image
    cells_img = img.copy()
    for row in cells:
        for cell in row:
            cv2.drawContours(cells_img, [cell], -1, (0, 0, 255), 2)
    
    return grid_img, cells_img, cells

def identify_cells_and_create_grid(image_path):
    # Detect the grid 
    contour_img, grid_img, grid_cells = detect_minesweeper_grid(image_path)

    # Read the original image
    original_img = cv2.imread(image_path)
   
    # Directly extract and classify each cell
    grid_data = []
    
    # Process each row of cells
    for row in grid_cells:
        data_row = []
        for cell_contour in row:
            # Get the cell bounds from the contour
            x_coords = [pt[0][0] for pt in cell_contour]
            y_coords = [pt[0][1] for pt in cell_contour]
            
            # Calculate the cell bounds
            left = min(x_coords)
            right = max(x_coords)
            top = min(y_coords)
            bottom = max(y_coords)
            
            # Extract the cell image
            cell_img = original_img[top:bottom, left:right]
            
            # Add a small margin to avoid border effects
            margin = 2
            if left + margin < right - margin and top + margin < bottom - margin:
                cell_img = original_img[top+margin:bottom-margin, left+margin:right-margin]
            
            # Classify what's in the cell
            cell_value = identify_cell_content(cell_img)
            data_row.append(cell_value)
        
        grid_data.append(data_row)
    
    return grid_data, contour_img, grid_img

def extract_cell(image, left, top, right, bottom):
    """Extract a cell from the image given its boundary coordinates"""
    # Extract the cell
    cell = image[top:bottom, left:right]
    
    # Resize to a standard size for classification
    standard_size = (28, 28)
    if cell.size > 0:
        cell = cv2.resize(cell, standard_size)
    
    return cell

def identify_cell_content(cell_img):
    """Identify what's inside a cell (number, empty, unopened, etc.)"""
    # Make sure the cell image is valid
    if cell_img is None or cell_img.size == 0 or cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
        return 'unknown'
    
    # Use both RGB and grayscale features for classification
    # For Windows XP Minesweeper
    return color_based_classification_xp(cell_img)

def color_based_classification_xp(cell_img):
    """Classify cell based on Windows XP Minesweeper colors"""
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    
    # Calculate average colors and texture
    avg_bgr = np.mean(cell_img, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    avg_gray = np.mean(gray)
    std_gray = np.std(gray)
    
    # Get dominant colors
    bgr_dominant = []
    for i in range(3):
        hist = cv2.calcHist([cell_img], [i], None, [256], [0, 256])
        bgr_dominant.append(np.argmax(hist))
    
    # Check edges for 3D effect
    edges = cv2.Canny(gray, 100, 200)
    edge_percent = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

    # Flag detection for red and black pattern
    black_pixels = np.sum((cell_img[:,:,0] < 60) & (cell_img[:,:,1] < 60) & (cell_img[:,:,2] < 60))
    red_pixels = np.sum((cell_img[:,:,2] > 180) & (cell_img[:,:,0] < 100) & (cell_img[:,:,1] < 100))
    if black_pixels > 10 and red_pixels > 10 and std_gray > 70:
        return -2

    # 4 (Dark Blue)
    if (165 < avg_bgr[0] < 175 and 
        130 < avg_bgr[1] < 135 and 
        130 < avg_bgr[2] < 135 and
        130 < avg_gray < 140 and
        75 < std_gray < 85 and
        0.16 < edge_percent < 0.17):
        return 4
    
    # 1 (Blue)
    if avg_bgr[0] > avg_bgr[1] + 15 and avg_bgr[0] > avg_bgr[2] + 15:
        return 1
    
    # 2 (Green) 
    if avg_bgr[1] > avg_bgr[0] + 15 and avg_bgr[1] > avg_bgr[2] + 15:
        return 2
    
    # 3 (Red)
    if avg_bgr[2] > avg_bgr[0] + 15 and avg_bgr[2] > avg_bgr[1] + 15:
        return 3
        
    # 5 (Dark Red)
    if avg_bgr[2] > avg_bgr[0] + 10 and avg_bgr[2] > avg_bgr[1] + 10 and avg_gray < 170:
        return 5
    
    # 6 (Cyan)
    if avg_bgr[0] > avg_bgr[2] + 15 and avg_bgr[1] > avg_bgr[2] + 15:
        return 6
    
    # 7 (Black)
    if avg_gray < 120:
        return 7
    
    # 8 (Gray)
    if 120 < avg_gray < 170 and std_gray < 20:
        return 8
    
    # Empty cells have edge percentages < 0.085
    if edge_percent < 0.085:
        return 0
    
    # All other cells are unopened
    return -1

# Example usage
if __name__ == "__main__":
    
    grid_data, contour_img, grid_img = identify_cells_and_create_grid('images/flagtestsm.png')
    
    # Print the grid data
    print("Minesweeper Grid:")
    for row in grid_data:
        print(row)
    
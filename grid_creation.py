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
    """
    Enhanced function to identify cells and create a grid using ratio-based classification
    """
    # Detect the grid 
    contour_img, grid_img, grid_cells = detect_minesweeper_grid(image_path)

    # Read the original image
    original_img = cv2.imread(image_path)
   
    # First pass: extract features and collect statistics
    cell_features = []
    edge_percentages = []
    cell_images = []  # Store the actual cell images
    
    for row_idx, row in enumerate(grid_cells):
        row_features = []
        row_images = []
        for col_idx, cell_contour in enumerate(row):
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
            
            margin = 2
            margin_applied = False
            if left + margin < right - margin and top + margin < bottom - margin:
                cell_img = original_img[top+margin:bottom-margin, left+margin:right-margin]
                top = top + margin
                left = left + margin
                bottom = bottom - margin
                right = right - margin
                margin_applied = True
            
            # Extract features for this cell
            features = extract_cell_features(cell_img)
            features['row'] = row_idx
            features['col'] = col_idx
            features['top'] = top
            features['left'] = left
            features['bottom'] = bottom
            features['right'] = right
            features['margin_applied'] = margin_applied
            
            row_features.append(features)
            row_images.append(cell_img) 
            edge_percentages.append(features['edge_percent'])
        
        cell_features.append(row_features)
        cell_images.append(row_images)
    
    if edge_percentages:
        sorted_edges = sorted(edge_percentages)
        
        # Try to find natural clustering in the edge percentages
        # This uses a simplified k-means approach with k=2 (empty vs. unopened)
        if len(sorted_edges) >= 2:
            centroid1 = sorted_edges[0] 
            centroid2 = sorted_edges[-1]
            
            for _ in range(5):
                cluster1 = []
                cluster2 = []
                
                for edge in sorted_edges:
                    if abs(edge - centroid1) < abs(edge - centroid2):
                        cluster1.append(edge)
                    else:
                        cluster2.append(edge)
                
                if cluster1:
                    centroid1 = sum(cluster1) / len(cluster1)
                if cluster2:
                    centroid2 = sum(cluster2) / len(cluster2)
            
            if cluster1 and cluster2:
                adaptive_threshold = (max(cluster1) + min(cluster2)) / 2
            else:
                adaptive_threshold = np.median(sorted_edges)
        else:
            adaptive_threshold = np.median(sorted_edges)
    else:
        adaptive_threshold = 0.05  # Default fallback
    
    # Classify cells using adaptive threshold and other features
    grid_data = []
    
    for row_idx, row_features in enumerate(cell_features):
        data_row = []
        for col_idx, features in enumerate(row_features):
            # Use the exact same image that was used for feature extraction
            cell_img = cell_images[row_idx][col_idx]
            
            # Use adaptive thresholding based on the edge percentage distribution
            cell_value = classify_cell_with_adaptive_threshold(
                cell_img, 
                features, 
                adaptive_threshold
            )
            
            data_row.append(cell_value)
        
        grid_data.append(data_row)
    
    return grid_data, contour_img, grid_img

def extract_cell_features(cell_img):
    """Extract comprehensive features from a cell for classification"""
    if cell_img is None or cell_img.size == 0 or cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
        return {
            'avg_bgr': [0, 0, 0],
            'avg_gray': 0,
            'std_gray': 0,
            'edge_percent': 0,
            'height': 0,
            'width': 0
        }
    
    height, width = cell_img.shape[:2]
    
    # Convert to different color spaces
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_percent = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    
    # Color statistics
    avg_bgr = np.mean(cell_img, axis=(0, 1))
    std_bgr = np.std(cell_img, axis=(0, 1))
    
    # Grayscale statistics
    avg_gray = np.mean(gray)
    std_gray = np.std(gray)
    
    # Compute horizontal and vertical gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # Brightness in different regions
    h, w = gray.shape
    top_left = gray[:h//2, :w//2]
    top_right = gray[:h//2, w//2:]
    bottom_left = gray[h//2:, :w//2]
    bottom_right = gray[h//2:, w//2:]
    
    avg_tl = np.mean(top_left)
    avg_tr = np.mean(top_right)
    avg_bl = np.mean(bottom_left)
    avg_br = np.mean(bottom_right)
    
    return {
        'avg_bgr': avg_bgr,
        'std_bgr': std_bgr,
        'avg_gray': avg_gray,
        'std_gray': std_gray,
        'edge_percent': edge_percent,
        'avg_gradient': avg_gradient,
        'brightness_tl': avg_tl,
        'brightness_tr': avg_tr,
        'brightness_bl': avg_bl,
        'brightness_br': avg_br,
        'brightness_diff': avg_tl - avg_br, 
        'height': height,
        'width': width
    }

def classify_cell_with_adaptive_threshold(cell_img, features, adaptive_threshold):
    """
    Classify a cell using adaptive thresholding based on image characteristics
    """
    # Flag detection with added edge checking
    black_pixels = np.sum((cell_img[:,:,0] < 60) & (cell_img[:,:,1] < 60) & (cell_img[:,:,2] < 60))
    red_pixels = np.sum((cell_img[:,:,2] > 180) & (cell_img[:,:,0] < 100) & (cell_img[:,:,1] < 100))
    std_gray = features['std_gray']
    edge_percent = features['edge_percent']
    
    # Flags should have significant edges due to the flag shape and pole
    if black_pixels > 10 and red_pixels > 10 and std_gray > 40 and edge_percent > 0.01:
        return -2 
    
    avg_bgr = features['avg_bgr']
    avg_gray = features['avg_gray']
    
    # 4 (Dark Blue)
    if (165 < avg_bgr[0] < 175 and 
        130 < avg_bgr[1] < 135 and 
        130 < avg_bgr[2] < 135 and
        130 < avg_gray < 140 and
        75 < std_gray < 85):
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
    
    # For empty vs. unopened classification, use the adaptive threshold
    edge_percent = features['edge_percent']
    brightness_diff = features['brightness_diff']
    avg_gradient = features['avg_gradient']
    
    # Define several factors to consider
    factors = []
    
    # Edge percentage relative to the adaptive threshold
    factors.append(1 if edge_percent > adaptive_threshold else 0)
    
    # Brightness difference (3D effect)
    factors.append(1 if brightness_diff > 10 else 0)
    
    # Gradient magnitude
    factors.append(1 if avg_gradient > 15 else 0)
    
    # Grayscale standard deviation
    factors.append(1 if std_gray > 30 else 0)
    
    # Make decision based on majority of factors
    if sum(factors) >= 2: 
        return -1  # Unopened
    else:
        return 0   # Empty


if __name__ == "__main__":

    image_path = 'images/IMG_0307.jpg'
    
    grid_data, contour_img, grid_img = identify_cells_and_create_grid(image_path)
    
    # Print the grid data
    print("Minesweeper Grid:")
    for row in grid_data:
        print(row)
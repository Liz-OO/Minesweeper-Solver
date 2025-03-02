def basic_pattern_det(grid_data):
    """
    Finds basic patterns (i.e. 1 only touching 1 unopened square becomes a flag)
    """

    numbers = [1,2,3,4,5,6,7,8]
    flag = -2
    empty = 0
    unopened = -1

    new_grid = [[None for col in row] for row in grid_data]
    for row_idx, row in enumerate(grid_data):
        for col_idx, value in enumerate(row):
            if grid_data[row_idx][col_idx] in numbers:
                new_grid[row_idx][col_idx] = grid_data[row_idx][col_idx]
                # Check surroundings: if num -1 = num, then all -2
                print(row_idx, col_idx)
                surroundings = check_surroundings(row_idx, col_idx, grid_data)
                unopen_count = 0
                unopened_positions = []
                
                # Get the positions of unopened cells
                for i, item in enumerate(surroundings):
                    if item is not None and item not in numbers and item != empty:
                        unopen_count += 1
                        # Convert surrounding index to relative coordinates
                        relative_coords = index_to_relative_coords(i)
                        # Calculate absolute coordinates
                        abs_row = row_idx + relative_coords[0]
                        abs_col = col_idx + relative_coords[1]
                        unopened_positions.append((abs_row, abs_col))
                
                # If number of unopened cells equals the number on the cell, mark all as flags
                if unopen_count == grid_data[row_idx][col_idx]:
                    # Place flags in the new grid
                    for pos_row, pos_col in unopened_positions:
                        new_grid[pos_row][pos_col] = flag
            
            # Copy over unopened or empty cells if not already set as a flag
            elif new_grid[row_idx][col_idx] is None:
                new_grid[row_idx][col_idx] = grid_data[row_idx][col_idx]

    return new_grid


def one_two_x_det(grid_data):
    """
    Find 1-2-X pattern and mark mines accordingly.
    """
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    flag = -2
    empty = 0
    unopened = -1

    # Create a copy of the original grid
    new_grid = [[grid_data[row][col] for col in range(len(grid_data[0]))] for row in range(len(grid_data))]
    
    rows = len(grid_data)
    cols = len(grid_data[0])
    
    # Function to check if coordinates are in bounds
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols
    
    # Function to count ALL unopened/flagged cells around a position
    def count_total_unopened(r, c):
        count = 0
        unopened_cells = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc):
                    if grid_data[nr][nc] == unopened or grid_data[nr][nc] == flag:
                        count += 1
                        unopened_cells.append((nr, nc))
        return count, unopened_cells
    
    # Scan the grid for cells with value 2
    for row in range(rows):
        for col in range(cols):
            if grid_data[row][col] == 2:
                # First check: cell must have EXACTLY 3 unopened cells total
                total_unopened, unopened_cells = count_total_unopened(row, col)
                
                if total_unopened != 3:
                    continue  # Skip if not exactly 3 unopened cells
                
                # Check if these 3 cells form a horizontal or vertical line
                rows_set = set(cell[0] for cell in unopened_cells)
                cols_set = set(cell[1] for cell in unopened_cells)
                
                forms_horizontal_line = len(rows_set) == 1  # All cells in the same row
                forms_vertical_line = len(cols_set) == 1    # All cells in the same column
                
                if not (forms_horizontal_line or forms_vertical_line):
                    continue  # Skip if not in a straight horizontal or vertical line
                
                # Check for an adjacent 1
                adjacents = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Cardinal directions only
                    r1 = row + dr
                    c1 = col + dc
                    
                    if in_bounds(r1, c1) and grid_data[r1][c1] == 1:
                        adjacents.append((r1, c1, dr, dc))
                
                # If no adjacent 1, skip
                if not adjacents:
                    continue
                
                # For each adjacent 1, check if it overlaps with any of the 3 unopened cells
                for r1, c1, dr, dc in adjacents:
                    # Get cells adjacent to the 1
                    _, cells_near_1 = count_total_unopened(r1, c1)
                    
                    # Check for overlap - some cells must be seen by both 1 and 2
                    overlap = [cell for cell in unopened_cells if cell in cells_near_1]
                    
                    if not overlap:
                        continue
                    
                    # Calculate the diagonal position away from the 1
                    if forms_horizontal_line:
                        # For horizontal line of cells, the mine is at the diagonal, away from the 1
                        if dc != 0:  # 1 is horizontally adjacent to 2
                            diag_r = unopened_cells[0][0]  # Row of the horizontal line of cells
                            diag_c = col - dc  # Move away from the 1
                            
                            if (diag_r, diag_c) in unopened_cells:
                                new_grid[diag_r][diag_c] = flag
                    
                    elif forms_vertical_line:
                        # For vertical line of cells, the mine is at the diagonal, away from the 1
                        if dr != 0:  # 1 is vertically adjacent to 2
                            diag_r = row - dr  # Move away from the 1
                            diag_c = unopened_cells[0][1]  # Column of the vertical line of cells
                            
                            if (diag_r, diag_c) in unopened_cells:
                                new_grid[diag_r][diag_c] = flag
    
    return new_grid

def empty_cell_det(grid_data):
    """
    Function that determines which cells can safely be selected due to the correct number of flags being found
    returns grid with new opened spaces
    """
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    flag = -2
    empty = 0
    unopened = -1

    # Create a copy of the original grid
    new_grid = [[grid_data[row][col] for col in range(len(grid_data[0]))] for row in range(len(grid_data))]
    
    # Scan through the grid
    for row_num in range(len(grid_data)):
        for col_num in range(len(grid_data[0])):
            # Only process if the current cell is a number
            if grid_data[row_num][col_num] in numbers:
                # Get surrounding cells
                surroundings = check_surroundings(row_num, col_num, grid_data)
                
                # Count flags and unopened cells
                flag_count = 0
                unopened_count = 0
                unopened_cells = []
                
                # Analyze surroundings
                for idx, cell_value in enumerate(surroundings):
                    if cell_value == flag:
                        flag_count += 1
                    elif cell_value == unopened:
                        unopened_count += 1
                        # Convert surrounding index to relative coordinates
                        relative_coords = index_to_relative_coords(idx)
                        # Calculate absolute coordinates
                        abs_row = row_num + relative_coords[0]
                        abs_col = col_num + relative_coords[1]
                        unopened_cells.append((abs_row, abs_col))
                
                # If the number of flags matches the cell's number, open the remaining cells
                if flag_count == grid_data[row_num][col_num]:
                    for ur, uc in unopened_cells:
                        new_grid[ur][uc] = empty  # Mark as safely openable
    
    return new_grid

def index_to_relative_coords(index):
    """
    Converts a surrounding index (0-7) to relative coordinates.
    
    0, 1, 2,
    7, x, 3
    6, 5, 4
    """
    relative_coords = [
        (-1, -1),  # 0: upper left
        (-1, 0),   # 1: up
        (-1, 1),   # 2: upper right
        (0, 1),    # 3: right
        (1, 1),    # 4: lower right
        (1, 0),    # 5: down
        (1, -1),   # 6: lower left
        (0, -1)    # 7: left
    ]
    return relative_coords[index]

def check_surroundings(row_num, col_num, grid_data):
    """
    starts with upper left and goes around clockwise to gather all relavant sides.
    0, 1, 2,
    7, x, 3
    6, 5, 4
    """
    grid_row_len = len(grid_data)
    grid_col_len = len(grid_data[0])

    surroundings = [None] * 8
    if row_num - 1 >= 0 and col_num - 1 >= 0:
        surroundings[0] = grid_data[row_num - 1][col_num - 1]
    
    if row_num - 1 >= 0:
        surroundings[1] = grid_data[row_num - 1][col_num]

    if row_num -1 >= 0 and col_num + 1 < grid_col_len:
        surroundings[2] = grid_data[row_num-1][col_num+1]
    
    if col_num + 1 < grid_col_len:
        surroundings[3] = grid_data[row_num][col_num+1]
    
    if col_num + 1 < grid_col_len and row_num + 1 < grid_row_len:
        surroundings[4] = grid_data[row_num + 1][col_num + 1]

    if row_num + 1 < grid_row_len:
        surroundings[5] = grid_data[row_num + 1][col_num]
    
    if row_num + 1 < grid_row_len and col_num - 1 >= 0:
        surroundings[6] = grid_data[row_num + 1][col_num - 1]
    
    if col_num - 1 >= 0:
        surroundings[7] = grid_data[row_num][col_num - 1]        

    print(surroundings)
    return surroundings

def visualize_grid(grid):
    """
    Creates a visual representation of the Minesweeper grid for debugging
    """
    # Character mappings for different cell types
    cell_chars = {
        -2: '⚑',  # Flag
        -1: '□',  # Unopened
        0: ' '    # Empty
    }
    
    # Maximum width needed for any cell (for alignment)
    max_width = 2
    
    # Build the horizontal separator line
    width = len(grid[0])
    separator = '+' + '-' * (max_width + 2) * width + '+'
    
    # Print the grid
    print(separator)
    for row in grid:
        line = '|'
        for cell in row:
            if cell is None:
                char = '?'
            elif cell in cell_chars:
                char = cell_chars[cell]
            else:
                char = str(cell)  # Numbers 1-8
            
            # Pad with spaces for alignment
            line += f' {char:^{max_width}} '
        line += '|'
        print(line)
    print(separator)

if __name__ == "__main__":
    from grid_creation import identify_cells_and_create_grid
    
    grid_data, _, _ = identify_cells_and_create_grid('images/5050.jpeg')

    new_grid = basic_pattern_det(grid_data)

    visualize_grid(new_grid)

    newer_grid = one_two_x_det(new_grid)

    visualize_grid(newer_grid)

    remed_grid = empty_cell_det(newer_grid)

    visualize_grid(remed_grid)
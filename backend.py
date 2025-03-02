from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import List, Dict, Any

from grid_creation import identify_cells_and_create_grid
from minesweeper_logic import (
    basic_pattern_det, 
    one_two_x_det, 
    empty_cell_det, 
)

app = FastAPI(title="Minesweeper Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

def save_uploaded_file(file: UploadFile) -> str:
    """
    Save the uploaded file to a temporary location
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            # Read the uploaded file contents
            contents = file.file.read()
            
            # Write contents to the temporary file
            temp_file.write(contents)
            
            # Return the path to the temporary file
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def process_grid(grid_data: List[List[int]]) -> Dict[str, Any]:
    """
    Apply solving algorithms to the grid
    """

    grid_with_flags = basic_pattern_det(grid_data)
    grid_with_more_flags = one_two_x_det(grid_with_flags)
    safe_cells = empty_cell_det(grid_with_more_flags)
    
    new_flags = []
    new_safe_cells = []
    
    for row_idx, row in enumerate(grid_data):
        for col_idx, cell in enumerate(row):
            # Check for new flags
            if grid_with_more_flags[row_idx][col_idx] == -2 and cell not in [-2]:
                new_flags.append({
                    "row": row_idx, 
                    "col": col_idx
                })
            
            # Check for new safe cells
            if safe_cells[row_idx][col_idx] == 0 and cell == -1:
                new_safe_cells.append({
                    "row": row_idx, 
                    "col": col_idx
                })
    
    return {
        "original_grid": grid_data,
        "grid_with_flags": grid_with_flags,
        "grid_with_more_flags": grid_with_more_flags,
        "safe_cells": safe_cells,
        "new_flags": new_flags,
        "new_safe_cells": new_safe_cells
    }

@app.post("/analyze-board")
async def analyze_minesweeper_board(file: UploadFile = File(...)):
    """
    Endpoint to analyze a Minesweeper board image
    """
    try:
        # Save the uploaded file
        image_path = save_uploaded_file(file)
        
        try:
            # Detect grid and cells
            grid_data, _, _ = identify_cells_and_create_grid(image_path)
            
            # Process the grid
            processed_grid = process_grid(grid_data)
            
            return JSONResponse(content=processed_grid)
        
        finally:
            # Clean up the temporary file
            os.unlink(image_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing board: {str(e)}")

@app.post("/solve-grid")
async def solve_minesweeper_grid(grid: List[List[int]]):
    """
    Endpoint to solve a Minesweeper grid directly
    """
    try:
        processed_grid = process_grid(grid)
        return JSONResponse(content=processed_grid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving grid: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
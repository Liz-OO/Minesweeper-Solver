import { Component } from '@angular/core';
import { MinesweeperService, MinesweeperAnalysisResult } from '../services/minesweeper.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  selectedFile: File | null = null;
  imagePreview: string | ArrayBuffer | null = null;
  isAnalyzing: boolean = false;
  analysisResult: MinesweeperAnalysisResult | null = null;
  errorMessage: string | null = null;

  // Added for large board support
  cellSize: number = 30; // Default cell size
  minCellSize: number = 16; // Minimum cell size
  maxCellSize: number = 40; // Maximum cell size
  defaultCellSize: number = 30; // Default cell size for reset
  
  // Grid dimensions
  gridSize: { rows: number, cols: number } = { rows: 0, cols: 0 };

  constructor(private minesweeperService: MinesweeperService) {}

  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    
    if (file) {
      this.selectedFile = file;
      this.analysisResult = null;
      this.errorMessage = null;
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        this.imagePreview = e.target?.result ?? null;
      };
      reader.readAsDataURL(file);
    }
  }

  analyzeImage(): void {
    if (!this.selectedFile) {
      this.errorMessage = 'No file selected';
      return;
    }

    this.isAnalyzing = true;
    this.errorMessage = null;
    
    this.minesweeperService.analyzeBoard(this.selectedFile)
      .subscribe({
        next: (result: MinesweeperAnalysisResult) => {
          this.analysisResult = result;
          this.isAnalyzing = false;
          console.log('Analysis result:', result);
          
          // Added: Auto-adjust cell size for large boards
          this.updateGridSize();
          this.autoCellSize();
        },
        error: (error: any) => {
          console.error('Error analyzing board:', error);
          this.errorMessage = 'Failed to analyze the board. Please try again with a clearer image.';
          this.isAnalyzing = false;
        }
      });
  }

  // Added: Update grid dimensions
  updateGridSize(): void {
    if (this.analysisResult) {
      this.gridSize = {
        rows: this.analysisResult.original_grid.length,
        cols: this.analysisResult.original_grid[0].length
      };
    }
  }

  // Added: Automatically adjust cell size based on grid dimensions
  autoCellSize(): void {
    if (this.gridSize.rows > 16 || this.gridSize.cols > 30) {
      // For expert boards (typically 16x30 or larger)
      const containerWidth = 700; // Approximate container width
      const containerHeight = 400; // Approximate container height
      
      // Calculate size based on available space and grid dimensions
      this.cellSize = Math.max(this.minCellSize, Math.min(
        Math.floor(containerWidth / this.gridSize.cols), 
        Math.floor(containerHeight / this.gridSize.rows),
        this.defaultCellSize  // Don't go larger than default
      ));
    } else {
      // For smaller boards, use default size
      this.cellSize = this.defaultCellSize;
    }
  }

  // Added: Update cell size (handled by ngModel)
  updateCellSize(): void {
    // This is intentionally left empty as ngModel handles the binding
  }

  // Added: Zoom in
  zoomIn(): void {
    this.cellSize = Math.min(this.cellSize + 2, this.maxCellSize);
  }

  // Added: Zoom out
  zoomOut(): void {
    this.cellSize = Math.max(this.cellSize - 2, this.minCellSize);
  }

  // Added: Reset zoom to auto-calculated level
  resetZoom(): void {
    this.autoCellSize();
  }

  // Helper methods for display
  getCellClass(cell: number): string {
    switch (cell) {
      case -4: return 'mine';
      case -2: return 'flag';
      case -1: return 'unopened';
      case 0: return 'empty';
      case 1: return 'one';
      case 2: return 'two';
      case 3: return 'three';
      case 4: return 'four';
      case 5: return 'five';
      case 6: return 'six';
      case 7: return 'seven';
      case 8: return 'eight';
      default: return '';
    }
  }

  getCellContent(cell: number): string {
    switch (cell) {
      case -2: return '🚩'; 
      case -1: return ''; 
      case 0: return ''; 
      default: return cell.toString();
    }
  }

  isHighlightedCell(row: number, col: number): boolean {
    if (!this.analysisResult) return false;
    
    // Check if it's a newly identified flag
    const isNewFlag = this.analysisResult.new_flags.some(
      (flag: {row: number, col: number}) => flag.row === row && flag.col === col
    );
    
    // Check if it's a newly identified safe cell
    const isNewSafeCell = this.analysisResult.new_safe_cells.some(
      (safeCell: {row: number, col: number}) => safeCell.row === row && safeCell.col === col
    );
    
    return isNewFlag || isNewSafeCell;
  }
}
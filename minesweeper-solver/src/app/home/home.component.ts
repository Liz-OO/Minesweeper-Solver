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
        },
        error: (error: any) => {
          console.error('Error analyzing board:', error);
          this.errorMessage = 'Failed to analyze the board. Please try again with a clearer image.';
          this.isAnalyzing = false;
        }
      });
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
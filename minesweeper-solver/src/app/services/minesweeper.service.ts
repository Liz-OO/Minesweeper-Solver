import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface MinesweeperAnalysisResult {
  original_grid: number[][];
  grid_with_flags: number[][];
  grid_with_more_flags: number[][];
  safe_cells: number[][];
  new_flags: { row: number, col: number }[];
  new_safe_cells: { row: number, col: number }[];
}

@Injectable({
  providedIn: 'root'
})
export class MinesweeperService {
  private apiUrl = 'https://minesweeper-solver-api.onrender.com';

  constructor(private http: HttpClient) { }

  analyzeBoard(file: File): Observable<MinesweeperAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.http.post<MinesweeperAnalysisResult>(`${this.apiUrl}/analyze-board`, formData);
  }

  solveGrid(grid: number[][]): Observable<MinesweeperAnalysisResult> {
    return this.http.post<MinesweeperAnalysisResult>(`${this.apiUrl}/solve-grid`, grid);
  }
}
export interface UploadResponse {
  session_id: string;
  row_count: number;
  columns: string[];
  preview: Record<string, string | number>[];
}

export interface AnalyzeResponse {
  accuracy: number;
  bias_score: number;
  raw_dpd: number;
  equalized_odds_diff?: number;
  groups: string[];
  group_rates: Record<string, number>;
  is_biased: boolean;
  sensitive_col: string;
  explanation: string;
}

export interface MitigateResponse {
  after_bias_score: number;
  after_accuracy: number;
  is_fixed: boolean;
  explanation: string;
}

export interface ChatResponse {
  answer: string;
}
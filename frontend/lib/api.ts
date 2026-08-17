import { UploadResponse, AnalyzeResponse, MitigateResponse, ChatResponse } from "./types";
import { getToken } from "./auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function authFetch(url: string, options: RequestInit = {}) {
  const token = getToken();
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return handle<{ access_token: string }>(res);
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await authFetch(`${API_BASE}/upload`, { method: "POST", body: formData });
  return handle<UploadResponse>(res);
}

export async function analyzeBias(
  sessionId: string,
  labelCol: string,
  sensitiveCol: string
): Promise<AnalyzeResponse> {
  const res = await authFetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, label_col: labelCol, sensitive_col: sensitiveCol }),
  });
  return handle<AnalyzeResponse>(res);
}

export async function mitigateBias(
  sessionId: string,
  labelCol: string,
  sensitiveCol: string
): Promise<MitigateResponse> {
  const res = await authFetch(`${API_BASE}/mitigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, label_col: labelCol, sensitive_col: sensitiveCol }),
  });
  return handle<MitigateResponse>(res);
}

export async function askQuestion(sessionId: string, question: string): Promise<ChatResponse> {
  const res = await authFetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
  return handle<ChatResponse>(res);
}

export async function downloadReport(sessionId: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/report/${sessionId}`, {
    method: "POST",
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      body.detail || "Failed to generate report"
    );
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "bias_report.pdf";
  a.click();

  URL.revokeObjectURL(url);
}
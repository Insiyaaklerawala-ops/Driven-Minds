"use client";

import { useState } from "react";
import { Panel } from "./ui/Panel";
import { UploadZone } from "./UploadZone";
import { Select } from "./ui/Select";
import { uploadCsv, analyzeBias } from "@/lib/api";
import { AnalyzeResponse } from "@/lib/types";
import { Zap } from "lucide-react";

interface ControlPanelProps {
  onAnalysisComplete: (
    results: AnalyzeResponse,
    sessionId: string,
    labelCol: string,
    sensitiveCol: string
  ) => void;
}

export function ControlPanel({ onAnalysisComplete }: ControlPanelProps) {
  const [fileName, setFileName] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [labelCol, setLabelCol] = useState("");
  const [sensitiveCol, setSensitiveCol] = useState("");

  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFileSelect(file: File) {
    setError(null);
    setIsUploading(true);
    setFileName(file.name);
    try {
      const res = await uploadCsv(file);
      setSessionId(res.session_id);
      setColumns(res.columns);
      setLabelCol("");
      setSensitiveCol("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setFileName(null);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAnalyze() {
    if (!sessionId || !labelCol || !sensitiveCol) return;
    setError(null);
    setIsAnalyzing(true);
    try {
      const results = await analyzeBias(sessionId, labelCol, sensitiveCol);
      onAnalysisComplete(results, sessionId, labelCol, sensitiveCol);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }

  const canAnalyze = sessionId && labelCol && sensitiveCol && labelCol !== sensitiveCol && !isAnalyzing;

  return (
    <Panel>
      <h2 className="font-display font-medium text-sm text-ink-muted uppercase tracking-wide mb-4">
        Control Panel
      </h2>

      <div className="space-y-5">
        <UploadZone fileName={fileName} isUploading={isUploading} onFileSelect={handleFileSelect} />

        {columns.length > 0 && (
          <>
            <Select
              label="Target column"
              value={labelCol}
              onChange={setLabelCol}
              options={columns}
              placeholder="What the model predicts"
            />
            <Select
              label="Sensitive column"
              value={sensitiveCol}
              onChange={setSensitiveCol}
              options={columns.filter((c) => c !== labelCol)}
              placeholder="Group to check for bias"
              disabled={!labelCol}
            />
          </>
        )}

        {error && (
          <p className="text-xs text-alert font-mono bg-alert/10 border border-alert/30 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          className="w-full flex items-center justify-center gap-2 bg-signal text-void font-display font-semibold
                     text-sm rounded-lg py-3 transition-opacity
                     disabled:opacity-30 disabled:cursor-not-allowed
                     hover:opacity-90"
        >
          <Zap className="w-4 h-4" />
          {isAnalyzing ? "Analyzing..." : "Analyze Bias"}
        </button>
      </div>
    </Panel>
  );
}
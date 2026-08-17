"use client";

import { useState } from "react";
import { Panel } from "../ui/Panel";
import { mitigateBias, downloadReport } from "@/lib/api";
import { AnalyzeResponse, MitigateResponse } from "@/lib/types";
import { Wrench, Download, ArrowRight } from "lucide-react";

interface MitigationPanelProps {
  sessionId: string;
  results: AnalyzeResponse;
  labelCol: string;
  sensitiveCol: string;
}

export function MitigationPanel({ sessionId, results, labelCol, sensitiveCol }: MitigationPanelProps) {
  const [after, setAfter] = useState<MitigateResponse | null>(null);
  const [isFixing, setIsFixing] = useState(false);
  const [error, setError] = useState<string | null>(null);

 async function handleFix() {
  setIsFixing(true);
  setError(null);

  try {
    const res = await mitigateBias(sessionId, labelCol, sensitiveCol);
    setAfter(res);
  } catch (e) {
    setError(e instanceof Error ? e.message : "Mitigation failed");
  } finally {
    setIsFixing(false);
  }
}

async function handleDownloadReport() {
  try {
    setError(null);
    await downloadReport(sessionId);
  } catch (e) {
    setError(e instanceof Error ? e.message : "Report download failed");
  }
}
  return (
    <Panel>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-ink-faint" />
          <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
            {results.is_biased ? "Fix the Bias" : "Mitigation"}
          </span>
        </div>
                <button
          onClick={handleDownloadReport}
          className="flex items-center gap-1.5 text-xs font-mono text-ink-muted hover:text-signal transition-colors"
        >
          <Download className="w-3.5 h-3.5" />
          PDF report
        </button>
      </div>

      {results.is_biased && !after && (
        <>
          <p className="text-sm text-ink-faint mb-4">
            Apply fairness-constrained retraining and see the improvement.
          </p>
          <button
            onClick={handleFix}
            disabled={isFixing}
            className="bg-alert/10 border border-alert/40 text-alert font-display font-medium text-sm
                       rounded-lg px-4 py-2.5 hover:bg-alert/20 transition-colors disabled:opacity-50"
          >
            {isFixing ? "Applying mitigation..." : "Fix Bias Now"}
          </button>
          {error && <p className="text-xs text-alert font-mono mt-3">{error}</p>}
        </>
      )}

      {after && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 font-mono">
            <div className="flex-1 bg-panel-raised rounded-lg p-3 border border-alert/30">
              <div className="text-xs text-ink-faint mb-1">before</div>
              <div className="text-xl text-alert font-semibold">{results.bias_score.toFixed(3)}</div>
            </div>
            <ArrowRight className="w-5 h-5 text-ink-faint shrink-0" />
            <div className="flex-1 bg-panel-raised rounded-lg p-3 border border-signal/30">
              <div className="text-xs text-ink-faint mb-1">after</div>
              <div className="text-xl text-signal font-semibold">{after.after_bias_score.toFixed(3)}</div>
            </div>
          </div>

          <div className="text-xs font-mono text-ink-muted">
            accuracy: {results.accuracy}% → {after.after_accuracy}%
          </div>

          <p className="text-sm text-ink-primary leading-relaxed border-t border-border pt-4">
            {after.explanation}
          </p>
        </div>
      )}

      {!results.is_biased && (
        <p className="text-sm text-ink-faint">
          No significant bias detected — mitigation isn't needed here.
        </p>
      )}
    </Panel>
  );
}
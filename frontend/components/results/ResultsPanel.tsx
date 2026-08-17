import { Panel } from "../ui/Panel";
import { MetricCard } from "./MetricCard";
import { ParityPulse } from "./ParityPulse";
import { AnalyzeResponse } from "@/lib/types";
import { Activity, Target, ShieldAlert, ShieldCheck } from "lucide-react";

export function ResultsPanel({ results }: { results: AnalyzeResponse }) {
  return (
    <div className="space-y-6">
      <Panel glow={!results.is_biased}>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          <MetricCard
            label="Model Accuracy"
            value={`${results.accuracy}%`}
            icon={<Target className="w-4 h-4 text-ink-faint" />}
          />
          <MetricCard
            label="Bias Score"
            value={results.bias_score.toFixed(3)}
            status={results.is_biased ? "bad" : "good"}
            sublabel={results.is_biased ? "above 0.1 threshold" : "within threshold"}
            icon={<Activity className="w-4 h-4 text-ink-faint" />}
          />
          <MetricCard
            label="Verdict"
            value={results.is_biased ? "BIASED" : "FAIR"}
            status={results.is_biased ? "bad" : "good"}
            sublabel={results.is_biased ? "needs attention" : "all good"}
            icon={
              results.is_biased ? (
                <ShieldAlert className="w-4 h-4 text-alert" />
              ) : (
                <ShieldCheck className="w-4 h-4 text-signal" />
              )
            }
          />
        </div>

        <div className="border-t border-border pt-6">
          <ParityPulse groupRates={results.group_rates} sensitiveCol={results.sensitive_col} />
        </div>
      </Panel>

      <Panel>
        <span className="text-xs font-mono text-ink-muted uppercase tracking-wide block mb-3">
          AI Analysis
        </span>
        <p className="text-sm text-ink-primary leading-relaxed">{results.explanation}</p>
      </Panel>
    </div>
  );
}
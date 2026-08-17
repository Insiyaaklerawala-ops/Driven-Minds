import clsx from "clsx";
import { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string;
  status?: "neutral" | "good" | "bad";
  sublabel?: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, status = "neutral", sublabel, icon }: MetricCardProps) {
  return (
    <div className="bg-panel-raised border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">{label}</span>
        {icon}
      </div>
      <div
        className={clsx(
          "font-mono font-semibold text-2xl leading-none",
          status === "good" && "text-signal",
          status === "bad" && "text-alert",
          status === "neutral" && "text-ink-primary"
        )}
      >
        {value}
      </div>
      {sublabel && (
        <div
          className={clsx(
            "text-xs font-mono mt-1.5",
            status === "good" && "text-signal-dim",
            status === "bad" && "text-alert-dim",
            status === "neutral" && "text-ink-faint"
          )}
        >
          {sublabel}
        </div>
      )}
    </div>
  );
}
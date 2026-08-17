"use client";

import { useState } from "react";
import { Panel } from "./ui/Panel";
import { ChevronDown, Database } from "lucide-react";
import clsx from "clsx";

interface DatasetInfo {
  name: string;
  file: string;
  target: string;
  sensitive: string[];
  description: string;
}

const DATASETS: DatasetInfo[] = [
  {
    name: "Adult Income",
    file: "adult.csv",
    target: "income",
    sensitive: ["gender", "race"],
    description: "Predicts whether someone earns above $50K/year based on census data.",
  },
  {
    name: "COMPAS",
    file: "compas.csv",
    target: "two_year_recid",
    sensitive: ["race"],
    description: "Predicts likelihood of re-offending within two years, used in criminal justice risk scoring.",
  },
  {
    name: "German Credit",
    file: "german_credit_data.csv",
    target: "Risk",
    sensitive: ["Sex"],
    description: "Predicts credit risk (good/bad) for loan applicants based on financial and personal data.",
  },
];

export function DatasetGuide() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <Panel>
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-4 h-4 text-ink-faint" />
        <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
          Sample Datasets
        </span>
      </div>

      <div className="space-y-2">
        {DATASETS.map((ds, i) => {
          const isOpen = openIndex === i;
          return (
            <div key={ds.file} className="border border-border rounded-lg overflow-hidden">
              <button
                onClick={() => setOpenIndex(isOpen ? null : i)}
                className="w-full flex items-center justify-between px-3 py-2.5 text-left
                           hover:bg-panel-raised transition-colors"
              >
                <div>
                  <div className="text-sm text-ink-primary font-medium">{ds.name}</div>
                  <div className="text-xs font-mono text-ink-faint">{ds.file}</div>
                </div>
                <ChevronDown
                  className={clsx(
                    "w-4 h-4 text-ink-faint transition-transform shrink-0",
                    isOpen && "rotate-180"
                  )}
                />
              </button>

              {isOpen && (
                <div className="px-3 pb-3 pt-1 border-t border-border bg-panel-raised/40">
                  <p className="text-xs text-ink-muted leading-relaxed mb-3">{ds.description}</p>

                  <div className="space-y-2">
                    <div>
                      <span className="text-[10px] font-mono text-ink-faint uppercase tracking-wide block mb-1">
                        Target column
                      </span>
                      <code className="text-xs font-mono text-signal bg-signal/10 border border-signal/20 rounded px-2 py-1">
                        {ds.target}
                      </code>
                    </div>

                    <div>
                      <span className="text-[10px] font-mono text-ink-faint uppercase tracking-wide block mb-1">
                        Sensitive column{ds.sensitive.length > 1 ? "s" : ""}
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {ds.sensitive.map((s) => (
                          <code
                            key={s}
                            className="text-xs font-mono text-alert bg-alert/10 border border-alert/20 rounded px-2 py-1"
                          >
                            {s}
                          </code>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-border">
        <span className="text-[10px] font-mono text-ink-faint uppercase tracking-wide block mb-2">
          Bias score scale
        </span>
        <div className="space-y-1.5 text-xs font-mono">
          <div className="flex justify-between">
            <span className="text-ink-muted">0.00 – 0.05</span>
            <span className="text-signal">no significant bias</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ink-muted">0.05 – 0.10</span>
            <span className="text-[#F5C147]">minor bias</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ink-muted">0.10 – 0.20</span>
            <span className="text-alert">significant bias</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ink-muted">0.20+</span>
            <span className="text-alert font-semibold">severe — action required</span>
          </div>
        </div>
      </div>
    </Panel>
  );
}
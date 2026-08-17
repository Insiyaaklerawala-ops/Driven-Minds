"use client";

import { useState } from "react";
import { CommandBar } from "@/components/CommandBar";
import { ControlPanel } from "@/components/ControlPanel";
import { Panel } from "@/components/ui/Panel";
import { AnalyzeResponse } from "@/lib/types";
import { ResultsPanel } from "@/components/results/ResultsPanel";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { MitigationPanel } from "@/components/results/MitigationPanel";
import { DatasetGuide } from "@/components/DatasetGuide";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";

export default function Home() {
  const router = useRouter();
  const [checked, setChecked] = useState(false);
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [labelCol, setLabelCol] = useState("");
  const [sensitiveCol, setSensitiveCol] = useState("");

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login");
    } else {
      setChecked(true);
    }
  }, [router]);

  if (!checked) return null;

  function handleAnalysisComplete(
    res: AnalyzeResponse,
    sid: string,
    label: string,
    sensitive: string
  ) {
    setResults(res);
    setSessionId(sid);
    setLabelCol(label);
    setSensitiveCol(sensitive);
  }

  return (
    <div className="min-h-screen bg-instrument">
      <CommandBar />

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
        <div className="space-y-6">
          <ControlPanel onAnalysisComplete={handleAnalysisComplete} />
          <DatasetGuide />
        </div>

        <div className="space-y-6">
          {results ? (
            <Panel>
               <ResultsPanel results={results} />
               <MitigationPanel
               sessionId={sessionId!}
               results={results}
               labelCol={labelCol}
               sensitiveCol={sensitiveCol}
               />
               <ChatPanel sessionId={sessionId!} />
            </Panel>
          ) : (
            <Panel>
              <p className="text-sm text-ink-faint">
                Upload a dataset and run analysis to see results.
              </p>
            </Panel>
          )}
        </div>
      </main>
    </div>
  );
}
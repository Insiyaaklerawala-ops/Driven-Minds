"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, ShieldCheck, ArrowRight } from "lucide-react";
import { login } from "@/lib/api";
import { saveToken } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    try {
      const res = await login(username, password);
      saveToken(res.access_token);
      router.push("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-instrument flex items-center justify-center px-6">
      <div className="w-full max-w-sm">
        {/* brand mark */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="w-9 h-9 rounded-md bg-signal/10 border border-signal/30 flex items-center justify-center">
            <div className="w-2.5 h-2.5 rounded-full bg-signal animate-pulse" />
          </div>
          <div>
            <h1 className="font-display font-semibold text-lg leading-none">
              UNBIASED<span className="text-signal">.AI</span>
            </h1>
            <p className="text-xs text-ink-faint font-mono mt-0.5">bias detection engine</p>
          </div>
        </div>

        {/* login panel */}
        <div className="relative bg-panel border border-border rounded-xl p-6 overflow-hidden">
          {/* top scanline accent */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-signal/60 to-transparent" />

          <div className="flex items-center gap-2 mb-6">
            <Lock className="w-4 h-4 text-ink-faint" />
            <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
              Restricted Access
            </span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-mono text-ink-muted uppercase tracking-wide mb-2">
                Username
              </label>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                type="text"
                autoComplete="username"
                className="w-full bg-panel-raised border border-border rounded-lg px-3 py-2.5
                           text-sm text-ink-primary font-mono
                           focus:outline-none focus:border-signal/60 focus:ring-1 focus:ring-signal/30"
                placeholder="judge"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-ink-muted uppercase tracking-wide mb-2">
                Password
              </label>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                autoComplete="current-password"
                className="w-full bg-panel-raised border border-border rounded-lg px-3 py-2.5
                           text-sm text-ink-primary font-mono
                           focus:outline-none focus:border-signal/60 focus:ring-1 focus:ring-signal/30"
                placeholder="••••••••"
              />
            </div>

            {error && (
              <p className="text-xs text-alert font-mono bg-alert/10 border border-alert/30 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={isLoading || !username || !password}
              className="w-full flex items-center justify-center gap-2 bg-signal text-void font-display font-semibold
                         text-sm rounded-lg py-3 transition-opacity
                         disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-90"
            >
              {isLoading ? "Authenticating..." : "Access Dashboard"}
              {!isLoading && <ArrowRight className="w-4 h-4" />}
            </button>
          </form>
        </div>

        <div className="flex items-center justify-center gap-1.5 mt-6 text-xs font-mono text-ink-faint">
          <ShieldCheck className="w-3.5 h-3.5" />
          session encrypted · access logged
        </div>
      </div>
    </div>
  );
}
export function CommandBar() {
  return (
    <header className="border-b border-border bg-panel/50 backdrop-blur-sm sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-signal/10 border border-signal/30 flex items-center justify-center">
            <div className="w-2.5 h-2.5 rounded-full bg-signal animate-pulse" />
          </div>
          <div>
            <h1 className="font-display font-semibold text-lg leading-none">
              UNBIASED<span className="text-signal">.AI</span>
            </h1>
            <p className="text-xs text-ink-faint font-mono mt-0.5">
              bias detection engine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-ink-muted">
          <span className="w-1.5 h-1.5 rounded-full bg-signal" />
          system online
        </div>
      </div>
    </header>
  );
}
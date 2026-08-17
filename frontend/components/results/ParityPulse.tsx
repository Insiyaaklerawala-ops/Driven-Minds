interface ParityPulseProps {
  groupRates: Record<string, number>;
  sensitiveCol: string;
}

export function ParityPulse({ groupRates, sensitiveCol }: ParityPulseProps) {
  const entries = Object.entries(groupRates);
  const average = entries.reduce((sum, [, v]) => sum + v, 0) / entries.length;

  const width = 600;
  const height = 180;
  const padding = 40;
  const centerY = height / 2;
  const usableHeight = height / 2 - padding / 2;

  const barWidth = Math.min(80, (width - padding * 2) / entries.length - 24);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
          Parity Pulse — {sensitiveCol}
        </span>
        <span className="text-xs font-mono text-ink-faint">
          avg {Math.round(average * 100)}%
        </span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {/* equilibrium line */}
        <line
          x1={padding}
          y1={centerY}
          x2={width - padding}
          y2={centerY}
          stroke="var(--color-ink-faint)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        <text x={width - padding} y={centerY - 8} textAnchor="end" className="fill-ink-faint" fontSize={10} fontFamily="var(--font-mono)">
          equilibrium
        </text>

        {entries.map(([group, rate], i) => {
          const deviation = rate - average;
          const barHeight = Math.abs(deviation) * usableHeight * 4; // amplify for visibility
          const cappedHeight = Math.min(barHeight, usableHeight);
          const x = padding + 20 + i * ((width - padding * 2 - 20) / entries.length);
          const isAbove = deviation >= 0;
          const y = isAbove ? centerY - cappedHeight : centerY;

          const severity = Math.abs(deviation);
          const color =
            severity > 0.1 ? "var(--color-alert)" : severity > 0.05 ? "#F5C147" : "var(--color-signal)";

          return (
            <g key={group}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={cappedHeight || 2}
                rx={3}
                fill={color}
                opacity={0.85}
              />
              <circle
                cx={x + barWidth / 2}
                cy={isAbove ? y : y + cappedHeight}
                r={4}
                fill={color}
              />
              <text
                x={x + barWidth / 2}
                y={height - 12}
                textAnchor="middle"
                className="fill-ink-muted"
                fontSize={11}
                fontFamily="var(--font-mono)"
              >
                {group}
              </text>
              <text
                x={x + barWidth / 2}
                y={isAbove ? y - 10 : y + cappedHeight + 16}
                textAnchor="middle"
                fill={color}
                fontSize={11}
                fontFamily="var(--font-mono)"
                fontWeight={600}
              >
                {Math.round(rate * 100)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
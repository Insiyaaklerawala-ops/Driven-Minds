interface SelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  disabled?: boolean;
  placeholder?: string;
}

export function Select({ label, value, onChange, options, disabled, placeholder }: SelectProps) {
  return (
    <div>
      <label className="block text-xs font-mono text-ink-muted uppercase tracking-wide mb-2">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-panel-raised border border-border rounded-lg px-3 py-2.5
                   text-sm text-ink-primary font-mono
                   focus:outline-none focus:border-signal/60 focus:ring-1 focus:ring-signal/30
                   disabled:opacity-40 disabled:cursor-not-allowed
                   appearance-none cursor-pointer"
      >
        <option value="" disabled>
          {placeholder || "Select..."}
        </option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
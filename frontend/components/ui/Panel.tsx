import { ReactNode } from "react";
import clsx from "clsx";

export function Panel({
  children,
  className,
  glow = false,
}: {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={clsx(
        "bg-panel border border-border rounded-xl p-6",
        glow && "shadow-[0_0_24px_-8px_var(--color-signal)]",
        className
      )}
    >
      {children}
    </div>
  );
}
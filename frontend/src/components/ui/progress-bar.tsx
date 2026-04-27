import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  label?: string;
  className?: string;
}

export function ProgressBar({ value, label, className }: ProgressBarProps) {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;

  return (
    <div className={cn("flex w-full flex-col gap-2", className)}>
      {label ? (
        <div className="text-xs font-medium text-[var(--ink-soft)]">
          {label}
        </div>
      ) : null}
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--panel-soft)]">
        <div
          className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--accent-2))] transition-[width] duration-300"
          style={{ width: `${safe}%` }}
        />
      </div>
    </div>
  );
}

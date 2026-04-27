import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type Tone = "neutral" | "success" | "warning" | "danger" | "info";

const toneClass: Record<Tone, string> = {
  neutral: "bg-[var(--panel-soft)] text-[var(--ink-soft)]",
  success: "bg-[color:var(--ok-faint)] text-[var(--ok)]",
  warning: "bg-[color:var(--warn-faint)] text-[var(--warn)]",
  danger: "bg-[color:var(--danger-faint)] text-[var(--danger)]",
  info: "bg-[color:var(--accent-faint)] text-[var(--accent)]",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold tracking-wide",
        toneClass[tone],
        className,
      )}
      {...props}
    />
  );
}

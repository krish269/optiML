import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  loading?: boolean;
}

const variantMap: Record<Variant, string> = {
  primary:
    "bg-[var(--accent)] text-white shadow-[0_12px_24px_-14px_var(--accent-shadow)] hover:bg-[var(--accent-strong)]",
  secondary:
    "bg-[var(--panel-soft)] text-[var(--ink)] border border-[var(--line)] hover:bg-[var(--panel)]",
  ghost:
    "bg-transparent text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--panel-soft)]",
};

export function Button({
  className,
  variant = "primary",
  loading = false,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex h-11 items-center justify-center rounded-xl px-4 text-sm font-medium transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50",
        variantMap[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? "Working..." : children}
    </button>
  );
}

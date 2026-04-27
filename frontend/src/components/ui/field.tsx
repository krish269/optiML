import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
} from "react";

import { cn } from "@/lib/utils";

interface FieldWrapperProps {
  label: string;
  hint?: string;
  children: ReactNode;
}

export function FieldWrapper({ label, hint, children }: FieldWrapperProps) {
  return (
    <label className="flex w-full flex-col gap-2 text-sm text-[var(--ink)]">
      <span className="font-medium">{label}</span>
      {children}
      {hint ? (
        <span className="text-xs text-[var(--ink-soft)]">{hint}</span>
      ) : null}
    </label>
  );
}

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "h-11 rounded-xl border border-[var(--line)] bg-white px-3 text-sm text-[var(--ink)] outline-none transition-colors focus:border-[var(--accent)] focus:ring-2 focus:ring-[color:var(--accent-faint)]",
        props.className,
      )}
    />
  );
}

export function SelectInput(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "h-11 rounded-xl border border-[var(--line)] bg-white px-3 text-sm text-[var(--ink)] outline-none transition-colors focus:border-[var(--accent)] focus:ring-2 focus:ring-[color:var(--accent-faint)]",
        props.className,
      )}
    />
  );
}

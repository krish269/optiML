import { cn } from "@/lib/utils";

interface StepperProps {
  steps: string[];
  currentStep: number;
}

export function Stepper({ steps, currentStep }: StepperProps) {
  return (
    <div className="grid gap-3 md:grid-cols-4">
      {steps.map((step, index) => {
        const active = index === currentStep;
        const completed = index < currentStep;

        return (
          <div
            key={step}
            className={cn(
              "rounded-xl border px-3 py-3 transition-colors",
              active && "border-[var(--accent)] bg-[color:var(--accent-faint)]",
              completed &&
                "border-[color:var(--ok-faint)] bg-[color:var(--ok-faint)]",
              !active && !completed && "border-[var(--line)] bg-[var(--panel)]",
            )}
          >
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--ink-soft)]">
              Step {index + 1}
            </div>
            <div className="text-sm font-semibold text-[var(--ink)]">
              {step}
            </div>
          </div>
        );
      })}
    </div>
  );
}

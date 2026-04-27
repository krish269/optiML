"use client";

import dynamic from "next/dynamic";
import type { Layout, PlotData } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="h-[320px] w-full animate-pulse rounded-xl bg-[var(--panel-soft)]" />
  ),
});

interface PlotChartProps {
  data: Partial<PlotData>[];
  layout?: Partial<Layout>;
  height?: number;
}

export function PlotChart({ data, layout, height = 320 }: PlotChartProps) {
  return (
    <Plot
      data={data as PlotData[]}
      layout={{
        autosize: true,
        height,
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        margin: { l: 40, r: 20, t: 30, b: 40 },
        font: { color: "#44524A", family: "var(--font-body)" },
        ...layout,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}

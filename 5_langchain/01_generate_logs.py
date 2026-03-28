from __future__ import annotations

from langchain_paths import ANOMALY_LOG_CSV_PATH, AUTO_LOG_CSV_PATH, INPUT_XLSX_PATH, ensure_output_dirs
from document_workflow import (
    build_anomaly_log,
    build_auto_log,
    build_event_segments,
    load_input_events,
    resolve_generation_context,
)


def main() -> None:
    ensure_output_dirs()
    input_frame = load_input_events(INPUT_XLSX_PATH)
    events_df = build_event_segments(input_frame)
    if events_df.empty:
        raise SystemExit("No A/S/D events were found in the input file.")

    auto_log_df, short_indices, _, missing_process_pairs = build_auto_log(events_df)
    generation_context = resolve_generation_context()
    anomaly_df = build_anomaly_log(events_df, short_indices, missing_process_pairs, generation_context)

    auto_log_df.to_csv(AUTO_LOG_CSV_PATH, index=False, encoding="utf-8-sig")
    anomaly_df.to_csv(ANOMALY_LOG_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {AUTO_LOG_CSV_PATH}")
    print(f"[SAVE] {ANOMALY_LOG_CSV_PATH}")
    print(f"[INFO] LLM mode: {'enabled' if generation_context.llm_enabled else 'fallback'}")
    print(f"[INFO] Reason: {generation_context.reason}")


if __name__ == "__main__":
    main()

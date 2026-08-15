# Reproducing every paper artifact

    cd C:\Users\Alberto\Desktop\AgenticRag
    python paper\code\make_figures.py
        regenerates the 21 selected figures from agentic\results\
        into paper\figures_regen\_selected  - must end: VERIFIED 21/21
    python paper\code\stats\phase2_analysis.py
        recomputes every revision statistic -> paper\tables_stats\phase2
    python paper\code\plot_reliability.py
        reliability panels + reliability_scores.csv (default source:
        agentic\results\final_prognostic; use --dir for another run)

Figure modules live in paper\code\figures\ and write to
paper\figures_regen\<set>\ (both standalone and via make_figures).
The committed selected set is paper\figures\ (png + pdf + SELECTED.md).

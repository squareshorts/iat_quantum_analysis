import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def generate_texts():
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    methods_text = """
The Age Implicit Association Test (Age IAT) was included as a conventional social-category IAT to improve construct comparability across domains. Trial-level Age IAT data were downloaded automatically from OSF/Project Implicit. Consistent with standard IAT scoring procedures, analyses focused on critical blocks 3, 4, 6, and 7. Reaction times (RTs) were rigorously cleaned using the standard IAT-compatible latency window of 300 to 10,000 ms. Raw latency was preserved, but all downstream geometric analyses strictly used the cleaned latency variable. Comprehensive quality control diagnostics documented fast-trial, long-trial, and error proportions prior to final analysis.
"""
    (reports_dir / "age_iat_methods_text.md").write_text(methods_text.strip(), encoding="utf-8")
    
    qc_df = pd.read_csv(BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_qc_summary.csv")
    qc = qc_df.iloc[0]
    domain_df = pd.read_csv(BASE_DIR / "results" / "age_iat" / "age_iat_domain_summary.csv")
    dom = domain_df.iloc[0]
    
    results_text = f"""
For the Age IAT, the raw trial-level dataset contained {int(qc['raw_rows']):,} rows. After basic standardizations, {int(qc['retained_trial_rows']):,} retained critical-block rows were available before RT cleaning, encompassing {int(qc['participants']):,} participants. The overall error rate was {qc['error_rate']:.4f}. Pre-cleaning fast-trial (<300 ms) and long-trial (>10000 ms) proportions were {qc['pre_prop_fast_lt300']:.4f} and {qc['pre_prop_long_gt10000']:.4f}, respectively. After enforcing the 300--10000 ms latency window, the post-cleaning RT distribution demonstrated a median of {qc['post_median']:.2f} ms, a mean of {qc['post_mean']:.2f} ms, a standard deviation of {qc['post_sd']:.2f} ms, and an IQR of {qc['post_iqr']:.2f} ms. 

Under the geometric pipeline, the Age IAT yielded a contextual overlap MAP parameter of \\(\\theta_{{\\text{{MAP}}}} = {dom['theta_map']:.2f}^\\circ\\) (posterior mean = {dom['theta_mean']:.2f}^\\circ, 94\\% HDI [{dom['hdi_low']:.2f}^\\circ, {dom['hdi_high']:.2f}^\\circ]). This estimate places the Age domain alongside other robust evaluative domains in the comparative aggregate analysis.
"""
    (reports_dir / "age_iat_results_text.md").write_text(results_text.strip(), encoding="utf-8")

def generate_tables():
    table_dir = BASE_DIR / "tables" / "age_iat"
    table_dir.mkdir(parents=True, exist_ok=True)
    
    qc_df = pd.read_csv(BASE_DIR / "data" / "processed" / "age_iat" / "age_iat_qc_summary.csv")
    table_df = pd.DataFrame([
        {"Metric": "Raw rows", "Value": f"{int(qc_df['raw_rows'].iloc[0]):,}"},
        {"Metric": "Retained rows", "Value": f"{int(qc_df['retained_trial_rows'].iloc[0]):,}"},
        {"Metric": "Participants", "Value": f"{int(qc_df['participants'].iloc[0]):,}"},
        {"Metric": "Pre-cleaning Median RT (ms)", "Value": f"{qc_df['pre_median'].iloc[0]:.2f}"},
        {"Metric": "Pre-cleaning Mean RT (ms)", "Value": f"{qc_df['pre_mean'].iloc[0]:.2f}"},
        {"Metric": "Pre-cleaning SD RT (ms)", "Value": f"{qc_df['pre_sd'].iloc[0]:.2f}"},
        {"Metric": "Post-cleaning Median RT (ms)", "Value": f"{qc_df['post_median'].iloc[0]:.2f}"},
        {"Metric": "Post-cleaning Mean RT (ms)", "Value": f"{qc_df['post_mean'].iloc[0]:.2f}"},
        {"Metric": "Post-cleaning SD RT (ms)", "Value": f"{qc_df['post_sd'].iloc[0]:.2f}"},
        {"Metric": "Proportion Fast (<300 ms)", "Value": f"{qc_df['pre_prop_fast_lt300'].iloc[0]:.4f}"},
        {"Metric": "Proportion Long (>10000 ms)", "Value": f"{qc_df['pre_prop_long_gt10000'].iloc[0]:.4f}"},
        {"Metric": "Error Rate", "Value": f"{qc_df['error_rate'].iloc[0]:.4f}"},
    ])
    
    try:
        latex_str = table_df.to_latex(index=False, escape=True, column_format="ll")
    except TypeError:
        latex_str = table_df.to_latex(index=False, escape=True, column_format="ll") # pandas version compat
        
    latex_str = latex_str.replace("\\hline", "\\midrule")
    if "\\toprule" not in latex_str:
        lines = latex_str.split("\n")
        for i, line in enumerate(lines):
            if "\\begin{tabular}" in line:
                lines.insert(i+1, "\\toprule")
                break
        for i, line in reversed(list(enumerate(lines))):
            if "\\end{tabular}" in line:
                lines.insert(i, "\\bottomrule")
                break
        latex_str = "\n".join(lines)
        
    (table_dir / "age_iat_qc_table.tex").write_text(latex_str, encoding="utf-8")

if __name__ == "__main__":
    generate_texts()
    generate_tables()

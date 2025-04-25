import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compute_partial_eta_sq(anova_table, effect):
    """
    Compute partial eta squared for a given effect from an ANOVA table.
    """
    if effect in anova_table.index:
        ss_effect = anova_table.loc[effect, 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']
        return ss_effect / (ss_effect + ss_error)
    else:
        return None

def compute_hedges_g_for_groups(endog, groups):
    """
    Compute Hedges' g for all pairwise comparisons between levels of a factor.
    If a computed effect size is negative, it is multiplied by -1 and the comparison label is flipped.
    Returns a summary DataFrame with all positive Hedges' g values.
    """
    unique_groups = groups.unique()
    results = []
    for (g1, g2) in itertools.combinations(unique_groups, 2):
        group1 = endog[groups == g1]
        group2 = endog[groups == g2]
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            continue
        sd1, sd2 = group1.std(ddof=1), group2.std(ddof=1)
        # Pooled standard deviation
        sp = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        d = (group1.mean() - group2.mean()) / sp
        # Correction for small sample sizes
        J = 1 - (3 / (4*(n1 + n2) - 9))
        hedges_g = d * J
        # If hedges_g is negative, flip it and reverse the group order:
        if hedges_g < 0:
            hedges_g = -hedges_g
            comp = f"{g2} vs {g1}"
        else:
            comp = f"{g1} vs {g2}"
        results.append({"Comparison": comp, "Hedges_g": hedges_g})
    return pd.DataFrame(results)

def plot_hedges(hedges_g_df, factor_label, sort_by='name', channel=None, save_dir='.'):
    """
    Create and save a bar chart showing Hedges' g for pairwise comparisons.
    """
    if hedges_g_df.empty:
        return
    if sort_by == 'effect':
        hedges_g_df = hedges_g_df.sort_values(by='Hedges_g', ascending=False)
    elif sort_by == 'name':
        hedges_g_df = hedges_g_df.sort_values(by='Comparison')
    plt.figure(figsize=(10, 6))
    x = np.arange(len(hedges_g_df))
    plt.bar(x, hedges_g_df['Hedges_g'], color='steelblue')
    plt.xticks(x, hedges_g_df['Comparison'], rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    if channel is not None:
        title_text = f"Channel {channel}: Hedges' g for {factor_label} Pairwise Comparisons"
        fname = f"Hedges_g_{factor_label}_ch{channel}.png"
    else:
        title_text = f"Hedges' g for {factor_label} Pairwise Comparisons"
        fname = f"Hedges_g_{factor_label}.png"
    plt.title(title_text)
    plt.ylabel("Hedges' g")
    plt.xlabel("Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()

def run_tukey(endog, groups, factor_label, channel=None, save_dir='.'):
    """
    Run Tukey's HSD test, and save the summary and Hedges' g results.
    """
    tukey_result = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=0.05)
    tukey_summary = tukey_result.summary()
    out_text = f"\nPost hoc: Tukey HSD for {factor_label} factor\n{tukey_summary}\n" + "-"*80 + "\n"
    if channel is not None:
        tukey_fname = f"Tukey_{factor_label}_ch{channel}.txt"
    else:
        tukey_fname = f"Tukey_{factor_label}.txt"
    with open(os.path.join(save_dir, tukey_fname), 'w') as f:
        f.write(out_text)
    hedges_g_df = compute_hedges_g_for_groups(endog, groups)
    if not hedges_g_df.empty:
        if channel is not None:
            csv_fname = f"Hedges_g_{factor_label}_ch{channel}.csv"
        else:
            csv_fname = f"Hedges_g_{factor_label}.csv"
        hedges_g_df.to_csv(os.path.join(save_dir, csv_fname), index=False)
        plot_hedges(hedges_g_df, factor_label, channel=channel, save_dir=save_dir)

def run_simple_effects_generic(ch_df, group_by, effect, dependent, description_format, channel=None, save_dir='.'):
    """
    Conduct simple effects analysis using a generic function.
    
    For each unique value in the grouping variable, run a one-way ANOVA on 'effect'
    and, if significant, run Tukey's HSD for that simple effect.
    """
    unique_values = ch_df[group_by].unique()
    for val in unique_values:
        subset_df = ch_df[ch_df[group_by] == val]
        if subset_df[effect].nunique() > 1:
            model_subset = ols(f"{dependent} ~ C({effect})", data=subset_df).fit()
            anova_subset = sm.stats.anova_lm(model_subset, typ=2)
            if f"C({effect})" in anova_subset.index and anova_subset.loc[f"C({effect})", "PR(>F)"] < 0.05:
                desc = description_format.format(val=val)
                run_tukey(subset_df[dependent], subset_df[effect], desc, channel=channel, save_dir=save_dir)

def plot_interaction_generic(ch_df, ch, group1, group2, dv, save_dir='.'):
    """
    Create and save a jittered interaction plot for the DV, using group1 on the x-axis
    and different levels of group2 represented by jitter offsets.
    """
    summary = ch_df.groupby([group1, group2])[dv].agg(['mean', 'sem']).reset_index()
    pivot_mean = summary.pivot(index=group1, columns=group2, values='mean')
    pivot_sem  = summary.pivot(index=group1, columns=group2, values='sem')
    xvals = np.array(pivot_mean.index)
    groups = list(pivot_mean.columns)
    n_groups = len(groups)
    width = 0.2
    offsets = np.linspace(-width, width, n_groups)
    plt.figure(figsize=(8, 6))
    for i, g in enumerate(groups):
        jittered_x = xvals + offsets[i]
        plt.errorbar(jittered_x, pivot_mean[g], yerr=pivot_sem[g], marker='o', linestyle='none', label=f"{group2}: {g}")
    plt.xticks(xvals)
    title_text = f"Channel {ch}: {dv} by {group1} and {group2}"
    plt.title(title_text)
    plt.xlabel(group1)
    plt.ylabel(dv)
    plt.legend()
    plt.tight_layout()
    fname = f"Interaction_{ch}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()

def summarize_channel_results_generic(ch, anova_results, eta_sq_dict, pvals_dict, factors, dv, save_dir='.'):
    """
    Generate and save a narrative summary of the results for a given channel.
    
    Parameters:
      ch: Channel identifier.
      anova_results: The ANOVA table.
      eta_sq_dict: Dictionary mapping each factor (and interaction key) to its partial eta squared.
      pvals_dict: Dictionary mapping each effect to its p-value.
      factors: List of factor names (e.g. [f1, f2]).
      dv: Dependent variable name.
      save_dir: Output folder.
    """
    summary_text = f"--- Summary for Channel {ch} ({dv} Analysis) ---\n"
    summary_text += f"Two-way ANOVA on {dv} with factors {factors}:\n"
    summary_text += "  Partial Eta Squared values:\n"
    for key, eta in eta_sq_dict.items():
        summary_text += f"    - {key}: {eta:.3f}\n"
    summary_text += "\nP-values for main effects and interaction:\n"
    for key, pval in pvals_dict.items():
        summary_text += f"    - {key}: {pval:.3f}\n"
    summary_text += "--- End of Summary ---\n"
    summary_fname = f"Summary_Channel_{ch}.txt"
    with open(os.path.join(save_dir, summary_fname), 'w') as f:
        f.write(summary_text)
    return {"Channel": ch, **eta_sq_dict, **pvals_dict}

# ------------------- Analysis Pipeline Function -------------------

def run_analysis_pipeline(df, channels, dependent, factors, plot_groups=None, output_folder=".", alpha=0.05):
    """
    Run the complete analysis pipeline for a two-factor ANOVA across channels.
    
    Parameters:
      df: DataFrame containing your data.
      channels: List of channel identifiers to analyze.
      dependent: Dependent variable (e.g., "theta_power").
      factors: List of two factor names for the main effects (e.g., ["start_position", "cue_onset"]).
      plot_groups: Tuple of two group names for the interaction plot (defaults to factors).
      output_folder: Base folder for results (subfolders per channel are created).
      alpha: Significance threshold.
    
    The function:
      - Subsets data by channel.
      - Fits a two-way ANOVA with interaction.
      - Computes partial eta squared and p-values.
      - Saves a summary file.
      - Saves an interaction plot.
      - Runs Tukey's HSD (with Hedges' g) for each significant main effect.
      - Runs simple effects analyses if the interaction is significant.
    
    Returns a DataFrame consolidating summary metrics across channels.
    """
    consolidated = []
    if plot_groups is None:
        plot_groups = factors
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for ch in channels:
        chan_folder = os.path.join(output_folder, f"Channel_{ch:03}")
        if not os.path.exists(chan_folder):
            os.makedirs(chan_folder)
        
        ch_df = df[df['channel'] == f"{ch:03}"]
        formula = f"{dependent} ~ C({factors[0]}) * C({factors[1]})"
        model = ols(formula, data=ch_df).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)
        
        eta_sq_main1 = compute_partial_eta_sq(anova_results, f"C({factors[0]})")
        eta_sq_main2 = compute_partial_eta_sq(anova_results, f"C({factors[1]})")
        eta_sq_inter = compute_partial_eta_sq(anova_results, f"C({factors[0]}):C({factors[1]})")
        p_main1 = anova_results.loc[f"C({factors[0]})", "PR(>F)"]
        p_main2 = anova_results.loc[f"C({factors[1]})", "PR(>F)"]
        p_inter = anova_results.loc[f"C({factors[0]}):C({factors[1]})", "PR(>F)"]
        
        eta_dict = {factors[0]: eta_sq_main1,
                    factors[1]: eta_sq_main2,
                    f"{factors[0]}:{factors[1]}": eta_sq_inter}
        pval_dict = {factors[0]: p_main1,
                     factors[1]: p_main2,
                     f"{factors[0]}:{factors[1]}": p_inter}
        
        chan_summary = summarize_channel_results_generic(ch, anova_results, eta_dict, pval_dict, factors, dependent, save_dir=chan_folder)
        consolidated.append(chan_summary)
        
        plot_interaction_generic(ch_df, ch, group1=plot_groups[0], group2=plot_groups[1], dv=dependent, save_dir=chan_folder)
        
        if p_main1 < alpha:
            run_tukey(ch_df[dependent], ch_df[factors[0]], factors[0], channel=ch, save_dir=chan_folder)
        if p_main2 < alpha:
            run_tukey(ch_df[dependent], ch_df[factors[1]], factors[1], channel=ch, save_dir=chan_folder)
            
        if p_inter < alpha:
            run_simple_effects_generic(ch_df, group_by=factors[1], effect=factors[0], dependent=dependent,
                                       description_format=f"{factors[0]} (within {factors[1]} '{{val}}')", channel=ch, save_dir=chan_folder)
            run_simple_effects_generic(ch_df, group_by=factors[0], effect=factors[1], dependent=dependent,
                                       description_format=f"{factors[1]} (within {factors[0]} '{{val}}')", channel=ch, save_dir=chan_folder)
    
    consolidated_df = pd.DataFrame(consolidated)
    csv_fname = os.path.join(output_folder, "Consolidated_Summary.csv")
    consolidated_df.to_csv(csv_fname, index=False)
    return consolidated_df

# ------------------- Example Usage -------------------
# For a Cue Period Analysis:
# output_folder = "/Users/liuyuanwei/Documents/GitHub/FYP-Pyhipp/LFP_Analysis/Figures/20181105/Cue_Period_Analysis/Results"
# channels = interesting_ch[day]  # list of channels
# result_df = run_analysis_pipeline(cue_segments_df, channels, dependent="theta_power", 
#                                   factors=["start_position", "cue_onset"],
#                                   plot_groups=("start_position", "cue_onset"),
#                                   output_folder=output_folder, alpha=0.05)
#
# For a Pillar Analysis (if your data already include mapped columns such as "pillar_start" and "pillar_destination"):
# result_df = run_analysis_pipeline(cue_segments_df, channels, dependent="theta_power",
#                                   factors=["pillar_start", "pillar_destination"],
#                                   plot_groups=("pillar_start", "pillar_destination"),
#                                   output_folder="/Your/Desired/Output/Folder", alpha=0.05)

"""Build publication figures and source tables for the dataset chapter (Python only)."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulated_pandapower"
OUT = ROOT / "thesis" / "chapter_dataset" / "figures"
TABLES = ROOT / "thesis" / "chapter_dataset" / "tables"
OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "legend.frameon": False, "savefig.facecolor": "white",
})
COL = {"normal":"#A7ADB5", "technical":"#3478B8", "non_technical":"#E08B3E", "mixed":"#8A5AA6"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load():
    return {k: pd.read_csv(DATA / f"{k}_line_loss.csv") for k in ["technical", "non_technical", "mixed"]}


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.25)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    boxes = [
        (0.25, 3.8, 2.1, 1.25, "Empirical profiles", "Hourly load series\n(train.csv + test.csv)", "#EAF2F8"),
        (0.25, 1.25, 2.1, 1.25, "Network model", "IEEE 33-bus feeder\n(pandapower case33bw)", "#EAF2F8"),
        (3.0, 2.5, 2.25, 1.4, "Scenario engine", "Daily continuous events\nseed = 20260806", "#EDF5E8"),
        (5.95, 3.8, 2.3, 1.25, "Technical branch", "AC power flow +\nphysical interventions", "#E7F0FA"),
        (5.95, 1.25, 2.3, 1.25, "NTL branch", "Meter manipulation +\nenergy accounting", "#FBEEDD"),
        (9.0, 2.5, 2.55, 1.4, "RCA benchmark", "Type + cause + location\n+ severity + split", "#F1EAF5"),
    ]
    for x,y,w,h,t,s,c in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=.08",fc=c,ec="#4B5563",lw=.8))
        ax.text(x+.12,y+h-.3,t,weight="bold",fontsize=8,va="top"); ax.text(x+.12,y+.22,s,fontsize=6.7,va="bottom",linespacing=1.35)
    arrows=[((2.35,4.42),(3,3.45)),((2.35,1.88),(3,2.95)),((5.25,3.2),(5.95,4.42)),((5.25,3.2),(5.95,1.88)),((8.25,4.42),(9,3.45)),((8.25,1.88),(9,2.95))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color="#5F6872"))
    ax.text(.25,5.62,"Empirically anchored pandapower simulation for line-loss root-cause analysis",fontsize=10,weight="bold")
    ax.text(6.2,.35,"Energy balance: feeder input = metered load + technical loss + NTL",ha="center",fontsize=7,color="#374151")
    fig.tight_layout(); save(fig,"fig_4_1_dataset_workflow")


def fig2_composition(ds):
    fig, axs = plt.subplots(2,2,figsize=(7.2,5.6));
    # a
    counts=[]
    for name,d in ds.items():
        vc=d.loss_type.value_counts(); counts.append([vc.get(x,0) for x in COL])
    bottom=np.zeros(3); x=np.arange(3)
    for j,cls in enumerate(COL):
        vals=np.array(counts)[:,j]; axs[0,0].bar(x,vals,bottom=bottom,color=COL[cls],label=cls.replace('_',' '),width=.65); bottom+=vals
    axs[0,0].set_xticks(x,["Technical","NTL","Mixed"]); axs[0,0].set_ylabel("Samples"); axs[0,0].legend(ncol=2,fontsize=6,loc="upper left")
    # b
    tech=ds["technical"].copy(); base=tech.loc[tech.technical_root_cause.eq("normal"),"technical_loss_kw"].median()
    order=[x for x in TECH_ORDER if x in set(tech.technical_root_cause)]
    dat=[tech.loc[tech.technical_root_cause.eq(c),"technical_loss_kw"] for c in order]
    axs[0,1].boxplot(dat,showfliers=False,patch_artist=True,boxprops={"facecolor":"#8CB8DE","linewidth":.6},medianprops={"color":"#173F5F"})
    axs[0,1].axhline(base,ls="--",lw=.8,color="#777",label="normal median")
    axs[0,1].set_xticks(range(1,len(order)+1),[SHORT[c] for c in order],rotation=35,ha="right"); axs[0,1].set_ylabel("Technical loss (kW)")
    # c
    ntl=ds["non_technical"]; ordern=[x for x in NTL_ORDER if x in set(ntl.ntl_root_cause)]
    sev=[ntl.loc[ntl.ntl_root_cause.eq(c),"non_technical_loss_kw"] for c in ordern]
    axs[1,0].boxplot(sev,showfliers=False,patch_artist=True,boxprops={"facecolor":"#F1BB81","linewidth":.6},medianprops={"color":"#8A4B12"})
    axs[1,0].set_xticks(range(1,len(ordern)+1),[SHORT[c] for c in ordern],rotation=35,ha="right"); axs[1,0].set_ylabel("Non-technical loss (kW)")
    # d mixed cause matrix
    m=ds["mixed"].query("technical_anomaly==1 and ntl_anomaly==1")
    tab=pd.crosstab(m.technical_root_cause,m.ntl_root_cause).reindex(index=TECH_ORDER[1:],columns=NTL_ORDER[1:],fill_value=0)
    im=axs[1,1].imshow(tab,cmap="Purples",aspect="auto")
    axs[1,1].set_xticks(range(len(tab.columns)),[SHORT[c] for c in tab.columns],rotation=35,ha="right"); axs[1,1].set_yticks(range(len(tab.index)),[SHORT[c] for c in tab.index]); axs[1,1].set_xlabel("NTL cause"); axs[1,1].set_ylabel("Technical cause")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]): axs[1,1].text(j,i,str(tab.iat[i,j]),ha="center",va="center",fontsize=5.5,color="white" if tab.iat[i,j]>tab.values.max()*.55 else "#222")
    for label,ax in zip("abcd",axs.ravel()): ax.text(-.13,1.06,label,transform=ax.transAxes,weight="bold",fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.97,wspace=.34,hspace=.48); save(fig,"fig_4_2_dataset_composition")
    tab.to_csv(TABLES/"source_fig_4_2_mixed_cause_matrix.csv")


def fig3_physics(ds):
    d=ds["technical"].copy(); rng=np.random.default_rng(12); sample=d.iloc[rng.choice(len(d),min(2200,len(d)),replace=False)]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.45))
    for cause,color in [("normal",COL["normal"]),("overload","#3478B8"),("line_resistance_increase","#B24A45")]:
        s=sample[sample.technical_root_cause.eq(cause)]; axs[0].scatter(s.actual_load_kw,s.technical_loss_kw,s=5,alpha=.38,color=color,label=SHORT[cause],edgecolors="none")
    axs[0].set_xlabel("Actual load (kW)"); axs[0].set_ylabel("Technical loss (kW)"); axs[0].legend(fontsize=5.8)
    order=["normal","low_source_voltage","poor_power_factor"]
    axs[1].boxplot([d.loc[d.technical_root_cause.eq(c),"min_voltage_pu"] for c in order],showfliers=False,patch_artist=True,boxprops={"facecolor":"#A7C9E8","linewidth":.6})
    axs[1].set_xticks(range(1,4),[SHORT[c] for c in order],rotation=30,ha="right"); axs[1].set_ylabel("Minimum voltage (p.u.)")
    order2=["normal","overload","topology_reconfiguration","reverse_power_flow"]
    ratios=[100*d.loc[d.technical_root_cause.eq(c),"technical_loss_kw"] / d.loc[d.technical_root_cause.eq(c),"actual_load_kw"] for c in order2]
    axs[2].boxplot(ratios,showfliers=False,patch_artist=True,boxprops={"facecolor":"#A9D2B0","linewidth":.6})
    axs[2].set_xticks(range(1,5),[SHORT[c] for c in order2],rotation=30,ha="right"); axs[2].set_ylabel("Technical loss / load (%)")
    for label,ax in zip("abc",axs): ax.text(-.16,1.06,label,transform=ax.transAxes,weight="bold",fontsize=9)
    fig.subplots_adjust(left=.085,right=.99,bottom=.25,top=.94,wspace=.38); save(fig,"fig_4_3_physical_validation")


def separated_figures(ds):
    """Export every evidence panel as an independent thesis figure."""
    # Fig. 4-2: three comparable donut charts rather than an uncommon stacked total.
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.35))
    for ax,(name,d) in zip(axs,ds.items()):
        vc=d.loss_type.value_counts(); labels=[x for x in COL if vc.get(x,0)>0]
        vals=[int(vc[x]) for x in labels]
        wedges,_=ax.pie(vals,startangle=90,counterclock=False,colors=[COL[x] for x in labels],
                        wedgeprops={"width":.34,"edgecolor":"white","linewidth":1})
        ax.text(0,0.08,f"{len(d):,}",ha="center",va="center",fontsize=10,weight="bold")
        ax.text(0,-.13,"samples",ha="center",va="center",fontsize=6,color="#666")
        ax.set_title({"technical":"Technical dataset","non_technical":"NTL dataset","mixed":"Mixed dataset"}[name],fontsize=8,weight="bold",pad=6)
        for w,label,value in zip(wedges,labels,vals):
            ang=np.deg2rad((w.theta1+w.theta2)/2); pct=100*value/sum(vals)
            if pct>=8: ax.text(.70*np.cos(ang),.70*np.sin(ang),f"{pct:.1f}%",ha="center",va="center",fontsize=5.5,color="white" if label!="normal" else "#333",weight="bold")
    handles=[mpl.patches.Patch(facecolor=COL[x],label=x.replace('_',' ')) for x in COL]
    fig.legend(handles=handles,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.01),fontsize=6.3)
    fig.subplots_adjust(left=.02,right=.98,bottom=.18,top=.88,wspace=.16); save(fig,"fig_4_2_dataset_composition_donuts")

    tech=ds["technical"]; base=tech.loc[tech.technical_root_cause.eq("normal"),"technical_loss_kw"].median()
    order=[x for x in TECH_ORDER if x in set(tech.technical_root_cause)]
    fig,ax=plt.subplots(figsize=(5.6,3.15)); ax.boxplot([tech.loc[tech.technical_root_cause.eq(c),"technical_loss_kw"] for c in order],showfliers=False,patch_artist=True,boxprops={"facecolor":"#8CB8DE","linewidth":.7},medianprops={"color":"#173F5F"})
    ax.axhline(base,ls="--",lw=.8,color="#777",label="Normal median"); ax.set_xticks(range(1,len(order)+1),[SHORT[c] for c in order],rotation=28,ha="right"); ax.set_ylabel("Technical loss (kW)"); ax.legend(fontsize=6)
    fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.95); save(fig,"fig_4_3_technical_loss_by_cause")

    ntl=ds["non_technical"]; ordern=[x for x in NTL_ORDER if x in set(ntl.ntl_root_cause)]
    fig,ax=plt.subplots(figsize=(5.6,3.15)); ax.boxplot([ntl.loc[ntl.ntl_root_cause.eq(c),"non_technical_loss_kw"] for c in ordern],showfliers=False,patch_artist=True,boxprops={"facecolor":"#F1BB81","linewidth":.7},medianprops={"color":"#8A4B12"})
    ax.set_xticks(range(1,len(ordern)+1),[SHORT[c] for c in ordern],rotation=28,ha="right"); ax.set_ylabel("Non-technical loss (kW)")
    fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.95); save(fig,"fig_4_4_ntl_loss_by_cause")

    m=ds["mixed"].query("technical_anomaly==1 and ntl_anomaly==1")
    tab=pd.crosstab(m.technical_root_cause,m.ntl_root_cause).reindex(index=TECH_ORDER[1:],columns=NTL_ORDER[1:],fill_value=0)
    fig,ax=plt.subplots(figsize=(5.8,3.8)); im=ax.imshow(tab,cmap="Purples",aspect="auto")
    ax.set_xticks(range(len(tab.columns)),[SHORT[c] for c in tab.columns],rotation=28,ha="right"); ax.set_yticks(range(len(tab.index)),[SHORT[c] for c in tab.index]); ax.set_xlabel("Non-technical root cause"); ax.set_ylabel("Technical root cause")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]): ax.text(j,i,str(tab.iat[i,j]),ha="center",va="center",fontsize=6.2,color="white" if tab.iat[i,j]>tab.values.max()*.55 else "#222")
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025); cb.set_label("Samples")
    fig.subplots_adjust(left=.22,right=.92,bottom=.22,top=.97); save(fig,"fig_4_5_mixed_cause_matrix")

    d=tech; rng=np.random.default_rng(12); sample=d.iloc[rng.choice(len(d),min(2200,len(d)),replace=False)]
    fig,ax=plt.subplots(figsize=(5.2,3.45))
    for cause,color in [("normal",COL["normal"]),("overload","#3478B8"),("line_resistance_increase","#B24A45")]:
        s=sample[sample.technical_root_cause.eq(cause)]; ax.scatter(s.actual_load_kw,s.technical_loss_kw,s=7,alpha=.38,color=color,label=SHORT[cause],edgecolors="none")
    ax.set_xlabel("Actual load (kW)"); ax.set_ylabel("Technical loss (kW)"); ax.legend(fontsize=6)
    fig.subplots_adjust(left=.14,right=.98,bottom=.16,top=.96); save(fig,"fig_4_6_load_loss_relationship")

    order=["normal","low_source_voltage","poor_power_factor"]
    fig,ax=plt.subplots(figsize=(4.6,3.25)); ax.boxplot([d.loc[d.technical_root_cause.eq(c),"min_voltage_pu"] for c in order],showfliers=False,patch_artist=True,boxprops={"facecolor":"#A7C9E8","linewidth":.7},medianprops={"color":"#E87522"})
    ax.set_xticks(range(1,4),[SHORT[c] for c in order]); ax.set_ylabel("Minimum voltage (p.u.)")
    fig.subplots_adjust(left=.16,right=.98,bottom=.15,top=.96); save(fig,"fig_4_7_voltage_response")

    order=["normal","overload","topology_reconfiguration","reverse_power_flow"]
    ratios=[100*d.loc[d.technical_root_cause.eq(c),"technical_loss_kw"]/d.loc[d.technical_root_cause.eq(c),"actual_load_kw"] for c in order]
    fig,ax=plt.subplots(figsize=(4.9,3.25)); ax.boxplot(ratios,showfliers=False,patch_artist=True,boxprops={"facecolor":"#A9D2B0","linewidth":.7},medianprops={"color":"#E87522"})
    ax.set_xticks(range(1,5),[SHORT[c] for c in order],rotation=20,ha="right"); ax.set_ylabel("Technical loss / load (%)")
    fig.subplots_adjust(left=.15,right=.98,bottom=.22,top=.96); save(fig,"fig_4_8_loss_ratio_response")


TECH_ORDER=["normal","overload","line_resistance_increase","low_source_voltage","poor_power_factor","reverse_power_flow","topology_reconfiguration"]
NTL_ORDER=["normal","uniform_scaling","meter_bypass","peak_period_theft","mean_replacement","intermittent_dropout","meter_bias"]
SHORT={"normal":"Normal","overload":"Overload","line_resistance_increase":"High R","low_source_voltage":"Low V","poor_power_factor":"Low PF","reverse_power_flow":"Reverse flow","topology_reconfiguration":"Topology","uniform_scaling":"Scaling","meter_bypass":"Bypass","peak_period_theft":"Peak theft","mean_replacement":"Mean repl.","intermittent_dropout":"Dropout","meter_bias":"Meter bias"}


def tables(ds):
    overview=[]; causes=[]
    for name,d in ds.items():
        overview.append({"dataset":name,"rows":len(d),"columns":len(d.columns),"train":(d.split=="train").sum(),"validation":(d.split=="validation").sum(),"test":(d.split=="test").sum(),"missing_cells":int(d.isna().sum().sum()),"max_balance_error_kw":d.energy_balance_residual_kw.abs().max()})
        for cause,g in d.groupby("root_cause"):
            causes.append({"dataset":name,"root_cause":cause,"n":len(g),"technical_loss_kw_mean":g.technical_loss_kw.mean(),"ntl_kw_mean":g.non_technical_loss_kw.mean(),"total_loss_kw_mean":g.total_loss_kw.mean(),"min_voltage_pu_mean":g.min_voltage_pu.mean(),"max_loading_pct_mean":g.max_line_loading_pct.mean()})
    pd.DataFrame(overview).to_csv(TABLES/"table_4_1_dataset_overview.csv",index=False)
    pd.DataFrame(causes).to_csv(TABLES/"table_4_3_cause_statistics.csv",index=False)
    params=pd.DataFrame([
        ["overload","load","P and Q multiplier","1.50–2.50"],["line_resistance_increase","line","R multiplier","1.50–3.00"],["low_source_voltage","external grid","voltage set point","0.92–0.97 p.u."],["poor_power_factor","load","power factor","0.68–0.82"],["reverse_power_flow","bus","DG active power","25–80% of system load"],["topology_reconfiguration","tie line","normally-open line","closed"],
        ["uniform_scaling","meter","reported/actual","0.20–0.80"],["meter_bypass","meter","reported/actual","0–0.18"],["peak_period_theft","meter","reported/actual at 17–22 h","0.25–0.60"],["mean_replacement","meter","daily mean multiplier","0.45–0.80"],["intermittent_dropout","meter","hourly activation probability","0.45"],["meter_bias","meter","reported/actual","0.72–0.94"],
    ],columns=["root_cause","target","parameter","range"])
    params.to_csv(TABLES/"table_4_2_intervention_parameters.csv",index=False)


if __name__ == "__main__":
    ds=load(); fig1_workflow(); fig2_composition(ds); fig3_physics(ds); separated_figures(ds); tables(ds)
    print(json.dumps({"figures":len(list(OUT.glob('*.svg'))),"tables":len(list(TABLES.glob('*.csv')))},ensure_ascii=False))

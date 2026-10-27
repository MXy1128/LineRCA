"""Project noisy meter readings to the physical energy-balance constraint."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"simulated_pandapower_v2"
FILES=["technical_line_loss.csv","non_technical_line_loss.csv","mixed_line_loss.csv"]

for name in FILES:
    path=DATA/name; d=pd.read_csv(path)
    meter=sorted(c for c in d if c.startswith("bus_") and c.endswith("_metered_load_kw"))
    values=d[meter].to_numpy(float); actual=d["actual_load_kw"].to_numpy(float)
    sums=values.sum(axis=1); factor=np.minimum(1.0,np.divide(actual,sums,out=np.ones_like(actual),where=sums>0))
    values*=factor[:,None]; d.loc[:,meter]=values
    d["metered_load_kw"]=values.sum(axis=1)
    d["non_technical_loss_kw"]=(d["actual_load_kw"]-d["metered_load_kw"]).clip(lower=0)
    d["total_loss_kw"]=d["technical_loss_kw"]+d["non_technical_loss_kw"]
    d["feeder_input_kw"]=d["actual_load_kw"]+d["technical_loss_kw"]
    d["energy_balance_residual_kw"]=d["feeder_input_kw"]-d["metered_load_kw"]-d["total_loss_kw"]
    d.to_csv(path,index=False,float_format="%.8f")

report=json.loads((DATA/"quality_report.json").read_text(encoding="utf-8"))
for name in FILES:
    d=pd.read_csv(DATA/name); key=name.replace("_line_loss.csv","")
    report["datasets"][key]["max_abs_energy_balance_residual_kw"]=float(d.energy_balance_residual_kw.abs().max())
(DATA/"quality_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

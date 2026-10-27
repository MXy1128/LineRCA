"""Build final table for the shared type-to-entity protocol."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];SRC=[("20260807_164905","metrics.csv"),("20260807_165350","metrics.csv"),("20260807_165742","metrics.csv"),("20260807_165929","metrics.csv")]
COLS=["Dataset","Model","Candidates","PR@1","PR@3","PR@5","PR@7","PR@10","MAP@3","MAP@5","MAP@7","MAP@10","MRR","NDCG@1","NDCG@3","NDCG@5","NDCG@7","NDCG@10"]
def main():
 frames=[]
 for run,file in SRC:
  p=ROOT/"results"/"fair_hierarchical_baselines"/run/file
  if p.exists():frames.append(pd.read_csv(p))
 out=pd.concat(frames,ignore_index=True);out=out[out.Status.eq("ok")].drop_duplicates(["Dataset","Model"],keep="last")[COLS].sort_values(["Dataset","PR@1"],ascending=[True,False]);counts=out.groupby("Dataset").Model.nunique().to_dict()
 if counts!={"mixed":14,"non_technical":14,"technical":14} or out[COLS[3:]].isna().any().any():raise RuntimeError(f"incomplete comparison: {counts}")
 dest=ROOT/"results"/"fair_hierarchical_comparison";dest.mkdir(parents=True,exist_ok=True);out.to_csv(dest/"all_models_all_datasets.csv",index=False);pct=out.copy()
 for c in COLS[3:]:pct[c]=pct[c].map(lambda x:f"{x*100:.2f}%")
 pct.to_csv(dest/"all_models_all_datasets_percent.csv",index=False);md="| "+" | ".join(pct.columns)+" |\n| "+" | ".join(["---"]*len(pct.columns))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in pct.itertuples(index=False,name=None));(dest/"all_models_all_datasets.md").write_text(md,encoding="utf-8");print(pct.to_string(index=False));print(dest)
if __name__=="__main__":main()

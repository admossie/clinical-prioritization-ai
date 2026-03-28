from __future__ import annotations
import argparse, pandas as pd

def simulate_workflow(df,prob_col="risk_score",target_col="target",capacity=500,intervention_rate=0.18,readmit_cost=12000.0,outreach_cost=120.0):
    ranked=df.sort_values(prob_col, ascending=False).copy(); queue=ranked.head(capacity)
    actual=float(queue[target_col].sum()); avoided=actual*intervention_rate; gross=avoided*readmit_cost; outreach=len(queue)*outreach_cost
    return {"capacity":int(capacity),"patients_reviewed":int(len(queue)),"actual_high_risk_in_queue":int(actual),"capture_rate_in_queue":float(actual/max(ranked[target_col].sum(),1)),"expected_avoided_readmissions":float(avoided),"gross_savings":float(gross),"outreach_spend":float(outreach),"net_savings":float(gross-outreach)}

def run_workflow_scenarios(df,capacities=(50,100,200,500)):
    return pd.DataFrame([simulate_workflow(df, capacity=c) for c in capacities])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scored-data", required=True); args=p.parse_args()
    df=pd.read_csv(args.scored_data); out=run_workflow_scenarios(df); out.to_csv("outputs/tables/workflow_scenarios.csv", index=False)
if __name__=="__main__": main()

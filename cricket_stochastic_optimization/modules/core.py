import numpy as np
import pandas as pd
import pulp

ROLES = ['Batsman','Bowler','All-Rounder','Wicketkeeper']
REQUIRED = ['Player ID','Player Name','Team','Player Role','Batting Average','Batting Strike Rate','Bowling Average','Bowling Economy','Total Runs','Total Wickets','Recent Match Performance','Batting Position','Player Availability','Performance Standard Deviation','Selection Category']

def validate(df):
    missing = set(REQUIRED)-set(df.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    df = df.copy()
    if df['Player ID'].duplicated().any(): raise ValueError('Player IDs must be unique')
    if not df['Player Role'].isin(ROLES).all(): raise ValueError('Invalid player role')
    for col in ['Batting Average','Batting Strike Rate','Bowling Average','Bowling Economy','Total Runs','Total Wickets','Recent Match Performance','Batting Position','Performance Standard Deviation']:
        df[col] = pd.to_numeric(df[col],errors='raise')
        if (df[col]<0).any(): raise ValueError(col+' must be nonnegative')
    df['Player Availability'] = df['Player Availability'].astype(str).str.lower().isin(['true','yes','1','available'])
    return df.reset_index(drop=True)

def simulate(df, n=1000, seed=42, variability=1.):
    rng = np.random.default_rng(seed)
    # Synthetic model: truncated-normal runs and Poisson wickets; not validated match forecasts.
    bat = df['Batting Average'].to_numpy(float)
    strike = df['Batting Strike Rate'].to_numpy(float)
    role = df['Player Role'].to_numpy()
    sd = df['Performance Standard Deviation'].to_numpy(float)*variability
    runs = np.maximum(0, rng.normal(bat[None,:], sd[None,:],(n,len(df))))
    runs = np.rint(np.minimum(runs,200))
    wk_lambda = np.where(np.isin(role,['Bowler','All-Rounder']),np.clip(25/np.maximum(df['Bowling Average'].to_numpy(float),10),.1,2.5),0.)
    wickets = np.minimum(rng.poisson(wk_lambda[None,:],size=(n,len(df))),5)
    # Illustrative utility units: run contribution, wicket bonus, strike rate and economy adjustments.
    economy = df['Bowling Economy'].to_numpy(float)
    score = runs*(.75+.25*np.minimum(strike,220)/140) + 18*wickets
    score += np.where(np.isin(role,['Bowler','All-Rounder']),np.maximum(0,10-economy)*2,0)[None,:]
    return score

def optimize(df, scenarios, risk=0.5, alpha=0.10, deterministic=False):
    n = len(df)
    if n<11 or df['Player Availability'].sum()<11: raise ValueError('Need at least 11 available players')
    prob = pulp.LpProblem('CricketXI',pulp.LpMaximize)
    x = pulp.LpVariable.dicts('select',range(n),cat='Binary')
    prob += pulp.lpSum(x[i] for i in range(n))==11
    roles = df['Player Role'].tolist()
    prob += pulp.lpSum(x[i] for i in range(n) if roles[i]=='Wicketkeeper')>=1
    prob += pulp.lpSum(x[i] for i in range(n) if roles[i]=='Bowler')>=3
    prob += pulp.lpSum(x[i] for i in range(n) if roles[i] in ['Batsman','Wicketkeeper','All-Rounder'])>=5
    for i,avail in enumerate(df['Player Availability']):
        if not avail: prob += x[i]==0
    means = scenarios.mean(axis=0)
    expected = pulp.lpSum(float(means[i])*x[i] for i in range(n))
    if deterministic or risk==0:
        prob += expected
    else:
        # Lower-tail CVaR of team performance: eta - E[(eta - team_score)+]/alpha.
        eta = pulp.LpVariable('eta',lowBound=None)
        short = pulp.LpVariable.dicts('shortfall',range(len(scenarios)),lowBound=0)
        for j,row in enumerate(scenarios):
            prob += short[j] >= eta-pulp.lpSum(float(row[i])*x[i] for i in range(n))
        cvar = eta-pulp.lpSum(short[j] for j in range(len(scenarios)))/(alpha*len(scenarios))
        prob += (1-risk)*expected+risk*cvar
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status]!='Optimal': raise ValueError('No feasible XI. Check role and availability constraints.')
    ids = [i for i in range(n) if pulp.value(x[i])>.5]
    return ids, float(pulp.value(prob.objective))

def metrics(scenarios,ids,target=450):
    totals = scenarios[:,ids].sum(axis=1)
    k = max(1,int(np.ceil(len(totals)*.1)))
    return {'Expected score':float(totals.mean()),'Standard deviation':float(totals.std()),'Bottom 10% mean':float(np.sort(totals)[:k].mean()),'Target probability':float(np.mean(totals>=target))},totals

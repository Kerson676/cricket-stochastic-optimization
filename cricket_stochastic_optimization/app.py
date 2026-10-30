from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.core import validate,simulate,optimize,metrics

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Cricket XI | Stochastic Lab',page_icon='🏏',layout='wide')
st.markdown('<style>.stApp{background:#081426;color:#f4f7ff}div[data-testid="stMetric"]{background:#142541;padding:16px;border-radius:12px;border:1px solid #24416c}h1,h2,h3{color:#e9bc60!important}</style>',unsafe_allow_html=True)
st.title('🏏 Cricket XI • Stochastic Decision Lab')
st.caption('Academic simulation | Synthetic cricket statistics, not real player forecasts')
up=st.sidebar.file_uploader('Upload compatible player CSV',type='csv')
try: df=validate(pd.read_csv(up if up else ROOT/'data/cricket_players.csv'))
except Exception as e: st.error(str(e));st.stop()
page=st.sidebar.radio('Navigate',['Dashboard','Player Analytics','Monte Carlo','Optimal XI','Team Comparison','Sensitivity Analysis','Results & Export'])
n=st.sidebar.select_slider('Training scenarios',options=[100,500,1000,5000,10000],value=500)
seed=st.sidebar.number_input('Random seed',0,1000000,42)
risk=st.sidebar.slider('Risk aversion λ',0.,1.,.5,.05)
var=st.sidebar.slider('Performance variability',.25,2.,1.,.05)
target=st.sidebar.number_input('Target team score',0,2000,450)
# Cap training scenarios for CVaR LP responsiveness; evaluation remains independent.
train_n=min(n,1000)
train=simulate(df,train_n,int(seed),var)
test=simulate(df,5000,int(seed)+101,var)

def solve_pair(lam=risk,train_data=train):
    d,_=optimize(df,train_data,deterministic=True)
    s,obj=optimize(df,train_data,risk=lam)
    return d,s,obj

def team_table(ids):
    return df.iloc[ids][['Player Name','Player Role','Team','Batting Average','Bowling Average','Player Availability']]

if page=='Dashboard':
    a,b,c,d=st.columns(4)
    a.metric('Players',len(df));b.metric('Available',int(df['Player Availability'].sum()));c.metric('Bowlers',int((df['Player Role']=='Bowler').sum()));d.metric('Wicketkeepers',int((df['Player Role']=='Wicketkeeper').sum()))
    st.plotly_chart(px.pie(df,names='Player Role',title='Squad Composition',hole=.5),use_container_width=True)
    st.info('The model maximizes expected simulated team utility and lower-tail CVaR under XI composition constraints. Utility is not literal cricket runs.')
elif page=='Player Analytics':
    role=st.multiselect('Roles',sorted(df['Player Role'].unique()),default=sorted(df['Player Role'].unique()))
    subset=df[df['Player Role'].isin(role)]
    st.dataframe(subset,use_container_width=True)
    metric=st.selectbox('Compare metric',['Batting Average','Batting Strike Rate','Bowling Economy','Performance Standard Deviation'])
    st.plotly_chart(px.bar(subset,x='Player Name',y=metric,color='Player Role'),use_container_width=True)
elif page=='Monte Carlo':
    player=st.selectbox('Player',df['Player Name'])
    idx=int(df.index[df['Player Name']==player][0]); samples=simulate(df,n,int(seed),var)[:,idx]
    a,b,c=st.columns(3);a.metric('Mean utility',f'{samples.mean():.1f}');b.metric('SD',f'{samples.std():.1f}');c.metric('90% interval',f'{np.quantile(samples,.05):.0f}–{np.quantile(samples,.95):.0f}')
    st.plotly_chart(px.histogram(x=samples,nbins=45,title='Simulated player utility'),use_container_width=True)
elif page in ['Optimal XI','Team Comparison','Results & Export']:
    try:
        with st.spinner('Solving selection models...'): d,s,obj=solve_pair()
    except Exception as e: st.error(str(e));st.stop()
    dm,dt=metrics(test,d,target);sm,stot=metrics(test,s,target)
    if page=='Optimal XI':
        st.subheader('Optimized Playing XI')
        st.dataframe(team_table(s),use_container_width=True)
        st.metric('Risk-adjusted objective',f'{obj:.1f}')
        st.write('Selected by a binary optimization model enforcing 11 players, a wicketkeeper, at least 3 specialist bowlers and 5 batting-capable players.')
    elif page=='Team Comparison':
        st.subheader('Out-of-sample comparison (5,000 identical scenarios)')
        st.dataframe(pd.DataFrame({'Deterministic':dm,'Stochastic':sm}),use_container_width=True)
        st.plotly_chart(px.histogram(pd.DataFrame({'Deterministic':dt,'Stochastic':stot}),barmode='overlay',title='Simulated team utility'),use_container_width=True)
        a,b=st.columns(2);a.write('Deterministic XI');a.dataframe(team_table(d));b.write('Stochastic XI');b.dataframe(team_table(s))
    else:
        st.dataframe(pd.DataFrame({'Deterministic':dm,'Stochastic':sm}),use_container_width=True)
        st.download_button('Download optimized XI',team_table(s).to_csv(index=False),'optimized_xi.csv','text/csv')
        st.download_button('Download comparison',pd.DataFrame({'Metric':list(dm),'Deterministic':list(dm.values()),'Stochastic':list(sm.values())}).to_csv(index=False),'comparison.csv','text/csv')
        st.download_button('Download evaluation scenarios',pd.DataFrame({'Deterministic':dt,'Stochastic':stot}).to_csv(index=False),'evaluation.csv','text/csv')
else:
    st.subheader('Risk sensitivity')
    rows=[]
    for lam in [0,.25,.5,.75,1.]:
        try:
            ids,obj=optimize(df,train[:min(250,len(train))],risk=lam)
            met,_=metrics(test,ids,target)
            rows.append({'λ':lam,'Expected score':met['Expected score'],'Bottom 10% mean':met['Bottom 10% mean'],'XI':', '.join(df.iloc[ids]['Player Name'])})
        except ValueError as e: st.error(str(e));break
    if rows:
        res=pd.DataFrame(rows);st.dataframe(res,use_container_width=True)
        st.plotly_chart(px.line(res,x='λ',y=['Expected score','Bottom 10% mean'],markers=True,title='Risk vs expected utility'),use_container_width=True)

"""Financial Risk Dashboard builder (17 tabs plus a 4-minute auto demo).
Runs the sales risk analysis (VaR/ES with confidence ranges, diagnostics, risk limits, risk contribution,
stress tests, scenarios, margin and discount, concentration, explorers, forecast, risk register,
data quality, presenter Q and A) and writes risk_dashboard.html. Requires: numpy pandas scipy.
Usage: python build_dashboard.py
"""
import numpy as np, pandas as pd, json
from scipy.stats import norm, skew, kurtosis, t as st
rng=np.random.default_rng(42); N=2_000_000
stores={100102:"Dairy Princess",100404:"Blue Elephant",100501:"Lucky Store",100701:"Sum Emperor",100903:"Burger King Star",100904:"Taste of Yummy Asia",101701:"Cafe Amazon Food",101702:"Dairy Yummy Queen",101801:"Emperor of China Dining",102301:"Park Cafe",102401:"Segafredo Dining Area",200102:"Apron Coffee",200301:"Burger Kingya",200902:"Dairy Queen Nice",201002:"Moonlight Cafe",201301:"Happy Plate",201901:"The Coffee Club",202101:"The Coffee Bean",202102:"Fresh Corner",301901:"Garden Kitchen",100301:"West Family",102101:"Golden House"}
P=[("FINALCOHOL",x) for x in [4.5,4.2,45,7.5,5]]+[("FINBEVERAG",x) for x in [2,2,2,1.5,3,3.5,2.5,3.8,4.5,4.8]]+[("FINFOOD",x) for x in [6.5,7.2,6.2,7.5,8,7.8,8.5,4.5,4,2.5,3,5.5,8.5,9.5,4.2]]+[("RETAIL",x) for x in [12,18,3.2,3.5,5.5,2.8,4.2,15,8]]
sc=np.array([100102,100301,100404,100501,100701,100903,100904,101701,101702,101801,102101,102301,102401,200102,200301,200902,201002,200301,201901,202101,202102,301901])
s=sc[rng.integers(0,len(sc),N)]; pi=rng.integers(0,len(P),N)
cat=np.array([p[0] for p in P])[pi]; price=np.array([p[1] for p in P])[pi]
q=rng.integers(1,6,N); gross=price*q; disc=rng.choice([0,0,0,0,.5,1,1.5],N); net=gross-disc
cost=price*.7*q; prof=net-cost
d=pd.DataFrame(dict(store=s,cat=cat,day=rng.integers(0,365,N),q=q,gross=gross,disc=disc,net=net,cost=cost,prof=prof))
tot=d[['net','cost','prof','q']].sum()
out={"n":N,"rev":tot.net,"cost":tot.cost,"profit":tot.prof,"margin":tot.prof/tot.net,"units":int(tot.q),"disc_rate":float((d.disc/d.gross).mean())}
d['m']=(pd.Timestamp('2022-01-01')+pd.to_timedelta(d.day,unit='D')).dt.month-1
mo=d.groupby('m')[['net','cost','prof']].sum(); out["monthly"]=[[round(a),round(b),round(c)] for a,b,c in mo.values]
cs=d.groupby('cat')[['net','prof','q']].sum().sort_values('net',ascending=False)
out["cats"]=[[k,r.net,r.prof,r.prof/r.net] for k,r in cs.iterrows()]
dy=d.groupby('day')[['net','cost','prof']].sum(); loss=-dy.prof.diff().dropna().values
out["daily"]=[round(x) for x in dy.prof.values]
out["loss"]=dict(mean=loss.mean(),sd=loss.std(ddof=1),max=loss.max(),skew=skew(loss),kurt=kurtosis(loss),n=len(loss))
v={}
for c in (.95,.99):
    h=np.quantile(loss,c); es=loss[loss>=h].mean()
    pv=loss.mean()+norm.ppf(c)*loss.std(ddof=1); pes=loss.mean()+loss.std(ddof=1)*norm.pdf(norm.ppf(c))/(1-c)
    nu,lo,sc_=st.fit(-loss); nu=max(nu,2.1); sim=-st.rvs(nu,loc=lo,scale=sc_,size=100000,random_state=rng)
    mv=np.quantile(sim,c); mes=sim[sim>=mv].mean()
    v[str(int(c*100))]=dict(hist=h,hes=es,par=pv,pes=pes,mc=mv,mes=mes,exc=int((loss>h).sum()))
out["var"]=v
out["hist"]=np.histogram(loss,bins=24)[0].tolist(); hb=np.histogram(loss,bins=24)[1]; out["histx"]=[round(x) for x in hb[:-1]]
g=lambda a,b:(d.net.sum()*(1+a)-d.cost.sum()*(1+b))-d.prof.sum()
out["stress"]={f"{c:.2f}":[round(g(sv,c)) for sv in [0,-.05,-.1,-.15,-.2]] for c in [0,.05,.1,.15,.2]}
sr=d.groupby('store').net.sum(); sh=sr/sr.sum()
out["hhi"]=dict(store=float((sh**2).sum()*1e4),cat=float(((cs.net/cs.net.sum())**2).sum()*1e4),top5=float(sh.nlargest(5).sum()),top=float(sh.max()),topname=stores[sh.idxmax()])
mq=d.groupby(['m','cat']).q.sum().unstack(); cv=(mq.std()/mq.mean()).sort_values(ascending=False); out["cv"]=[[k,v_] for k,v_ in cv.items()]
sd_=d.groupby(['store','day']).prof.sum().groupby('store').agg(lambda x:x.std()/x.mean())
g_=d.groupby('store').agg(net=('net','sum'),prof=('prof','sum'),disc=('disc','sum'),gross=('gross','sum'))
X=pd.DataFrame({"Low margin":1-g_.prof/g_.net,"Discount rate":g_.disc/g_.gross,"Profit volatility":sd_,"Revenue share":sh})
Z=(X-X.min())/(X.max()-X.min()+1e-12)+1e-9; Pp=Z/Z.sum(); E=-(Pp*np.log(Pp)).sum()/np.log(len(Z)); W=(1-E)/(1-E).sum()
score=(Z*W).sum(axis=1)*100 
out["weights"]=W.round(3).to_dict()
out["scores"]=sorted([[stores[k],round(float(v_),1),round(float(g_.net[k])),round(float(g_.prof[k]/g_.net[k])*100,2)] for k,v_ in score.items()],key=lambda r:-r[1])

from scipy.stats import chi2, jarque_bera
n=len(loss); jb=jarque_bera(loss); ac=[np.corrcoef(loss[:-k],loss[k:])[0,1] for k in range(1,11)]
Q=n*(n+2)*sum(a*a/(n-k) for k,a in enumerate(ac,1))
def kup(x,p):
    ph=x/n; ll=lambda q:(n-x)*np.log(1-q)+x*np.log(q) if 0<q<1 else 0
    lr=-2*(ll(p)-ll(ph)); return float(lr),float(chi2.sf(lr,1))
out["diag"]=dict(jb=float(jb.statistic),jbp=float(jb.pvalue),lb=float(Q),lbp=float(chi2.sf(Q,10)),ac1=float(ac[0]),k95=kup(out["var"]["95"]["exc"],.05),k99=kup(out["var"]["99"]["exc"],.01),med=float(np.median(loss)),mn=float(loss.min()),n=n)
ok=(d.net>0).values; dr=(d.disc/d.gross).values[ok]; mr=(d.prof/d.net).values[ok]
out["md"]=dict(r=float(np.corrcoef(dr,mr)[0,1]),n=N,buckets=[[float(k),float(g.prof.sum()/g.net.sum()),int(len(g))] for k,g in d.groupby('disc')])
pr=out["stress"]["0.10"][2]/out["profit"]; vr=out["var"]["95"]["hist"]/(out["profit"]/365)
R=lambda c,a,b:"High" if c>a else "Moderate" if c>b else "Low"
out["reg"]=[["Daily profit volatility",f"95% VaR is {vr:.1%} of average daily profit",R(vr,.10,.04),"Hold a buffer equal to the 99% VaR in weekly cash planning"],
["Cost inflation with weak sales",f"Costs +10% and sales -10% change profit by {pr:.1%}",R(-pr,.5,.25),"Review supplier contracts and set price-pass-through triggers"],
["Category concentration",f"HHI {out['hhi']['cat']:.0f}",R(out['hhi']['cat'],2500,1500),"Grow the lower-share categories and track category margin monthly"],
["Store concentration",f"Top 5 stores earn {out['hhi']['top5']:.0%} of revenue; HHI {out['hhi']['store']:.0f}",R(out['hhi']['store'],2500,1500),"Keep monitoring; no action needed at this level"],
["Discount leakage",f"Average discount {out['disc_rate']:.1%} of gross; correlation with margin {out['md']['r']:.2f}",R(out['disc_rate'],.08,.03),"Cap discounts by category and approve any above the cap"],
["Demand volatility",f"{out['cv'][0][0]} varies {out['cv'][0][1]:.1%} month to month",R(out['cv'][0][1],.25,.10),"Plan inventory for this category on a rolling three-month view"],
["Single-store dependence",f"{out['scores'][0][0]} scores {out['scores'][0][1]} of 100",R(out['scores'][0][1],50,25),"Set a contingency plan for this store"]]

def fc(y,h=3):
    y=np.array(y,float); x=np.arange(len(y)); b,a=np.polyfit(x,y,1); se=np.sqrt(((y-(a+b*x))**2).sum()/(len(y)-2)); sxx=((x-x.mean())**2).sum(); r=[]
    for k in range(1,h+1):
        xx=len(y)-1+k; m=a+b*xx; w=1.28*se*np.sqrt(1+1/len(y)+(xx-x.mean())**2/sxx); r.append([float(m),float(m-w),float(m+w)])
    return r
out["fc"]=dict(rev=fc([r[0] for r in out["monthly"]]),profit=fc([r[2] for r in out["monthly"]]))

out["disc_total"]=float(d.disc.sum()); out["acf"]=[float(a) for a in ac]
out["stores"]={stores[k]:[float(g_.net[k]),float(g_.prof[k]),float(g_.prof[k]/g_.net[k]),float(g_.disc[k]/g_.gross[k]),float(sd_[k]),float(sh[k]),float(score[k])] for k in g_.index}

s21=(-st.rvs(nu,loc=lo,scale=sc_,size=(100000,21),random_state=rng)).sum(axis=1)
out["mh"]={c:[float(np.quantile(s21,int(c)/100)),float(s21[s21>=np.quantile(s21,int(c)/100)].mean())] for c in ("95","99")}

pvt=d.groupby(['store','m']).prof.sum().unstack(fill_value=0); out["sm"]={stores[k]:[float(x) for x in r] for k,r in pvt.iterrows()}
dmx=d.groupby(['cat','disc'])[['prof','net']].sum(); out["dm"]={c_:[float(dmx.loc[(c_,x),'prof']/dmx.loc[(c_,x),'net']) for x in [0,.5,1,1.5]] for c_ in cs.index}
out["dq"]=[["Rows loaded",N],["Rows with a net sale of zero or less",int((d.net<=0).sum())],["Rows with a discount above the gross sale",int((d.disc>d.gross).sum())],["Rows with a quantity of zero or less",int((d.q<=0).sum())],["Rows with a negative cost",int((d.cost<0).sum())],["Rows with missing values",int(d.isna().sum().sum())]]

cn_=d.groupby(['cat','m']).net.sum().unstack(fill_value=0); cp_=d.groupby(['cat','m']).prof.sum().unstack(fill_value=0)
out["cm"]={c_:[[float(cn_.loc[c_,m_]),float(cp_.loc[c_,m_])] for m_ in range(12)] for c_ in cn_.index}

cd_=d.groupby(['day','cat']).prof.sum().unstack(fill_value=0).reindex(range(365),fill_value=0)
sd2=d.groupby(['day','store']).prof.sum().unstack(fill_value=0).reindex(range(365),fill_value=0)
Lc=-np.diff(cd_.values,axis=0); Ls=-np.diff(sd2.values,axis=0); Lt=-np.diff(cd_.values.sum(axis=1)); tail=Lt>=np.quantile(Lt,.95)
def comp(L,names,rs):
    es=L[tail].mean(axis=0); tot=es.sum(); return [[names[i],float(es[i]/tot),float(es[i]),float(np.quantile(L[:,i],.95)),float(rs[i])] for i in range(len(names))]
cn2=list(cd_.columns); sn2=[stores[k] for k in sd2.columns]
out["rcat"]=comp(Lc,cn2,(cs.net/cs.net.sum()).reindex(cn2).values); out["rst"]=comp(Ls,sn2,sh.reindex(sd2.columns).values)
pq=np.quantile(Lt,.95); out["div"]=dict(cat=float(sum(r[3] for r in out["rcat"])/pq),store=float(sum(r[3] for r in out["rst"])/pq))
out["corr"]=dict(names=cn2,m=np.corrcoef(Lc.T).tolist())
bt=rng.choice(Lt,size=(2000,len(Lt))); out["ci"]={c_:[float(np.quantile(np.quantile(bt,int(c_)/100,axis=1),q_)) for q_ in (.025,.975)] for c_ in ("95","99")}
out["worst"]=[[(pd.Timestamp('2022-01-01')+pd.Timedelta(days=int(k)+1)).strftime('%d %b'),float(Lt[k]),cn2[Lc[k].argmax()],float(Lc[k].max()),sn2[Ls[k].argmax()],float(Ls[k].max())] for k in np.argsort(-Lt)[:10]]

json.dump(out, open("data.json", "w"), default=float)  # numbers used by the dashboard (also used by the video script)

HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><title>Sales Risk Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap" rel="stylesheet">
<style>
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);--bg:#f3f5f7;--card:#fff;--tx:#14202e;--mu:#5d6b7c;--bd:#dde3ea;--ac:#1d6f8a;--ok:#2f8f6b;--wn:#d99a1f;--rk:#c0392b;--ink:#0e2233;--ink2:#9fb4c6;--hi:#ffffff}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b141d;--card:#122130;--tx:#e6edf4;--mu:#93a4b6;--bd:#223548;--ac:#5bb6d4;--ok:#4cc79a;--wn:#f0b84a;--rk:#ff6b5a;--ink:#071019}}
:root[data-theme="dark"]{--bg:#0b141d;--card:#122130;--tx:#e6edf4;--mu:#93a4b6;--bd:#223548;--ac:#5bb6d4;--ok:#4cc79a;--wn:#f0b84a;--rk:#ff6b5a;--ink:#071019}
html{scroll-padding-top:env(safe-area-inset-top,0px)}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 Figtree,system-ui,-apple-system,"Segoe UI",sans-serif}
.hero{background:var(--ink);color:var(--hi)}.hi{max-width:1120px;margin:auto;padding:34px 22px 26px;display:grid;gap:26px;grid-template-columns:minmax(0,5fr) minmax(0,7fr);align-items:end}
.br{color:var(--ink2);font-size:14px;margin:0 0 14px;font-weight:500}
h1{font:500 clamp(26px,3.6vw,40px)/1.15 Fraunces,Georgia,serif;margin:0 0 14px;letter-spacing:-.01em}h1 b{font-weight:700;color:var(--wn)}
.hi p{color:var(--ink2);margin:0 0 6px;max-width:46ch}.hi p strong{color:var(--hi)}
#hero{width:100%;height:auto}.hl{fill:var(--ink2);font-size:11px}
nav{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--bd)}
nav div{max-width:1120px;margin:auto;padding:0 22px;display:flex;gap:4px;overflow-x:auto}
nav button{background:none;border:0;border-bottom:3px solid transparent;color:var(--mu);padding:14px 14px 11px;font:500 15px Figtree,sans-serif;cursor:pointer;white-space:nowrap}
nav button.on{color:var(--tx);border-color:var(--ac)}nav button:focus-visible{outline:2px solid var(--ac);outline-offset:-2px}
main{max-width:1120px;margin:auto;padding:22px 22px 56px}section{display:none}section.on{display:block}
.g{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin-bottom:16px}
.c{background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:18px 20px;margin-bottom:16px}.g .c{margin:0;border-radius:6px 6px 6px 22px}
.k{color:var(--mu);font-size:13px}.v{font:700 28px/1.25 Fraunces,Georgia,serif;margin-top:2px}.s{font-size:13px;color:var(--mu)}
h2{font:700 18px Fraunces,Georgia,serif;margin:0 0 4px}.n{color:var(--mu);font-size:13.5px;margin:0 0 14px;max-width:70ch}
.w{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid var(--bd);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--mu);font-weight:500;font-size:13px}tr:last-child td{border-bottom:0}
.b{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:7px 0;font-size:14px}.b span:first-child{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.b div{height:12px;background:var(--bd);border-radius:6px;overflow:hidden}.b i{display:block;height:100%;border-radius:6px}.b em{font-style:normal;text-align:right;font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}.t{fill:var(--mu);font-size:11px}.gd{stroke:var(--bd)}
.lg{display:flex;gap:16px;font-size:13px;color:var(--mu);margin-bottom:6px}.lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12.5px;font-weight:700;color:#fff}.High{background:var(--rk)}.Moderate{background:var(--wn);color:#14202e}.Low{background:var(--ok)}
.kf{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.kf div{border-left:3px solid var(--ac);padding-left:14px}.kf b{display:block;font:700 16px Fraunces,Georgia,serif;margin-bottom:2px}.kf span{color:var(--mu);font-size:14px}
td.l{white-space:normal;text-align:left;min-width:220px}
footer{max-width:1120px;margin:0 auto;padding:0 22px 40px;color:var(--mu);font-size:13px}
.two{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.gl dt{font-weight:700;margin-top:12px}.gl dd{margin:2px 0 0;color:var(--mu);max-width:70ch}
.up{color:var(--ok)}.dn{color:var(--rk)}
@media print{nav,.hero svg{display:none}section{display:block!important;break-inside:avoid}body{background:#fff}.hero{background:#fff;color:#000}}
.sl{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin-bottom:6px}.sl label{display:flex;justify-content:space-between;font-weight:500;margin-bottom:6px}.sl output{font-variant-numeric:tabular-nums;color:var(--ac);font-weight:700}
input[type=range]{width:100%;accent-color:var(--ac)}select{font:500 15px Figtree,sans-serif;padding:9px 12px;border:1px solid var(--bd);border-radius:6px;background:var(--card);color:var(--tx);margin-bottom:14px;max-width:100%}
nav button.tg{margin:9px 0 9px auto;border:1px solid var(--bd);border-radius:14px;padding:3px 12px;font-size:13px}
.ul{margin:0;padding-left:18px}.ul li{margin:9px 0}
details{border-bottom:1px solid var(--bd);padding:14px 0}details:last-child{border:0}summary{cursor:pointer;font:700 16px Fraunces,Georgia,serif}details p{margin:8px 0 0;color:var(--mu);max-width:72ch}
@media(max-width:760px){.hi{grid-template-columns:1fr}.b{grid-template-columns:110px 1fr 56px}}
@media(prefers-reduced-motion:no-preference){.hero #hero rect{transform-origin:bottom;transform-box:fill-box;animation:gr .8s ease-out both}@keyframes gr{from{transform:scaleY(0)}}}

:root{--sh:0 1px 2px rgba(14,34,51,.06),0 14px 32px -18px rgba(14,34,51,.28)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--sh:0 14px 32px -20px rgba(0,0,0,.7)}}
:root[data-theme="dark"]{--sh:0 14px 32px -20px rgba(0,0,0,.7)}
body{background:radial-gradient(900px 500px at 100% 0,rgba(29,111,138,.08),transparent),var(--bg);background-attachment:fixed}
.hero{background:radial-gradient(900px 380px at 88% -20%,rgba(91,182,212,.32),transparent 70%),radial-gradient(600px 300px at 0 110%,rgba(240,184,74,.14),transparent 70%),linear-gradient(180deg,#0a1b2a,#0e2b40);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(90deg,rgba(255,255,255,.035) 0 1px,transparent 1px 64px);pointer-events:none}
.hi{position:relative;padding:46px 22px 30px;gap:34px}h1{font-size:clamp(30px,4.4vw,52px);line-height:1.08}.br{letter-spacing:.01em}
#hero{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.13);border-radius:18px;padding:16px 14px 8px;backdrop-filter:blur(6px)}
.rb{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid rgba(255,255,255,.16);margin-top:4px;padding-top:20px;gap:18px}
.rb div{border-left:2px solid var(--wn);padding-left:14px}.rb span{display:block;color:var(--ink2);font-size:13px}.rb b{font:700 28px/1.2 Fraunces,Georgia,serif;color:#fff}
nav{background:color-mix(in srgb,var(--bg) 86%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--bd)}
nav div{padding:10px 22px;gap:6px}nav button{border:1px solid transparent;border-radius:999px;padding:8px 16px;font-size:14.5px}
nav button:hover{background:var(--card)}nav button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) nav button.on{background:var(--ac);color:#071019;border-color:var(--ac)}}
:root[data-theme="dark"] nav button.on{background:var(--ac);color:#071019;border-color:var(--ac)}
nav button.tg{margin:0 0 0 auto;border:1px solid var(--bd);padding:6px 14px}
main{padding-top:30px}.sh{margin:0 0 22px}.sh h2{font:700 clamp(26px,3.4vw,34px)/1.15 Fraunces,Georgia,serif;margin:0 0 6px;letter-spacing:-.01em}.sh p{margin:0;color:var(--mu);font-size:16px;max-width:62ch}
.c{border-radius:16px;box-shadow:var(--sh);padding:22px 24px}.g .c{border-radius:16px;border-top:3px solid var(--ac);padding:18px 20px}.g .c:nth-child(3n+2){border-top-color:var(--wn)}.g .c:nth-child(3n){border-top-color:var(--ok)}
.v{font-size:30px;letter-spacing:-.01em}h2{font-size:19px}.c>h2{padding-bottom:2px}
.w{border-radius:10px}th{background:color-mix(in srgb,var(--bd) 35%,transparent);font-size:12.5px}tbody tr:hover td,table tr:hover td{background:color-mix(in srgb,var(--ac) 7%,transparent)}
.b div{height:14px}.b i{background-image:linear-gradient(90deg,rgba(255,255,255,.0),rgba(255,255,255,.22))}
.pill{padding:3px 12px}.kf div{border-left-width:4px;background:color-mix(in srgb,var(--ac) 5%,transparent);padding:12px 14px;border-radius:0 10px 10px 0}
footer{border-top:1px solid var(--bd);padding-top:22px;margin-top:10px}
nav{position:static}nav div{flex-wrap:wrap;overflow:visible;padding:14px 22px;gap:6px}nav button{padding:7px 14px;font-size:14px}
nav button .no{opacity:.5;margin-right:7px;font-variant-numeric:tabular-nums}nav button.on .no{opacity:.85}
main{padding-bottom:100px}
.pg{position:fixed;left:0;right:0;bottom:0;z-index:9;background:color-mix(in srgb,var(--card) 93%,transparent);backdrop-filter:blur(10px);border-top:1px solid var(--bd);padding:12px 22px calc(12px + env(safe-area-inset-bottom,0px))}
.pg b{position:absolute;top:-1px;left:0;height:3px;width:0;background:var(--ac);transition:width .3s}
.pg div{max-width:1120px;margin:auto;display:flex;align-items:center;gap:14px}.pg span{flex:1;text-align:center;color:var(--mu);font-size:14.5px}
.pg button{font:500 15px Figtree,sans-serif;border:1px solid var(--bd);background:var(--card);color:var(--tx);border-radius:999px;padding:8px 20px;cursor:pointer}.pg button:hover:not(:disabled){border-color:var(--ac)}.pg button:disabled{opacity:.35;cursor:default}
@media(max-width:760px){nav div{flex-wrap:nowrap;overflow-x:auto}.pg button{padding:8px 14px}}
@media print{.pg{display:none}}
.cap{position:fixed;left:50%;transform:translateX(-50%);bottom:calc(76px + env(safe-area-inset-bottom,0px));width:min(920px,calc(100% - 28px));z-index:10;background:var(--ink);color:#fff;border-radius:16px;padding:14px 20px 16px;box-shadow:0 18px 40px -16px rgba(0,0,0,.6);display:none}
.cap.on{display:block}.cap i{display:block;height:3px;background:var(--wn);width:0;border-radius:2px;margin-bottom:12px;transition:width .25s linear}.cap p{margin:0 0 12px;font-size:17px;line-height:1.55}
.cap div{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--ink2)}.cap div span{flex:1}.cap button{font:500 13px Figtree,sans-serif;border:1px solid rgba(255,255,255,.3);background:transparent;color:#fff;border-radius:999px;padding:5px 14px;cursor:pointer}
nav button#dm{background:var(--wn);color:#14202e;border-color:transparent;font-weight:700}nav button#tg{margin-left:6px}
@media print{.cap{display:none!important}}
</style></head><body>
<div class="hero"><div class="hi"><div><p class="br">F&amp;B and retail sales, 22 stores, 2M transactions</p><h1>Profit can drop by <b id="h95"></b> on 1 day in 20</h1><p id="hp"></p></div><svg id="hero" viewBox="0 0 640 250" role="img" aria-label="Distribution of daily profit losses with VaR markers"></svg><div class="rb" id="rb"></div></div></div>
<nav><div id="nav"></div></nav><main>
<section id="s0"><div class="g" id="kp"></div><div class="c"><h2>Key findings</h2><p class="n" id="ov"></p><div class="kf" id="kf"></div></div><div class="c"><h2>Revenue, cost and profit by month</h2><div class="lg"><span><i style="background:var(--ac)"></i>Revenue</span><span><i style="background:var(--wn)"></i>Cost</span><span><i style="background:var(--ok)"></i>Profit</span></div><svg id="ln" viewBox="0 0 700 250"></svg></div><div class="c"><h2>Revenue by category</h2><p class="n" id="cn"></p><div id="ct"></div></div></section>
<section id="sdq"><div class="c"><h2>Data quality scorecard</h2><p class="n">Checks run before any risk figure is calculated, so data problems are not mistaken for business risk. Totals in this dashboard include any flagged rows; the discount and margin correlation leaves out rows with a net sale of zero.</p><div class="w"><table id="dq"></table></div></div></section>
<section id="stf"><div class="c"><h2>Daily profit through the year</h2><p class="n">Daily profit with its 7-day average. Red dots mark days that fell past the 95% VaR.</p><svg id="dc" viewBox="0 0 700 250"></svg></div><div class="c"><h2>Three-month outlook</h2><p class="n">A straight-line trend fitted to the 12 monthly totals. The shaded band covers 80% of likely outcomes.</p><div class="lg"><span><i style="background:var(--ok)"></i>Actual profit</span><span><i style="background:var(--wn)"></i>Forecast</span></div><svg id="fcs" viewBox="0 0 700 250"></svg><div class="w" style="margin-top:12px"><table id="ft"></table></div></div><div class="c"><h2>Profit by day of week</h2><p class="n">Average daily profit for each weekday across the year.</p><div id="wd"></div></div><div class="c"><h2>Month by month</h2><div class="w"><table id="mt"></table></div></div></section>
<section id="sx0"><div class="g" id="xk"></div><div class="two"><div class="c"><h2>What is going well</h2><ul class="ul" id="xg"></ul></div><div class="c"><h2>What needs attention</h2><ul class="ul" id="xa"></ul></div></div><div class="c"><h2>Action plan</h2><p class="n">Ordered by rating. Timing is a suggestion based on the rating.</p><div class="w"><table id="xp"></table></div></div></section>
<section id="s1"><div class="g" id="vk"></div><div class="c"><h2>VaR explorer</h2><p class="n">Drag to change the confidence level and see how the loss threshold and the average tail loss respond.</p><div class="sl"><div><label for="cf">Confidence level <output id="oc"></output></label><input id="cf" type="range" min="90" max="99.5" step=".5" value="95"></div></div></div><div class="g" id="ek"></div>
<div class="c"><h2>Rolling 60-day VaR against actual losses</h2><p class="n">The line is the 95% VaR estimated from the previous 60 days. Red dots are days when the actual loss was larger.</p><svg id="rv" viewBox="0 0 700 250"></svg></div>
<div class="c"><h2>Over a 21-day horizon</h2><p class="n">Monte Carlo on 100,000 simulated 21-day paths (about one trading month). Square-root-of-time scaling of the one-day Normal VaR is shown for comparison.</p><div class="w"><table id="mh"></table></div></div>
<div class="c"><h2>How certain is the VaR?</h2><p class="n">With one year of data every VaR is an estimate. Resampling the daily losses 2,000 times gives a 95% range around each figure.</p><div class="w"><table id="ci"></table></div></div><div class="c"><h2>How the three models compare</h2><p class="n">Value at Risk (VaR) is the loss level that daily profit falls past on the stated share of days. Expected Shortfall (ES) is the average loss on those worst days.</p><div class="w"><table id="vt"></table></div></div></section>
<section id="sd"><div class="g" id="dk"></div><div class="c"><h2>Is the loss pattern normal, and does it repeat?</h2><p class="n">These tests decide how far the Normal-model VaR can be trusted and whether one day's loss predicts the next.</p><div class="w"><table id="dt"></table></div></div><div class="c"><h2>Does one day predict the next?</h2><p class="n">Correlation between the loss on one day and the loss some days later. Bars outside the shaded band are statistically meaningful.</p><svg id="ac" viewBox="0 0 700 230"></svg></div><div class="c"><h2>Backtest: did the VaR hold up?</h2><p class="n">Kupiec test compares actual days past VaR with the number the confidence level expects. A p-value above 0.05 means the model passes.</p><div class="w"><table id="kt"></table></div></div></section>
<section id="slm"><div class="c"><h2>Set a daily loss limit</h2><p class="n">Choose the largest one-day fall in profit the business is willing to absorb. The page shows how often the year would have broken that limit.</p><div class="sl"><div><label for="lm">Daily loss limit <output id="ol"></output></label><input id="lm" type="range" min="500" max="4000" step="100" value="1500"></div></div></div><div class="g" id="lk"></div><div class="c"><h2>Where the limit sits</h2><div id="lb"></div></div></section>
<section id="src"><div class="g" id="rk"></div><div class="two"><div class="c"><h2>Tail risk by category</h2><p class="n">Share of the total loss on the worst 5% of days that comes from each category.</p><div class="w"><table id="rc"></table></div></div><div class="c"><h2>Do categories move together?</h2><p class="n">Correlation of daily profit changes. Values near 1 mean losses arrive together, so there is little diversification.</p><div class="w"><table id="cr"></table></div></div></div><div class="c"><h2>Tail risk by store</h2><p class="n">A risk-to-size ratio above 1 means the store adds more tail risk than its share of revenue suggests.</p><div class="w"><table id="rs"></table></div></div><div class="c"><h2>The ten worst days</h2><p class="n">Each day with its biggest losing category and store.</p><div class="w"><table id="w10"></table></div></div></section>
<section id="s2"><div class="c"><h2>What happens to annual profit under stress</h2><p class="n">Each cell shows the change in profit if sales fall and costs rise by the amounts shown. Darker red means a bigger loss.</p><div class="w"><table id="st"></table></div></div></section>
<section id="ssim"><div class="c"><h2>Build your own scenario</h2><p class="n">Move the sliders to test a what-if. Volume effects of price or discount changes are not modelled, so read results as a first estimate.</p><div class="sl"><div><label for="a1">Sales change <output id="o1"></output></label><input id="a1" type="range" min="-30" max="20" value="0"></div><div><label for="a2">Cost change <output id="o2"></output></label><input id="a2" type="range" min="-10" max="30" value="0"></div><div><label for="a3">Discount spend cut <output id="o3"></output></label><input id="a3" type="range" min="0" max="100" value="0"></div></div></div><div class="g" id="sk"></div><div class="g" id="be"></div></section>
<section id="sm"><div class="g" id="mk"></div><div class="c"><h2>Margin by discount level</h2><p class="n">Each discount step removes margin from the same sale. Zero-discount sales keep the full margin.</p><div id="mb"></div></div><div class="c"><h2>Margin by category and discount</h2><p class="n">Redder cells lose more margin to discounting.</p><div class="w"><table id="dmt"></table></div></div></section>
<section id="s3"><div class="g" id="hk"></div><div class="c"><h2>Category explorer</h2><p class="n">Pick a category to see its year in detail.</p><select id="cs" aria-label="Category"></select><div class="g" id="ck"></div><h2 style="margin-top:18px">Monthly profit</h2><svg id="cc" viewBox="0 0 700 215"></svg></div><div class="c"><h2>Category detail</h2><div class="w"><table id="cd"></table></div></div><div class="c"><h2>Demand volatility by category</h2><p class="n">Variation in monthly units relative to the average. A higher bar means less predictable demand.</p><div id="cv"></div></div></section>
<section id="s4"><div class="c"><h2>Store explorer</h2><p class="n">Pick a store to see how it compares with the chain.</p><select id="ss" aria-label="Store"></select><div class="g" id="sx"></div></div><div class="c"><h2>Compare two stores</h2><p class="n">Choose any two stores to see them side by side.</p><select id="sa" aria-label="First store"></select> <select id="sb" aria-label="Second store"></select><div class="w"><table id="cmp"></table></div></div><div class="c"><h2>Store risk score</h2><p class="n" id="wt"></p><div id="sr"></div></div><div class="c"><h2>Risk drivers by store</h2><p class="n">Darker red means that store is higher on that driver compared with the other stores.</p><div class="w"><table id="hm"></table></div></div></section>
<section id="stp"><div class="c"><h2>Monthly profit by store</h2><p class="n">Darker blue means higher profit that month. Cells show thousands of dollars.</p><div class="w"><table id="sp"></table></div></div></section>
<section id="sr2"><div class="c"><h2>Risk register</h2><p class="n">Every risk measured in this report, rated from the data, with a suggested management action.</p><div class="w"><table id="rt"></table></div></div></section>
<section id="sme"><div class="two"><div class="c"><h2>How the numbers are built</h2><dl class="gl"><dt>Daily loss</dt><dd>Each transaction's profit is net sale minus cost. Profit is added up by day, and the loss is the fall in profit from the day before.</dd><dt>Historical VaR</dt><dd>The loss level that daily losses stay under on 95% or 99% of days, read directly from the data.</dd><dt>Parametric VaR</dt><dd>The same level estimated from the average and spread of losses, assuming a Normal curve.</dd><dt>Monte Carlo VaR</dt><dd>100,000 daily losses simulated from a fitted Student-t curve, which allows heavier tails than Normal.</dd><dt>Expected Shortfall</dt><dd>The average loss on the days that fall past VaR.</dd></dl></div><div class="c"><h2>Reading the other measures</h2><dl class="gl"><dt>Concentration index (HHI)</dt><dd>Sum of squared revenue shares times 10,000. Below 1,500 is low, 1,500 to 2,500 moderate, above 2,500 high.</dd><dt>Demand volatility</dt><dd>Standard deviation of monthly units divided by their average.</dd><dt>Store risk score</dt><dd>Four indicators (low margin, discount rate, profit volatility, revenue share) scaled 0 to 1 and weighted by entropy, so indicators that differ more between stores count more.</dd><dt>Stress test</dt><dd>Stressed profit equals sales times (1 + sales shock) minus cost times (1 + cost shock).</dd><dt>Data</dt><dd>2,000,000 generated transactions across 22 stores for 2022. Results change if you load real sales data.</dd></dl></div></div></section>
<section id="sqa"><div class="c" id="qa"></div></section>
</main><footer>Financial risk model for F&amp;B and retail sales. VaR describes a loss threshold under stated assumptions; it is not a forecast of the exact loss.</footer><div class="cap" id="cap" role="status" aria-live="polite"><i id="cpb"></i><p id="cpt"></p><div><span id="cpm"></span><button id="cpp" type="button">Pause</button><button id="cps" type="button">Stop</button></div></div><div class="pg"><b id="pb"></b><div><button id="pp" type="button">Previous</button><span id="pl"></span><button id="pn" type="button">Next</button></div></div><script>
const D=__DATA__,$=i=>document.getElementById(i),M=x=>(x<0?"-":"")+"$"+Math.abs(x/1e6).toFixed(2)+"M",F=x=>(x<0?"-":"")+"$"+Math.abs(Math.round(x)).toLocaleString(),P=x=>(x*100).toFixed(1)+"%",v=D.var;
const tabs=["Overview", "Data quality", "Trends and forecast", "Executive summary", "Value at Risk", "Diagnostics", "Risk limits", "Risk contribution", "Stress test", "Scenario simulator", "Margin and discount", "Concentration", "Store risk", "Store performance", "Risk register", "Method", "Presenter Q and A"];
$("nav").innerHTML=tabs.map((t,i)=>`<button data-i="${i}"><span class="no">${i+1}</span>${t}</button>`).join("");
let cur=0;const show=i=>{cur=i;window.scrollTo({top:0});if(window.pg)pg(i);document.querySelectorAll("section").forEach((s,j)=>s.classList.toggle("on",i==j));document.querySelectorAll("nav button").forEach((b,j)=>b.classList.toggle("on",i==j))};
$("nav").onclick=e=>{const b=e.target.closest("button[data-i]");if(b)show(+b.dataset.i)};show(0);
const kp=a=>a.map(([k,x,s])=>`<div class="c"><div class="k">${k}</div><div class="v">${x}</div><div class="s">${s||""}</div></div>`).join("");
$("h95").textContent=F(v[95].hist);
$("hp").innerHTML=`On the worst 1 day in 100 the drop reaches <strong>${F(v[99].hist)}</strong>, and the average of the worst 5% of days is <strong>${F(v[95].hes)}</strong>. The worst day on record lost <strong>${F(D.loss.max)}</strong>.`;
(function(){const h=D.hist,n=h.length,w=D.histx[1]-D.histx[0],x0=D.histx[0],L=46,R=620,W=R-L,T=40,B=204,mx=Math.max(...h),X=a=>L+(a-x0)/(w*n)*W,Y=c=>B-c/mx*(B-T),bw=W/n,$$=t=>(t<0?"-":"")+"$"+Math.abs(t).toLocaleString();
const gr=(id,a,b)=>`<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient>`;
let s=`<defs>${gr("g1","#8fdcf2","#2a7b9a")}${gr("g2","#ffd77a","#c98a14")}${gr("g3","#ff9a8c","#c0392b")}<filter id="gl" x="-50%" y="-5%" width="200%" height="110%"><feGaussianBlur stdDeviation="3"/></filter></defs>`;
[0,.25,.5,.75,1].forEach(f=>{const y=Y(mx*f);s+=`<line x1="${L}" x2="${R}" y1="${y}" y2="${y}" stroke="#fff" opacity="${f?.08:.22}"/><text x="${L-8}" y="${y+4}" text-anchor="end" class="hl">${Math.round(mx*f)}</text>`});
const x95=X(v[95].hist);s+=`<rect x="${x95}" y="34" width="${R-x95}" height="${B-34}" fill="#f0b84a" opacity=".09"/>`;
h.forEach((c,i)=>{const e=D.histx[i],g=e>=v[99].hist?"g3":e>=v[95].hist?"g2":"g1";s+=`<rect x="${L+i*bw+1.5}" y="${Y(c)}" width="${bw-3}" height="${B-Y(c)}" rx="3" fill="url(#${g})"/>`});
const pts=h.map((c,i)=>[L+i*bw+bw/2,Y(((h[i-1]??c)+2*c+(h[i+1]??c))/4)]);let d="M"+pts[0];
for(let i=0;i<pts.length-1;i++){const a=pts[i-1]||pts[i],b=pts[i],c=pts[i+1],e=pts[i+2]||c;d+=` C${b[0]+(c[0]-a[0])/6},${b[1]+(c[1]-a[1])/6} ${c[0]-(e[0]-b[0])/6},${c[1]-(e[1]-b[1])/6} ${c[0]},${c[1]}`}
s+=`<path d="${d}" fill="none" stroke="#fff" stroke-width="2" opacity=".85"/><line x1="${X(0)}" x2="${X(0)}" y1="${T}" y2="${B}" stroke="#fff" stroke-dasharray="2 4" opacity=".4"/>`;
[[95,"#f0b84a","end",-8],[99,"#ff6b5a","start",8]].forEach(([c,col,an,dx])=>{const x=X(v[c].hist);s+=`<line x1="${x}" x2="${x}" y1="34" y2="${B}" stroke="${col}" stroke-width="5" opacity=".35" filter="url(#gl)"/><line x1="${x}" x2="${x}" y1="34" y2="${B}" stroke="${col}" stroke-width="1.6" stroke-dasharray="4 3"/><circle cx="${x}" cy="34" r="3.5" fill="${col}"/><text x="${x+dx}" y="14" text-anchor="${an}" class="hl" style="fill:${col}">${c}% VaR</text><text x="${x+dx}" y="30" text-anchor="${an}" style="fill:#fff;font-size:14px;font-weight:700">${$$(Math.round(v[c].hist))}</text>`});
for(let t=Math.ceil(x0/1000)*1000;t<x0+w*n;t+=1000)s+=`<text x="${X(t)}" y="${B+16}" text-anchor="middle" class="hl">${t?$$(t):"No change"}</text>`;
s+=`<line x1="${L}" x2="${R}" y1="${B}" y2="${B}" stroke="#fff" opacity=".35"/><text x="${(L+R)/2}" y="${B+36}" text-anchor="middle" class="hl">Daily change in profit: gains to the left, losses to the right. Bars count days.</text>`;$("hero").innerHTML=s})();
$("kp").innerHTML=kp([["Revenue",M(D.rev),"Net sales after discounts"],["Profit",M(D.profit),"After cost of goods"],["Profit margin",P(D.margin),"Profit divided by revenue"],["Units sold",D.units.toLocaleString(),"Across all categories"],["Average discount",P(D.disc_rate),"Share of gross sales"]]);
(function(){const m=D.monthly,mx=Math.max(...m.map(r=>r[0]))*1.05,L=50,W=630,X=i=>L+i*W/(m.length-1),Y=a=>220-a/mx*200,cl=["var(--ac)","var(--wn)","var(--ok)"],mn="JFMAMJJASOND";let s="";
[0,.25,.5,.75,1].forEach(f=>s+=`<line x1="${L}" x2="${L+W}" y1="${Y(mx*f)}" y2="${Y(mx*f)}" class="gd"/><text x="0" y="${Y(mx*f)+4}" class="t">${M(mx*f)}</text>`);
s+=`<defs><linearGradient id="gp" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--ok)" stop-opacity=".4"/><stop offset="1" stop-color="var(--ok)" stop-opacity="0"/></linearGradient></defs><polygon points="${X(0)},220 ${m.map((r,i)=>X(i)+","+Y(r[2])).join(" ")} ${X(m.length-1)},220" fill="url(#gp)"/>`;
[0,1,2].forEach(k=>s+=`<polyline fill="none" stroke="${cl[k]}" stroke-width="2.6" stroke-linejoin="round" points="${m.map((r,i)=>X(i)+","+Y(r[k])).join(" ")}"/>`);
m.forEach((r,i)=>s+=`<text x="${X(i)}" y="240" text-anchor="middle" class="t">${mn[i%12]}</text>`);$("ln").innerHTML=s})();
const bars=(rows,mx,fmt,c)=>rows.map(([n,x])=>`<div class="b"><span>${n}</span><div><i style="width:${x/mx*100}%;background:${c}"></i></div><em>${fmt(x)}</em></div>`).join("");
$("ct").innerHTML=bars(D.cats.map(c=>[c[0],c[1]]),D.cats[0][1],M,"var(--ac)");$("cn").textContent=`Category concentration index is ${Math.round(D.hhi.cat)}, above the 2,500 level usually read as high.`;
$("vk").innerHTML=kp([["95% VaR",F(v[95].hist),"Historical"],["99% VaR",F(v[99].hist),"Historical"],["95% Expected Shortfall",F(v[95].hes),"Average of the worst 5%"],["Days past 95% VaR",v[95].exc,"Of "+D.loss.n+" observed days"]]);
$("vt").innerHTML="<tr><th>Model</th><th>95% VaR</th><th>95% ES</th><th>99% VaR</th><th>99% ES</th></tr>"+[["Historical","hist","hes"],["Parametric (Normal)","par","pes"],["Monte Carlo (Student-t)","mc","mes"]].map(([n,a,b])=>`<tr><td>${n}</td><td>${F(v[95][a])}</td><td>${F(v[95][b])}</td><td>${F(v[99][a])}</td><td>${F(v[99][b])}</td></tr>`).join("")+`<tr><td>Backtest exceptions</td><td>${v[95].exc}</td><td></td><td>${v[99].exc}</td><td></td></tr>`;
const ss=["No change","-5%","-10%","-15%","-20%"],all=Object.values(D.stress).flat(),mn=Math.min(...all);
$("st").innerHTML="<tr><th>Cost increase</th>"+ss.map((s,i)=>`<th>${i?"Sales "+s:"Sales flat"}</th>`).join("")+"</tr>"+Object.entries(D.stress).map(([c,r])=>`<tr><td>${c==0?"No change":"Costs +"+Math.round(c*100)+"%"}</td>${r.map(x=>`<td style="background:rgba(192,57,43,${(Math.min(x,0)/mn*.55).toFixed(2)})">${x?F(x):"-"}</td>`).join("")}</tr>`).join("");
$("hk").innerHTML=kp([["Store concentration",Math.round(D.hhi.store),"Low. Revenue is spread across stores"],["Category concentration",Math.round(D.hhi.cat),"High. Few categories dominate"],["Top 5 stores",P(D.hhi.top5),"Share of revenue"],["Largest store",P(D.hhi.top),D.hhi.topname]]);
$("cv").innerHTML=bars(D.cv,D.cv[0][1],P,"var(--wn)");
$("wt").textContent="Weights come from the data (entropy method): "+Object.entries(D.weights).map(([k,x])=>k.toLowerCase()+" "+P(x)).join(", ")+". Scores run from 0 (lowest risk) to 100.";
$("sr").innerHTML=D.scores.map(r=>`<div class="b" style="grid-template-columns:170px 1fr 90px"><span>${r[0]}</span><div><i style="width:${r[1]}%;background:${r[1]>50?"var(--rk)":r[1]>15?"var(--wn)":"var(--ok)"}"></i></div><em>${r[1]} &nbsp;<span class="s">${M(r[2])}</span></em></div>`).join("");

const G=D.diag,pv=p=>p<.001?"< 0.001":p.toFixed(3);
$("kf").innerHTML=[["Daily swings are the main exposure",`Profit falls by more than ${F(v[95].hist)} on 1 day in 20, and ${F(v[99].hist)} on 1 day in 100.`],["Discounts cost the most margin",`Margin drops from ${P(D.md.buckets[0][1])} with no discount to ${P(D.md.buckets[3][1])} at the deepest discount.`],["Categories are concentrated",`The category index is ${Math.round(D.hhi.cat)}, while stores are well spread at ${Math.round(D.hhi.store)}.`]].map(([a,b])=>`<div><b>${a}</b><span>${b}</span></div>`).join("");
$("dk").innerHTML=kp([["Skewness",D.loss.skew.toFixed(2),"Lean of the loss distribution"],["Excess kurtosis",D.loss.kurt.toFixed(2),"Tail weight versus Normal"],["Median daily loss",F(G.med),"Typical day"],["Largest one-day gain",F(-G.mn),"Best day on record"]]);
const ok=(p,y,x)=>p>.05?y:x;
$("dt").innerHTML="<tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Reading</th></tr>"+[["Jarque-Bera (normality)",G.jb.toFixed(2),G.jbp,ok(G.jbp,"Close to Normal, so Parametric VaR is reasonable","Not Normal, so prefer Historical or Monte Carlo VaR")],["Ljung-Box, 10 lags (independence)",G.lb.toFixed(1),G.lbp,ok(G.lbp,"Losses look independent day to day","Losses are linked from day to day; a bad day tends to be followed by a recovery (lag-1 correlation "+G.ac1.toFixed(2)+")")]].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${pv(r[2])}</td><td class="l">${r[3]}</td></tr>`).join("");
$("kt").innerHTML="<tr><th>Confidence</th><th>Expected days</th><th>Actual days</th><th>Test statistic</th><th>p-value</th><th>Result</th></tr>"+[[95,.05,G.k95],[99,.01,G.k99]].map(([c,p,k])=>`<tr><td>${c}%</td><td>${(p*G.n).toFixed(1)}</td><td>${v[c].exc}</td><td>${k[0].toFixed(2)}</td><td>${pv(k[1])}</td><td><span class="pill ${k[1]>.05?"Low":"High"}">${k[1]>.05?"Pass":"Fail"}</span></td></tr>`).join("");
$("mk").innerHTML=kp([["Correlation, discount and margin",D.md.r.toFixed(2),"Strong negative link"],["Sales with a discount",P(1-D.md.buckets[0][2]/D.md.n),"Share of transactions"],["Margin without discount",P(D.md.buckets[0][1]),"Full-price sales"],["Margin at deepest discount",P(D.md.buckets[3][1]),"Discount of 1.50"]]);
$("mb").innerHTML=bars(D.md.buckets.map(b=>[b[0]?"Discount "+b[0].toFixed(2):"No discount",b[1]]),D.md.buckets[0][1],P,"var(--ok)");
$("rt").innerHTML="<tr><th>Risk</th><th>What we measured</th><th>Rating</th><th>Suggested action</th></tr>"+D.reg.map(r=>`<tr><td class="l"><b>${r[0]}</b></td><td class="l">${r[1]}</td><td><span class="pill ${r[2]}">${r[2]}</span></td><td class="l">${r[3]}</td></tr>`).join("");

const Hs=D.daily,MN=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
(function(){const L=50,W=640,mx=Math.max(...Hs),mi=Math.min(...Hs),X=i=>L+i*W/(Hs.length-1),Y=a=>220-(a-mi)/(mx-mi)*200;let s="";
[0,.5,1].forEach(f=>{const a=mi+(mx-mi)*f;s+=`<line x1="${L}" x2="${L+W}" y1="${Y(a)}" y2="${Y(a)}" class="gd"/><text x="0" y="${Y(a)+4}" class="t">${F(a)}</text>`});
s+=`<polyline fill="none" stroke="var(--ac)" opacity=".45" stroke-width="1" points="${Hs.map((a,i)=>X(i)+","+Y(a)).join(" ")}"/>`;
const ma=Hs.map((_,i)=>{const w=Hs.slice(Math.max(0,i-6),i+1);return w.reduce((a,b)=>a+b)/w.length});
s+=`<polyline fill="none" stroke="var(--ac)" stroke-width="2.6" points="${ma.map((a,i)=>X(i)+","+Y(a)).join(" ")}"/>`;
Hs.forEach((a,i)=>{if(i&&Hs[i-1]-a>v[95].hist)s+=`<circle cx="${X(i)}" cy="${Y(a)}" r="3.5" fill="var(--rk)"/>`});
MN.forEach((m,i)=>s+=`<text x="${L+(i+.5)*W/12}" y="240" text-anchor="middle" class="t">${m}</text>`);$("dc").innerHTML=s})();
(function(){const a=D.monthly.map(r=>r[2]),f=D.fc.profit,all=[...a,...f.map(r=>r[2])],lo=Math.min(...a,...f.map(r=>r[1]))*.97,hi=Math.max(...all)*1.03,L=60,W=630,n=15,X=i=>L+i*W/(n-1),Y=z=>220-(z-lo)/(hi-lo)*200;let s="";
[0,.5,1].forEach(q=>{const z=lo+(hi-lo)*q;s+=`<line x1="${L}" x2="${L+W}" y1="${Y(z)}" y2="${Y(z)}" class="gd"/><text x="0" y="${Y(z)+4}" class="t">${M(z)}</text>`});
s+=`<polygon fill="var(--wn)" opacity=".2" points="${X(11)},${Y(a[11])} ${f.map((r,i)=>X(12+i)+","+Y(r[2])).join(" ")} ${f.map((r,i)=>X(14-i)+","+Y(f[2-i][1])).join(" ")}"/>`;
s+=`<polyline fill="none" stroke="var(--ok)" stroke-width="2.8" points="${a.map((z,i)=>X(i)+","+Y(z)).join(" ")}"/><polyline fill="none" stroke="var(--wn)" stroke-width="2.8" stroke-dasharray="6 4" points="${X(11)},${Y(a[11])} ${f.map((r,i)=>X(12+i)+","+Y(r[0])).join(" ")}"/>`;
for(let i=0;i<n;i++)s+=`<text x="${X(i)}" y="240" text-anchor="middle" class="t">${MN[i%12]}</text>`;$("fcs").innerHTML=s})();
$("ft").innerHTML="<tr><th>Month</th><th>Revenue</th><th>Profit</th><th>Profit range (80%)</th></tr>"+[0,1,2].map(i=>`<tr><td>${MN[i]} (forecast)</td><td>${M(D.fc.rev[i][0])}</td><td>${M(D.fc.profit[i][0])}</td><td>${M(D.fc.profit[i][1])} to ${M(D.fc.profit[i][2])}</td></tr>`).join("");
$("mt").innerHTML="<tr><th>Month</th><th>Revenue</th><th>Cost</th><th>Profit</th><th>Margin</th><th>Profit vs prior month</th></tr>"+D.monthly.map((r,i)=>{const c=i?r[2]/D.monthly[i-1][2]-1:null;return `<tr><td>${MN[i]}</td><td>${M(r[0])}</td><td>${M(r[1])}</td><td>${M(r[2])}</td><td>${P(r[2]/r[0])}</td><td class="${c>0?"up":"dn"}">${c==null?"-":(c>0?"+":"")+P(c)}</td></tr>`}).join("");
const cn=k=>D.reg.filter(r=>r[2]==k).length;$("ov").innerHTML=`Overall: <span class="pill High">${cn("High")} high</span> <span class="pill Moderate">${cn("Moderate")} moderate</span> <span class="pill Low">${cn("Low")} low</span> across the seven risks in the register.`;

const SC=D.stores,sn=Object.keys(SC),avg=i=>sn.reduce((a,k)=>a+SC[k][i],0)/sn.length;
(function(){const a=D.acf,L=50,W=630,bw=W/a.length,b=1.96/Math.sqrt(D.diag.n),Y=z=>115-z*200;let s=`<rect x="${L}" y="${Y(b)}" width="${W}" height="${Y(-b)-Y(b)}" fill="var(--ac)" opacity=".12"/><line x1="${L}" x2="${L+W}" y1="115" y2="115" class="gd"/>`;
a.forEach((z,i)=>{s+=`<rect x="${L+i*bw+8}" y="${Math.min(Y(0),Y(z))}" width="${bw-16}" height="${Math.abs(Y(z)-Y(0))}" rx="2" fill="${Math.abs(z)>b?"var(--rk)":"var(--ac)"}"/><text x="${L+i*bw+bw/2}" y="222" text-anchor="middle" class="t">Lag ${i+1}</text>`});
[-.5,0,.5].forEach(z=>s+=`<text x="0" y="${Y(z)+4}" class="t">${z}</text>`);$("ac").innerHTML=s})();
function sim(){const s=+$("a1").value/100,c=+$("a2").value/100,d=+$("a3").value/100;$("o1").textContent=(s>0?"+":"")+(s*100)+"%";$("o2").textContent=(c>0?"+":"")+(c*100)+"%";$("o3").textContent=(d*100)+"%";
const R=D.rev*(1+s)+d*D.disc_total,C=D.cost*(1+c),Pf=R-C,dl=Pf-D.profit;
$("sk").innerHTML=kp([["Revenue",M(R),(R>=D.rev?"+":"")+M(R-D.rev)+" vs base"],["Cost",M(C),(C>=D.cost?"+":"")+M(C-D.cost)+" vs base"],["Profit",M(Pf),(dl>=0?"+":"")+M(dl)+" vs base"],["Margin",P(Pf/R),(((Pf/R)-D.margin)*100).toFixed(1)+" points vs base"]])}
["a1","a2","a3"].forEach(i=>$(i).oninput=sim);sim();
$("be").innerHTML=kp([["Cost rise that erases profit","+"+P(D.profit/D.cost),"With sales unchanged"],["Sales fall that erases profit","-"+P(D.profit/D.rev),"With costs unchanged"],["Discount spend",M(D.disc_total),"Total given away in the year"]]);
$("cd").innerHTML="<tr><th>Category</th><th>Revenue</th><th>Share</th><th>Profit</th><th>Margin</th></tr>"+D.cats.map(r=>`<tr><td>${r[0]}</td><td>${M(r[1])}</td><td>${P(r[1]/D.rev)}</td><td>${M(r[2])}</td><td>${P(r[3])}</td></tr>`).join("");
$("ss").innerHTML=D.scores.map(r=>`<option>${r[0]}</option>`).join("");
function st2(){const k=$("ss").value,r=SC[k],rk=D.scores.findIndex(x=>x[0]==k)+1,dv=(a,b,f)=>(a>=b?"Above":"Below")+" chain average "+f(b);
$("sx").innerHTML=kp([["Risk score",r[6].toFixed(1),"Rank "+rk+" of "+sn.length+" (1 is riskiest)"],["Revenue",M(r[0]),"Share "+P(r[5])],["Profit margin",P(r[2]),dv(r[2],avg(2),P)],["Discount rate",P(r[3]),dv(r[3],avg(3),P)],["Profit volatility",r[4].toFixed(2),dv(r[4],avg(4),x=>x.toFixed(2))]])}
$("ss").onchange=st2;st2();

const LS=D.daily.slice(1).map((a,i)=>D.daily[i]-a),qn=(a,c)=>[...a].sort((x,y)=>x-y)[Math.min(a.length,Math.ceil(c*a.length))-1],mu=LS.reduce((a,b)=>a+b)/LS.length,sdv=Math.sqrt(LS.reduce((a,b)=>a+(b-mu)**2,0)/(LS.length-1));
const zi=p=>{const t=Math.sqrt(-2*Math.log(1-p));return t-(2.515517+.802853*t+.010328*t*t)/(1+1.432788*t+.189269*t*t+.001308*t**3)};
function ev(){const c=+$("cf").value/100,h=qn(LS,c),es=LS.filter(x=>x>=h).reduce((a,b)=>a+b)/LS.filter(x=>x>=h).length,z=zi(c),pv=mu+z*sdv,pe=mu+sdv*Math.exp(-z*z/2)/Math.sqrt(2*Math.PI)/(1-c);
$("oc").textContent=(c*100).toFixed(1)+"%";$("ek").innerHTML=kp([["Historical VaR",F(h),"Observed daily losses"],["Historical ES",F(es),"Average of losses past VaR"],["Parametric VaR",F(pv),"Normal model"],["Days past VaR",LS.filter(x=>x>h).length,"Of "+LS.length+" days"]])}
$("cf").oninput=ev;ev();
(function(){const w=60,L=50,W=640,mx=Math.max(...LS)*1.05,mn=Math.min(...LS),X=i=>L+i*W/(LS.length-1),Y=a=>220-(a-mn)/(mx-mn)*200;let s="",vl=[];
[mn,0,mx].forEach(a=>s+=`<line x1="${L}" x2="${L+W}" y1="${Y(a)}" y2="${Y(a)}" class="gd"/><text x="0" y="${Y(a)+4}" class="t">${F(a)}</text>`);
s+=`<polyline fill="none" stroke="var(--ac)" opacity=".5" stroke-width="1" points="${LS.map((a,i)=>X(i)+","+Y(a)).join(" ")}"/>`;
for(let i=w;i<LS.length;i++)vl.push([i,qn(LS.slice(i-w,i),.95)]);
s+=`<polyline fill="none" stroke="var(--wn)" stroke-width="2.4" points="${vl.map(r=>X(r[0])+","+Y(r[1])).join(" ")}"/>`;
vl.forEach(r=>{if(LS[r[0]]>r[1])s+=`<circle cx="${X(r[0])}" cy="${Y(LS[r[0]])}" r="3.5" fill="var(--rk)"/>`});
MN.forEach((m,i)=>s+=`<text x="${L+(i+.5)*W/12}" y="240" text-anchor="middle" class="t">${m}</text>`);$("rv").innerHTML=s})();
$("mh").innerHTML="<tr><th>Measure</th><th>95% confidence</th><th>99% confidence</th></tr>"+[["21-day Monte Carlo VaR",D.mh[95][0],D.mh[99][0]],["21-day Monte Carlo ES",D.mh[95][1],D.mh[99][1]],["1-day Normal VaR x square root of 21",v[95].par*Math.sqrt(21),v[99].par*Math.sqrt(21)]].map(r=>`<tr><td>${r[0]}</td><td>${F(r[1])}</td><td>${F(r[2])}</td></tr>`).join("");
(function(){const rows=D.scores.map(r=>{const k=D.stores[r[0]];return[r[0],1-k[2],k[3],k[4],k[5],r[1]]}),cols=[["Low margin",P],["Discount rate",P],["Profit volatility",x=>x.toFixed(2)],["Revenue share",P],["Risk score",x=>x.toFixed(1)]];
const mm=cols.map((_,j)=>[Math.min(...rows.map(r=>r[j+1])),Math.max(...rows.map(r=>r[j+1]))]);
$("hm").innerHTML="<tr><th>Store</th>"+cols.map(c=>`<th>${c[0]}</th>`).join("")+"</tr>"+rows.map(r=>`<tr><td>${r[0]}</td>`+cols.map((c,j)=>`<td style="background:rgba(192,57,43,${(.06+.6*(r[j+1]-mm[j][0])/(mm[j][1]-mm[j][0]||1)).toFixed(2)})">${c[1](r[j+1])}</td>`).join("")+"</tr>").join("")})();
$("nav").insertAdjacentHTML("beforeend",'<button class="tg" id="tg" type="button">Switch theme</button>');
$("tg").onclick=()=>{const r=document.documentElement,c=r.dataset.theme||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");r.dataset.theme=c=="dark"?"light":"dark"};

const SH=["Headline results for the year and where the main risks sit.", "Checks on the data behind every figure.", "Daily and monthly profit, with a three-month outlook.", "The conclusions and the actions that follow from them.", "How far profit can fall on a bad day, and how reliable that estimate is.", "Statistical checks on the loss pattern and a backtest of the VaR.", "Set a loss limit and see how often the year would have broken it.", "Which categories and stores drive the worst days, and how well risk is diversified.", "Profit under combined sales declines and cost increases.", "Test your own what-if on sales, costs and discounts.", "How discounting eats into profit margin.", "Whether revenue leans too heavily on a few stores or categories.", "Which stores carry the most risk, and why.", "How each store earned its profit, month by month.", "Every measured risk, rated, with a suggested action.", "How each figure in this report is calculated.", "Likely audience questions, answered from the numbers."];
document.querySelectorAll("section").forEach((e,i)=>e.insertAdjacentHTML("afterbegin",`<div class="sh"><h2>${tabs[i]}</h2><p>${SH[i]}</p></div>`));
$("rb").innerHTML=[["Revenue",M(D.rev)],["Profit",M(D.profit)],["Profit margin",P(D.margin)],["99% daily VaR",F(v[99].hist)]].map(a=>`<div><span>${a[0]}</span><b>${a[1]}</b></div>`).join("");

const RK={High:0,Moderate:1,Low:2},TM={High:"Within 30 days",Moderate:"Within 90 days",Low:"Review each quarter"},rg=[...D.reg].sort((a,b)=>RK[a[2]]-RK[b[2]]),nh=rg.filter(r=>r[2]=="High").length,last=D.monthly[11][2],f3=D.fc.profit[2][0],ok95=D.diag.k95[1]>.05&&D.diag.k99[1]>.05;
$("xk").innerHTML=kp([["Overall risk level",nh>=3?"Elevated":nh?"Moderate":"Low",nh+" of "+rg.length+" risks rated high"],["Daily profit at risk",F(v[95].hist),"95% VaR, one day in 20"],["Profit margin",P(D.margin),M(D.profit)+" profit on "+M(D.rev)+" revenue"],["Three-month outlook",(f3>=last?"+":"")+P(f3/last-1),"Profit by March versus December"]]);
$("xg").innerHTML=[`Profit margin holds at ${P(D.margin)} after the cost of goods.`,`Revenue is spread across stores (concentration index ${Math.round(D.hhi.store)}), so no single store dominates.`,ok95?"The VaR model passed its backtest at both 95% and 99% confidence.":"The VaR backtest needs review at one confidence level.",`Profit is projected at ${M(f3)} in March against ${M(last)} in December.`].map(x=>`<li>${x}</li>`).join("");
$("xa").innerHTML=rg.filter(r=>r[2]!="Low").map(r=>`<li><b>${r[0]}:</b> ${r[1]}.</li>`).join("");
$("xp").innerHTML="<tr><th>Rating</th><th>Risk</th><th>Action</th><th>Timing</th></tr>"+rg.map(r=>`<tr><td><span class="pill ${r[2]}">${r[2]}</span></td><td class="l"><b>${r[0]}</b></td><td class="l">${r[3]}</td><td>${TM[r[2]]}</td></tr>`).join("");
const sq=Object.entries(D.sm).map(([k,a])=>[k,a,a.reduce((x,y)=>x+y,0)]).sort((a,b)=>b[2]-a[2]),al=sq.flatMap(r=>r[1]),lo2=Math.min(...al),hi2=Math.max(...al);
$("sp").innerHTML="<tr><th>Store</th>"+MN.map(m=>`<th>${m}</th>`).join("")+"<th>Total</th></tr>"+sq.map(r=>`<tr><td>${r[0]}</td>`+r[1].map(x=>`<td style="background:rgba(29,111,138,${(.08+.6*(x-lo2)/(hi2-lo2)).toFixed(2)})">${(x/1000).toFixed(0)}K</td>`).join("")+`<td><b>${M(r[2])}</b></td></tr>`).join("");
(function(){const nm=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],a=nm.map((_,k)=>{const x=D.daily.filter((_,i)=>(i+5)%7==k);return x.reduce((p,q)=>p+q,0)/x.length});$("wd").innerHTML=bars(nm.map((n,i)=>[n,a[i]]),Math.max(...a),F,"var(--ac)")})();
$("dmt").innerHTML="<tr><th>Category</th>"+[0,.5,1,1.5].map(x=>`<th>${x?"Discount "+x.toFixed(2):"No discount"}</th>`).join("")+"</tr>"+Object.entries(D.dm).map(([k,a])=>`<tr><td>${k}</td>`+a.map(x=>`<td style="background:rgba(192,57,43,${Math.max(0,(.3-x)/.15*.55).toFixed(2)})">${P(x)}</td>`).join("")+"</tr>").join("");
$("dq").innerHTML="<tr><th>Check</th><th>Count</th><th>Share of rows</th></tr>"+D.dq.map(r=>`<tr><td>${r[0]}</td><td>${r[1].toLocaleString()}</td><td>${P(r[1]/D.n)}</td></tr>`).join("")+`<tr><td><b>Rows kept for modelling</b></td><td><b>${(D.n-D.dq[1][1]).toLocaleString()}</b></td><td><b>${P(1-D.dq[1][1]/D.n)}</b></td></tr>`;

$("pp").onclick=()=>show(cur-1);$("pn").onclick=()=>show(cur+1);
addEventListener("keydown",e=>{if(["INPUT","SELECT"].includes(e.target.tagName))return;if(e.key=="ArrowRight"&&cur<tabs.length-1)show(cur+1);if(e.key=="ArrowLeft"&&cur>0)show(cur-1)});
function pg(i){$("pp").disabled=i==0;$("pn").disabled=i==tabs.length-1;$("pl").textContent=(i+1)+" of "+tabs.length+": "+tabs[i];$("pb").style.width=(i+1)/tabs.length*100+"%"}
pg(0);

$("cs").innerHTML=D.cats.map(c=>`<option>${c[0]}</option>`).join("");
function cx(){const k=$("cs").value,c=D.cats.find(r=>r[0]==k),m=D.cm[k],cv=D.cv.find(r=>r[0]==k)[1],dm=D.dm[k],mx=Math.max(...m.map(r=>r[1])),bw=680/12;
$("ck").innerHTML=kp([["Revenue",M(c[1]),P(c[1]/D.rev)+" of total"],["Profit",M(c[2]),P(c[3])+" margin"],["Demand volatility",P(cv),"Monthly units variation"],["Margin at deepest discount",P(dm[3]),"Against "+P(dm[0])+" with no discount"]]);
$("cc").innerHTML=m.map((r,i)=>`<rect x="${10+i*bw+8}" y="${190-r[1]/mx*150}" width="${bw-16}" height="${r[1]/mx*150}" rx="4" fill="var(--ac)" opacity=".85"/><text x="${10+i*bw+bw/2}" y="${184-r[1]/mx*150}" text-anchor="middle" class="t">${(r[1]/1000).toFixed(0)}K</text><text x="${10+i*bw+bw/2}" y="208" text-anchor="middle" class="t">${MN[i]}</text>`).join("")}
$("cs").onchange=cx;cx();
$("sa").innerHTML=$("sb").innerHTML=D.scores.map(r=>`<option>${r[0]}</option>`).join("");$("sb").selectedIndex=D.scores.length-1;
function cp(){const a=SC[$("sa").value],b=SC[$("sb").value],rows=[["Risk score",6,x=>x.toFixed(1)],["Revenue",0,M],["Profit",1,M],["Profit margin",2,P],["Discount rate",3,P],["Profit volatility",4,x=>x.toFixed(2)],["Revenue share",5,P]];
$("cmp").innerHTML=`<tr><th>Measure</th><th>${$("sa").value}</th><th>${$("sb").value}</th><th>Difference</th></tr>`+rows.map(([n,i,f])=>{const df=a[i]-b[i];return `<tr><td>${n}</td><td>${f(a[i])}</td><td>${f(b[i])}</td><td>${df>0?"+":df<0?"-":""}${f(Math.abs(df))}</td></tr>`}).join("")}
$("sa").onchange=$("sb").onchange=cp;cp();
function lim(){const x=+$("lm").value,br=LS.filter(a=>a>x),n=br.length,wd=Math.max(...LS);$("ol").textContent=F(x);
$("lk").innerHTML=kp([["Days over the limit",n,P(n/LS.length)+" of "+LS.length+" days"],["Average excess on those days",n?F(br.reduce((a,b)=>a+b,0)/n-x):"None","Beyond the limit"],["Worst day against limit",F(wd),(wd/x).toFixed(1)+" times the limit"],["Limit against 99% VaR",P(x/v[99].hist),x>=v[99].hist?"Limit covers 99% of days":"Tighter than the 99% VaR"]]);
$("lb").innerHTML=bars([["Your limit",x],["95% VaR",v[95].hist],["99% VaR",v[99].hist],["95% ES",v[95].hes],["Worst day",wd]],Math.max(x,wd),F,"var(--ac)")}
$("lm").oninput=lim;lim();
$("qa").innerHTML=[["What does 95% VaR mean here?",`On 19 days in 20 the fall in daily profit stays under ${F(v[95].hist)}. On the other day it can be larger, and the average of those worst days is ${F(v[95].hes)}.`],
["Can the VaR model be trusted?",ok95?`It passed the Kupiec backtest at 95% and 99% (p-values ${pv(G.k95[1])} and ${pv(G.k99[1])}). Losses are close to Normal, but they are linked from one day to the next.`:"It failed the backtest at one level, so prefer the Historical figure and review the model."],
["What is the biggest risk?",`${rg[0][0]}: ${rg[0][1]}. Suggested action: ${rg[0][3].toLowerCase()}.`],
["How much does discounting cost?",`Margin falls from ${P(D.md.buckets[0][1])} with no discount to ${P(D.md.buckets[3][1])} at the deepest level. Discounts add up to ${M(D.disc_total)} for the year.`],
["What if costs rise and sales fall?",`With costs up 10% and sales down 10%, profit changes by ${F(D.stress["0.10"][2])}, or ${P(D.stress["0.10"][2]/D.profit)} of annual profit.`],
["Is revenue too concentrated?",`Across stores, no: the index is ${Math.round(D.hhi.store)} and the top five stores earn ${P(D.hhi.top5)}. Across categories, yes: the index is ${Math.round(D.hhi.cat)}.`],
["What drives the worst days?",`${D.rcat.slice().sort((x,y)=>y[1]-x[1])[0][0]} supplies ${P(D.rcat.slice().sort((x,y)=>y[1]-x[1])[0][1])} of the loss on the worst 5% of days, and the ten worst days are listed on the Risk contribution tab.`],["Which store should we watch?",`${D.scores[0][0]} has the highest risk score at ${D.scores[0][1]} out of 100.`],
["How reliable is the forecast?",`It is a straight-line trend on 12 monthly totals. For March the 80% range for profit is ${M(D.fc.profit[2][1])} to ${M(D.fc.profit[2][2])}.`],
["Where does the data come from?","The figures use 2,000,000 generated transactions for 22 stores. Replacing them with real sales data would change every number."]].map(([q,a])=>`<details><summary>${q}</summary><p>${a}</p></details>`).join("");

const so=a=>[...a].sort((x,y)=>y[1]-x[1]),tc=so(D.rcat)[0],ts=so(D.rst)[0];
$("rk").innerHTML=kp([["Category diversification",D.div.cat.toFixed(2)+"x","Sum of standalone VaR over portfolio VaR"],["Store diversification",D.div.store.toFixed(2)+"x","Higher means more risk spread away"],["Largest category in the tail",tc[0],P(tc[1])+" of tail loss"],["Largest store in the tail",ts[0],P(ts[1])+" of tail loss"]]);
const rt=rows=>"<tr><th>Name</th><th>Share of tail loss</th><th>Share of revenue</th><th>Risk to size</th><th>Standalone 95% VaR</th></tr>"+so(rows).map(r=>{const q=r[1]/r[4];return `<tr><td>${r[0]}</td><td>${P(r[1])}</td><td>${P(r[4])}</td><td style="color:${q>1.25?"var(--rk)":q<.8?"var(--ok)":"inherit"}"><b>${q.toFixed(2)}x</b></td><td>${F(r[3])}</td></tr>`}).join("");
$("rc").innerHTML=rt(D.rcat);$("rs").innerHTML=rt(D.rst);
$("cr").innerHTML="<tr><th></th>"+D.corr.names.map(n=>`<th>${n}</th>`).join("")+"</tr>"+D.corr.m.map((r,i)=>`<tr><td>${D.corr.names[i]}</td>`+r.map(x=>`<td style="background:rgba(29,111,138,${(Math.abs(x)*.6).toFixed(2)})">${x.toFixed(2)}</td>`).join("")+"</tr>").join("");
$("w10").innerHTML="<tr><th>Date</th><th>Total loss</th><th>Biggest losing category</th><th>Biggest losing store</th></tr>"+D.worst.map(r=>`<tr><td>${r[0]}</td><td>${F(r[1])}</td><td>${r[2]} (${F(r[3])})</td><td>${r[4]} (${F(r[5])})</td></tr>`).join("");
$("ci").innerHTML="<tr><th>Confidence</th><th>Estimate</th><th>95% range</th><th>Range width</th></tr>"+[95,99].map(c=>{const r=D.ci[c],p=v[c].hist;return `<tr><td>${c}% VaR</td><td>${F(p)}</td><td>${F(r[0])} to ${F(r[1])}</td><td>${P((r[1]-r[0])/p)} of the estimate</td></tr>`}).join("");

$("tg").insertAdjacentHTML("beforebegin",'<button class="tg" id="dm" type="button">Play demo</button>');
const fmt=x=>Math.floor(x/60)+":"+String(Math.floor(x%60)).padStart(2,"0");
const DS=[["Overview",50,()=>`Hello, everyone. This is our Sales Risk Dashboard for the food and beverage and retail business: 22 stores and two million transactions. Revenue is ${M(D.rev)} and profit is ${M(D.profit)}, a margin of ${P(D.margin)}. Profit can fall by about ${F(v[95].hist)} on one day in twenty.`,[]],
["Data quality",20,()=>`Before trusting any number, we check the data. Only ${D.dq[1][1].toLocaleString()} of ${D.n.toLocaleString()} rows have a net sale of zero or less, so data problems are not distorting the results.`,[]],
["Trends and forecast",25,()=>"Here is profit through the year, with red dots on the days that fell past the 95% VaR, and a simple three-month outlook with an 80% range. It is a straight-line trend, so treat it as a guide.",[]],
["Value at Risk",35,()=>`Value at Risk is the loss that daily profit stays under on most days: ${F(v[95].hist)} at 95% confidence and ${F(v[99].hist)} at 99%. The backtest ${ok95?"passes at both levels, so we can rely on these figures":"needs review"}.`,[[14,()=>{$("cf").value=99;ev()}]]],
["Risk contribution",25,()=>`Now, where the risk comes from. ${tc[0]} supplies ${P(tc[1])} of the loss on the worst 5% of days, and stores spread risk with diversification of about ${D.div.store.toFixed(1)} times.`,[]],
["Margin and discount",30,()=>`Discounting is the clearest margin leak. Margin falls from ${P(D.md.buckets[0][1])} with no discount to ${P(D.md.buckets[3][1])} at the deepest discount, and the correlation between discount and margin is ${D.md.r.toFixed(2)}.`,[]],
["Scenario simulator",30,()=>"Here you can test what-ifs. As I move the sliders for costs, sales and discount spend, profit and margin update instantly.",[[6,()=>{$("a2").value=10;sim()}],[14,()=>{$("a1").value=-10;sim()}],[22,()=>{$("a3").value=50;sim()}]]],
["Risk register",15,()=>`The risk register rates ${rg.length} risks, and ${nh} are high: ${rg.filter(r=>r[2]=="High").map(r=>r[0].toLowerCase()).join(", ")}. Each has a suggested action.`,[]],
["Executive summary",10,()=>"To summarise: profit is healthy, daily risk is measured, and the priorities are costs, discounts and category concentration. Thank you. I am happy to take questions.",[]]];
let dem=null;
function dstop(){if(dem)clearInterval(dem.t);dem=null;$("cap").classList.remove("on");$("dm").textContent="Play demo"}
function dgo(k){dem.k=k;dem.e=0;dem.f=0;show(tabs.indexOf(DS[k][0]));$("cpt").textContent=DS[k][2]()}
function tick(){if(!dem||dem.p)return;dem.e+=.25;dem.tot+=.25;const s=DS[dem.k];
s[3].forEach((a,j)=>{if(dem.e>=a[0]&&!(dem.f&(1<<j))){dem.f|=1<<j;a[1]()}});
if(!(dem.f&64)&&dem.e>=s[1]/2){dem.f|=64;scrollTo({top:innerHeight*.55,behavior:"smooth"})}
$("cpb").style.width=Math.min(100,dem.tot/240*100)+"%";$("cpm").textContent="Step "+(dem.k+1)+" of "+DS.length+"  |  "+fmt(dem.tot)+" of 4:00";
if(dem.e>=s[1]){if(dem.k<DS.length-1)dgo(dem.k+1);else dstop()}}
$("dm").onclick=()=>{if(dem){dstop();return}["a1","a2","a3"].forEach(i=>$(i).value=0);$("cf").value=95;sim();ev();$("cap").classList.add("on");$("cpp").textContent="Pause";$("dm").textContent="Stop demo";dem={k:0,e:0,f:0,tot:0,p:false};dem.t=setInterval(tick,250);dgo(0)};
$("cpp").onclick=()=>{if(dem){dem.p=!dem.p;$("cpp").textContent=dem.p?"Resume":"Pause"}};$("cps").onclick=dstop;
</script></body></html>
"""

with open("risk_dashboard.html", "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(out, default=float)))
print("Done -> risk_dashboard.html")

"""Train five models, select by trip-held-out macro F1, tune with a GP and export."""
import argparse
import json
import time
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning

from .models import SEED, NAMES, DEFAULTS, SPACES, make, probability
from .data import GAP, FEATURES, K, read_inputs
from .decoder import ordered_f1, decode

FIT_OBS=260
CV_OBS=450
FOLDS=3


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")


def sample(id,n,limit,salt):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,int(id),salt]))
    return np.sort(rng.choice(n,min(n,limit),replace=False))


def load_data(cache):
    data={}
    for part in ["train","student_test"]:
        data[part]={}
        for file in sorted((cache/part).glob("*.npz")):
            with np.load(file,allow_pickle=False) as raw:
                rec=dict(raw)
            rec["y"]=(rec["ids"]==rec["ref"][:,None]).astype(np.uint8)
            data[part][file.stem]=rec
    if len(data["train"])!=80 or len(data["student_test"])!=10:
        raise ValueError("Run pathsnap.prepare first, using the supplied splits")
    return data


def gather(records,names,limit,salt):
    xs,ys=[],[]
    for id in names:
        r=records[id]
        ix=sample(id,len(r["y"]),limit,salt)
        xs.append(r["x"][ix]); ys.append(r["y"][ix])
    return np.concatenate(xs),np.concatenate(ys)


def fit(model,x,y):
    return model.fit(x.reshape(-1,x.shape[-1]),y.ravel())


def classification(y,p):
    y=y.ravel().astype(bool)
    p=np.asarray(p,dtype=np.float64).ravel().clip(1e-7,1-1e-7)
    pred=p>=.5
    tn=int((~pred & ~y).sum()); fp=int((pred & ~y).sum())
    fn=int((~pred & y).sum()); tp=int((pred & y).sum())
    pos=2*tp/max(1,2*tp+fp+fn); neg=2*tn/max(1,2*tn+fp+fn)
    return dict(macro_f1=(pos+neg)/2,candidate_accuracy=(tp+tn)/len(y),
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                log_loss=float(-(np.log(p[y]).sum()+np.log1p(-p[~y]).sum())/len(y)),
                tn=tn,fp=fp,fn=fn,tp=tp)


def route_scores(r,chosen,route=None):
    n=len(chosen); idx=np.arange(n)
    # A state always belongs to the retained candidates.
    col=(r["ids"]==chosen[:,None]).argmax(axis=1)
    ep=r["endpoints"][idx,col]
    contiguous=(chosen[1:]==chosen[:-1])|(ep[:-1,1]==ep[1:,0])
    ok=np.diff(r["t"])<=GAP
    pred=chosen if route is None else route
    return dict(n=n,correct=int((chosen==r["ref"]).sum()),links=int(ok.sum()),
                joined=int(contiguous[ok].sum()),ordered_f1=float(ordered_f1(pred,r["route"])),
                observation_macro_f1=float(f1_score(r["ref"],chosen,average="macro",zero_division=0)),
                misses=int((~r["y"].any(axis=1)).sum()))


def combine(rows):
    n=sum(z["n"] for z in rows)
    return dict(top1=sum(z["correct"] for z in rows)/n,
                continuity=sum(z["joined"] for z in rows)/max(1,sum(z["links"] for z in rows)),
                ordered_f1=float(np.mean([z["ordered_f1"] for z in rows])),
                observation_macro_f1=float(np.mean([z["observation_macro_f1"] for z in rows])),
                miss_rate=sum(z["misses"] for z in rows)/n)


def make_folds(records):
    keys=list(sorted(records))
    plan=np.random.default_rng(SEED).permutation(keys)
    return [sorted(group.tolist()) for group in np.array_split(plan,FOLDS)]


def cv_inputs(train,folds):
    parts=[]
    for valid in folds:
        fit_ids=sorted(set(train)-set(valid))
        x,y=gather(train,fit_ids,FIT_OBS,0)
        vx,vy=gather(train,valid,CV_OBS,1)
        parts.append((x,y,vx,vy))
    return parts


def cross_validate(name,params,parts):
    results=[]
    for i,(x,y,vx,vy) in enumerate(parts):
        model=fit(make(name,params),x,y)
        p=probability(model,vx)
        stats=classification(vy,p)
        # Misses remain false even if every candidate is negative.
        stats["top1"]=float(vy[np.arange(len(vy)),p.argmax(axis=1)].mean())
        results.append(dict(fold=i+1,**stats))
    return results


def final_scores(name,params,data,fx,fy):
    t0=time.perf_counter()
    model=fit(make(name,params),fx,fy)
    fit_sec=time.perf_counter()-t0
    train_score=classification(fy,probability(model,fx))["macro_f1"]
    times=0.; results=[]; ys=[]; ps=[]
    for id,r in data["student_test"].items():
        t0=time.perf_counter()
        p=probability(model,r["x"])
        times+=time.perf_counter()-t0
        chosen=r["ids"][np.arange(len(p)),p.argmax(1)]
        results.append(dict(trip=id,**route_scores(r,chosen)))
        ys.append(r["y"].ravel()); ps.append(p.ravel())
    summary=dict(train_macro_f1=train_score,fit_s=fit_sec,inference_s=times,
                 **classification(np.concatenate(ys),np.concatenate(ps)),**combine(results))
    return model,summary,results


def bayes(name,baseline,parts,trials=8):
    space=SPACES[name]
    keys=list(space)
    def transform(row):
        p=DEFAULTS[name].copy()
        for u,key in zip(row,keys):
            lo,hi,kind=space[key]
            val=np.exp(np.log(lo)+u*np.log(hi/lo)) if kind=="log" else lo+u*(hi-lo)
            p[key]=int(round(val)) if kind=="int" else float(round(val,5))
        return p
    initial=[]
    for key in keys:
        lo,hi,kind=space[key]; val=DEFAULTS[name][key]
        initial.append(np.log(val/lo)/np.log(hi/lo) if kind=="log" else (val-lo)/(hi-lo))
    grid=qmc.Sobol(len(keys),scramble=True,seed=SEED).random_base2(9)
    xx=[initial]; scores=[baseline]; params=[DEFAULTS[name].copy()]
    logs=[dict(trial=1,proposal="baseline",cv_macro_f1=baseline,params=json.dumps(params[0]),seconds=0.)]
    for i in range(1,trials):
        available=np.array([v for v in grid if transform(v) not in params])
        if not len(available): break
        if i<=2:
            ix=0
        else:
            gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*Matern(length_scale=np.ones(len(keys)),nu=2.5),
                                         alpha=.003,normalize_y=True,random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",ConvergenceWarning)
                gp.fit(xx,scores)
            m,s=gp.predict(available,return_std=True)
            gain=m-max(scores)-.00005
            z=gain/np.maximum(s,1e-12)
            ei=gain*ndtr(z)+s*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            ix=int(np.argmax(ei))
        x0=available[ix]; p=transform(x0)
        tick=time.perf_counter()
        q=cross_validate(name,p,parts)
        score=float(np.mean([a["macro_f1"] for a in q]))
        xx.append(x0); scores.append(score); params.append(p)
        logs.append(dict(trial=i+1,proposal="Sobol" if i<=2 else "GP expected improvement",
                         cv_macro_f1=score,params=json.dumps(p),seconds=time.perf_counter()-tick))
        print(f"Bayesian trial {i+1}/{trials}  {name}  {score:.5f}",flush=True)
    j=int(np.argmax(scores))
    return params[j],scores[j],logs


def run_decoder(data,model,name,label,beta,root,out,rows):
    for id,r in data["student_test"].items():
        tr=read_inputs(root/"student_test"/id)
        p=probability(model,r["x"])
        independent=r["ids"][np.arange(len(p)),p.argmax(1)]
        start=time.perf_counter()
        chosen,breaks,paths=decode(tr,r["ids"],r["x"],p,beta)
        dt=time.perf_counter()-start
        for mode,choice,seq,nb,sec in [("Independent",independent,None,None,0.),
                                        ("Viterbi",chosen,np.concatenate(paths),int(breaks.sum()),dt)]:
            rows.append(dict(model=name,experiment=label,mode=mode,trip=id,segments=nb,decoder_s=sec,
                             **route_scores(r,choice,seq)))
        if label=="Selected":
            save_json(out/"predictions"/(id+".json"),dict(trip=id,predicted_arc_ids=chosen.tolist(),
                                                          segments=paths,break_rows=np.flatnonzero(breaks).tolist()))


def main():
    p=argparse.ArgumentParser()
    for a,d in [("data",None),("cache","cache"),("out","outputs"),("models","models")]:
        p.add_argument("--"+a,type=Path,required=d is None,default=d)
    p.add_argument("--only",choices=NAMES,nargs="+",help="Run selected baseline models, omitting selection and tuning")
    p.add_argument("--reuse-baselines",action="store_true",help="Reuse saved baselines from this same dataset and configuration")
    args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True); args.models.mkdir(parents=True,exist_ok=True)
    (out/"baselines").mkdir(exist_ok=True); (out/"predictions").mkdir(exist_ok=True)
    data=load_data(args.cache); train=data["train"]; folds=make_folds(train)
    save_json(out/"folds.json",dict(seed=SEED,validation_trips=folds,fit_observations_cap=FIT_OBS,validation_observations_cap=CV_OBS))
    parts=cv_inputs(train,folds);fx,fy=gather(train,sorted(train),FIT_OBS,0)
    for name in args.only or NAMES:
        file=out/"baselines"/(name.replace(" ","_")+".json")
        if args.reuse_baselines and file.exists() and not args.only:
            continue
        print(f"Fitting {name}",flush=True)
        tick=time.perf_counter()
        cv=cross_validate(name,DEFAULTS[name],parts)
        cv_sec=time.perf_counter()-tick
        _,summary,trips=final_scores(name,DEFAULTS[name],data,fx,fy)
        save_json(file,dict(model=name,params=DEFAULTS[name],cv=cv,cv_s=cv_sec,test=summary,trips=trips))
        print(name,summary["macro_f1"],summary["top1"],flush=True)
    if args.only: return
    base=[json.loads((out/"baselines"/(n.replace(" ","_")+".json")).read_text()) for n in NAMES]
    table=[]
    for b in base:
        score=[f["macro_f1"] for f in b["cv"]]
        table.append(dict(model=b["model"],cv_macro_f1=np.mean(score),cv_sd=np.std(score),cv_s=b["cv_s"],**b["test"]))
    pd.DataFrame(table).to_csv(out/"model_comparison.csv",index=False)
    best=max(table,key=lambda r:r["cv_macro_f1"])
    name=best["model"]
    params,score,logs=bayes(name,float(best["cv_macro_f1"]),parts)
    pd.DataFrame(logs).to_csv(out/"bayesian_trials.csv",index=False)
    model,summary,_=final_scores(name,params,data,fx,fy)
    pd.DataFrame([dict(stage="Untuned",**best),dict(stage="Tuned",model=name,cv_macro_f1=score,**summary)]).to_csv(out/"tuning_comparison.csv",index=False)
    beta=max(6.,sum(float(r["residual_sum"]) for r in train.values())/max(1,sum(int(r["residual_count"]) for r in train.values())))
    config=dict(model=name,params=params,features=FEATURES,k=K,beta_m=beta,max_gap_s=GAP,
                candidate_limit_m=400,seed=SEED,fit_observations=len(fy))
    # Boosters have a native portable representation, independent of Python class pickling.
    if name=="XGBoost":
        weights="selected.json"; model.booster.save_model(str(args.models/weights))
    elif name=="LightGBM":
        weights="selected.txt"; model.booster.save_model(str(args.models/weights))
    else:
        weights="selected.joblib"; joblib.dump(model,args.models/weights,compress=3)
    config["weights"]=weights
    save_json(args.models/"config.json",config)
    hmm=fit(make("HMM",DEFAULTS["HMM"]),fx,fy)
    rows=[];run_decoder(data,hmm,"HMM","Standalone HMM",beta,args.data,out,rows)
    run_decoder(data,model,name,"Selected",beta,args.data,out,rows)
    pd.DataFrame(rows).to_csv(out/"decoder_per_trip.csv",index=False)
    result=[]
    for label in ["Standalone HMM","Selected"]:
        for mode in ["Independent","Viterbi"]:
            group=[r for r in rows if r["experiment"]==label and r["mode"]==mode]
            result.append(dict(experiment=label,model=group[0]["model"],mode=mode,**combine(group),
                               decoder_s=sum(z["decoder_s"] for z in group)))
    pd.DataFrame(result).to_csv(out/"decoder_summary.csv",index=False)
    print("Selected",name,"configuration",params,"beta",beta,flush=True)

if __name__=="__main__":
    main()

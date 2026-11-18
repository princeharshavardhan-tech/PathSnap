"""Prediction reads only .track, .nodes and .arcs. The API scores after prediction."""
from pathlib import Path
import json
import joblib
import numpy as np
from .data import read_inputs, candidates, FEATURES, K, GAP
from .decoder import decode, ordered_f1
from .models import Boosted, probability


def load_model(directory):
    directory=Path(directory)
    config=json.loads((directory/"config.json").read_text())
    if config["features"]!=FEATURES or config["k"]!=K or config["max_gap_s"]!=GAP:
        raise ValueError("Model and feature configuration do not match")
    filename=directory/config["weights"]
    name=config["model"]
    if name=="XGBoost":
        import xgboost as xgb
        model=Boosted(name,config["params"])
        model.booster=xgb.Booster(params={"nthread":3},model_file=str(filename))
    elif name=="LightGBM":
        import lightgbm as lgb
        model=Boosted(name,config["params"])
        model.booster=lgb.Booster(model_file=str(filename))
    else:
        model=joblib.load(filename)
    return model,config


def infer(folder,model,config):
    tr=read_inputs(folder)
    ids,x=candidates(tr)
    chosen,breaks,paths=decode(tr,ids,x,probability(model,x),config["beta_m"])
    return tr,ids,chosen,breaks,paths


def map_layers(tr,ids,paths):
    pts=tr.track[:,:2]
    lo=pts.min(0)-.002; hi=pts.max(0)+.002
    e=tr.edges[tr.edge_ids]
    a,b=tr.nodes[e[:,0]],tr.nodes[e[:,1]]
    hits=np.flatnonzero((np.maximum(a,b)>=lo).all(1)&(np.minimum(a,b)<=hi).all(1))
    # Keep the road layer modest on a laptop; include every shortlisted road.
    other=tr.edge_ids[hits[::max(1,int(np.ceil(len(hits)/2400)))]]
    network=np.unique(np.r_[other,ids.ravel()])
    # A road can have two directed arcs with the same geometry. Draw it once here.
    undirected=np.unique(np.sort(tr.edges[network],axis=1),axis=0)
    roads=tr.nodes[undirected][:,:,::-1].round(6).tolist()
    routes=[]
    for segment in paths:
        e=tr.edges[np.asarray(segment)]
        line=tr.nodes[np.r_[e[0,0],e[:,1]]][:,::-1].round(6).tolist()
        routes.append(line)
    return dict(roads=roads,gps=pts[:,::-1].round(6).tolist(),predicted=routes)


def score_after_prediction(tr,chosen,paths,reference_path):
    from .reference import align
    route,truth,_,_=align(tr,reference_path)
    # Reference geometry and labels never leave this function.
    return dict(top1=float(np.mean(truth==chosen)),ordered_f1=float(ordered_f1(np.concatenate(paths),route)))


def strict_continuity(tr,chosen):
    ok=np.diff(tr.track[:,2])<=GAP
    adjacent=(chosen[1:]==chosen[:-1])|(tr.edges[chosen[:-1],1]==tr.edges[chosen[1:],0])
    return float(adjacent[ok].mean()) if np.any(ok) else None

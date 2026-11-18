"""The five candidate scorers share the same fit and probability interface."""
import numpy as np
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

SEED=817
NAMES=["XGBoost","LightGBM","Random Forest","KNN","HMM"]
DEFAULTS={
    "XGBoost":dict(rounds=70,max_depth=5,learning_rate=.10),
    "LightGBM":dict(rounds=70,num_leaves=23,learning_rate=.09,min_data_in_leaf=45),
    "Random Forest":dict(n_estimators=40,max_depth=15,min_samples_leaf=6),
    "KNN":dict(n_neighbors=21,weights="distance"),
    "HMM":dict(bins=20,smoothing=1.,strength=.8),
}
SPACES={
    "XGBoost":{"rounds":(40,140,"int"),"max_depth":(3,8,"int"),"learning_rate":(.035,.18,"log")},
    "LightGBM":{"rounds":(40,140,"int"),"num_leaves":(12,55,"int"),"learning_rate":(.035,.18,"log"),"min_data_in_leaf":(18,100,"int")},
    "Random Forest":{"n_estimators":(25,85,"int"),"max_depth":(8,21,"int"),"min_samples_leaf":(2,16,"int")},
    "KNN":{"n_neighbors":(7,45,"int")},
    "HMM":{"bins":(8,48,"int"),"smoothing":(.3,8.,"log"),"strength":(.45,1.35,"real")},
}


class EmissionModel:
    """Supervised binned likelihoods for match vs background; later combined with graph transitions."""
    channels=(0,2,7)
    def __init__(self,bins=20,smoothing=1.,strength=.8):
        self.bins,self.smoothing,self.strength=bins,smoothing,strength

    def fit(self,x,y):
        counts=np.bincount(y,minlength=2)
        self.prior=float(np.log((counts[1]+1)/(counts[0]+1)))
        self.tables=[]
        for f in self.channels:
            cut=np.unique(np.quantile(x[:,f],np.linspace(0,1,self.bins+1)[1:-1]))
            vals=np.searchsorted(cut,x[:,f],side="right")
            n=len(cut)+1
            density=[(np.bincount(vals[y==i],minlength=n)+self.smoothing)/(counts[i]+n*self.smoothing) for i in (0,1)]
            self.tables.append((cut,np.log(density[1]/density[0])))
        return self

    def predict_proba(self,x):
        logodds=np.full(len(x),self.prior)
        for f,(cut,ratio) in zip(self.channels,self.tables):
            logodds+=self.strength*ratio[np.searchsorted(cut,x[:,f],side="right")]
        p=expit(logodds)
        return np.column_stack([1-p,p])


class Boosted:
    def __init__(self,name,params):
        self.name=name
        self.params=params.copy()

    def fit(self,x,y):
        args=self.params.copy()
        rounds=args.pop("rounds")
        if self.name=="XGBoost":
            import xgboost as xgb
            args.update(objective="binary:logistic",tree_method="hist",seed=SEED,nthread=3,verbosity=0)
            self.booster=xgb.train(args,xgb.DMatrix(x,label=y,nthread=3),num_boost_round=rounds)
        else:
            import lightgbm as lgb
            args.update(objective="binary",verbosity=-1,seed=SEED,num_threads=3,force_col_wise=True)
            self.booster=lgb.train(args,lgb.Dataset(x,label=y.astype(np.float32)),num_boost_round=rounds)
        return self

    def predict_proba(self,x):
        if self.name=="XGBoost":
            import xgboost as xgb
            p=self.booster.predict(xgb.DMatrix(x,nthread=3))
        else:
            p=self.booster.predict(x,num_threads=3)
        return np.column_stack([1-p,p])


def make(name,params):
    if name in ("XGBoost","LightGBM"):
        return Boosted(name,params)
    if name=="Random Forest":
        return RandomForestClassifier(**params,max_features=.8,n_jobs=3,random_state=SEED)
    if name=="KNN":
        return make_pipeline(StandardScaler(),KNeighborsClassifier(**params,algorithm="kd_tree",leaf_size=48,n_jobs=3))
    if name=="HMM":
        return EmissionModel(**params)
    raise ValueError(f"Unknown model {name}")


def probability(model,x):
    return np.clip(model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:-1]),1e-7,1-1e-7)

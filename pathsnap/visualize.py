"""Produce study figures and an offline Folium map from recorded development results."""
import argparse
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import folium
from branca.element import Element
from .data import read_inputs, candidates
from .service import map_layers

DARK="#174b42"; TEAL="#087c68"; GOLD="#c86b31"; GRAY="#9caeaa"


def chart_setup():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.spines.left":False,"axes.spines.bottom":False,
                         "axes.titleweight":"bold","axes.titlecolor":DARK,"axes.labelcolor":DARK,
                         "xtick.color":"#435b56","ytick.color":"#435b56","savefig.facecolor":"white"})


def charts(out):
    chart_setup()
    figdir=out/"figures";figdir.mkdir(exist_ok=True)
    inv=pd.read_csv(out/"inventory.csv",dtype={"trip":str})
    fig,axes=plt.subplots(1,2,figsize=(9.4,2.8),constrained_layout=True)
    for part,color in [("train",TEAL),("student_test",GOLD)]:
        g=inv[inv.split==part]
        axes[0].scatter(g.observations,g.arcs,color=color,s=19,alpha=.70,label=part.replace("_"," "))
        axes[1].scatter(g.aligned_distance_p50,100*g.candidate_misses/g.observations,color=color,s=19,alpha=.75)
    axes[0].set(xlabel="GPS observations per trip",ylabel="Directed arcs",xscale="log",yscale="log",
                title="Trip and network sizes");axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(xlabel="Median distance to aligned route (m)",ylabel="Candidate misses (%)",title="Reference alignment and coverage")
    for ax in axes:ax.grid(alpha=.14)
    fig.savefig(figdir/"eda.png",dpi=155);plt.close(fig)
    t=pd.read_csv(out/"model_comparison.csv");x=np.arange(len(t)); names=["XGBoost","LightGBM","RF","KNN","HMM*"]
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.1),constrained_layout=True)
    axes[0].bar(x,t.cv_macro_f1,yerr=t.cv_sd,color=TEAL,capsize=3);axes[0].set_ylim(.84,.96)
    axes[0].set_title("Grouped CV · candidate macro F1")
    axes[1].bar(x-.18,t.top1,.36,color=GOLD,label="Top-1");axes[1].bar(x+.18,t.macro_f1,.36,color=TEAL,label="Candidate macro F1")
    axes[1].set_ylim(.80,.96);axes[1].set_title("Student test · agreement");axes[1].legend(frameon=False,fontsize=7,ncol=2,loc="lower center")
    axes[2].bar(x,t.log_loss,color="#50878e");axes[2].set_ylim(0,max(t.log_loss)*1.20);axes[2].set_title("Student test · binary log loss")
    for ax in axes:ax.set_xticks(x,names,rotation=25,ha="right");ax.grid(axis="y",alpha=.14)
    fig.savefig(figdir/"comparison.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(10.4,2.55),constrained_layout=True)
    for ax,(_,row),lab in zip(axes,t.iterrows(),names):
        mat=np.array([[row.tn,row.fp],[row.fn,row.tp]],int)
        frac=mat/np.maximum(1,mat.sum(axis=1,keepdims=True))
        ax.imshow(frac,vmin=0,vmax=1,cmap="GnBu")
        for i in [0,1]:
            for j in [0,1]:
                ax.text(j,i,f"{mat[i,j]:,}",ha="center",va="center",fontsize=8,color="white" if frac[i,j]>.6 else DARK)
        ax.set(xticks=[0,1],xticklabels=["No","Yes"],yticks=[0,1],yticklabels=["No","Yes"],title=lab,xlabel="Predicted")
    axes[0].set_ylabel("Reference")
    fig.savefig(figdir/"confusion.png",dpi=160);plt.close(fig)
    trials=pd.read_csv(out/"bayesian_trials.csv")
    fig,ax=plt.subplots(figsize=(8.3,2.3),constrained_layout=True)
    ax.plot(trials.trial,trials.cv_macro_f1,"o-",color=TEAL,lw=1.5,ms=5)
    ax.axhline(trials.cv_macro_f1.iloc[0],color=GOLD,ls="--",label="Initial configuration")
    ax.set(xlabel="Trial",ylabel="CV macro F1",xticks=trials.trial);ax.legend(frameon=False,loc="lower right");ax.grid(alpha=.15)
    fig.savefig(figdir/"bayes.png",dpi=155);plt.close(fig)
    results=pd.read_csv(out/"decoder_per_trip.csv",dtype={"trip":str});g=results[results.experiment=="Selected"].copy();g["top1"]=g.correct/g.n
    pivot=g.pivot(index="trip",columns="mode",values="top1")
    fig,ax=plt.subplots(figsize=(9,2.6),constrained_layout=True)
    x=np.arange(len(pivot));ax.vlines(x,pivot.Independent,pivot.Viterbi,color=GRAY,lw=2)
    ax.scatter(x,pivot.Independent,color=GOLD,label="Independent",s=30);ax.scatter(x,pivot.Viterbi,color=TEAL,label="Viterbi",s=30)
    ax.set(xticks=x,xticklabels=[str(int(i)) for i in pivot.index],xlabel="Student-test trip ID",ylabel="Top-1 agreement",ylim=(.70,1.02))
    ax.legend(frameon=False,ncol=2,loc="lower left");ax.grid(axis="y",alpha=.17)
    fig.savefig(figdir/"decoder.png",dpi=155);plt.close(fig)


def spatial(root,out,trip_id):
    folder=root/"student_test"/trip_id
    tr=read_inputs(folder);ids,_=candidates(tr)
    p=json.loads((out/"predictions"/(trip_id+".json")).read_text())
    layers=map_layers(tr,ids,p["segments"])
    route=np.loadtxt(folder/(trip_id+".route"),ndmin=1,dtype=np.int32)
    ref=tr.nodes[tr.edges[route]][:,:,::-1].tolist()
    gps=layers["gps"]
    m=folium.Map(location=np.median(gps,axis=0).tolist(),tiles=None,zoom_start=13,prefer_canvas=False,control_scale=True,zoom_animation=False,fade_animation=False)
    vendor=Path(__file__).resolve().parents[1]/"ui/vendor"
    base=os.path.relpath(vendor,out).replace(os.sep,"/")
    m.default_js=[("leaflet",base+"/leaflet.js"),("jquery",base+"/jquery.min.js")]
    m.default_css=[("leaflet_css",base+"/leaflet.css")]
    road=folium.FeatureGroup(name="Road network").add_to(m)
    folium.PolyLine(layers["roads"],color="#a4b1ac",weight=1.4,opacity=.76).add_to(road)
    reference=folium.FeatureGroup(name="Reference route").add_to(m)
    folium.PolyLine(ref,color="#714aa0",weight=7,opacity=.68).add_to(reference)
    pred=folium.FeatureGroup(name="Predicted route").add_to(m)
    folium.PolyLine(layers["predicted"],color=TEAL,weight=3.5,dash_array="7 4",opacity=.98).add_to(pred)
    points=folium.FeatureGroup(name="GPS observations").add_to(m)
    for coord in gps:folium.CircleMarker(coord,radius=1.5,stroke=False,fill=True,fill_color=GOLD,fill_opacity=.85).add_to(points)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([np.min(gps,axis=0).tolist(),np.max(gps,axis=0).tolist()],padding=(24,24))
    legend=f'<div style="position:fixed;left:15px;bottom:32px;z-index:9999;background:white;padding:10px 14px;font:14px Arial;color:#193e37;border:1px solid #bfcbc4">Trip {trip_id} &nbsp; <b style="color:{GOLD}">● GPS</b> &nbsp; <b style="color:#714aa0">━ Reference</b> &nbsp; <b style="color:{TEAL}">┄ Predicted</b> &nbsp; <span style="color:#788883">━ Roads</span></div>'
    m.get_root().html.add_child(Element(legend));m.save(str(out/"example_map.html"))
    # Project the very same Folium layers into a static north-up export for print.
    lat0,lon0=np.mean(gps,axis=0); scale=np.deg2rad(1)*6371.0088
    def xy(arr):
        arr=np.asarray(arr)
        return np.stack(((arr[...,1]-lon0)*scale*np.cos(np.deg2rad(lat0)),(arr[...,0]-lat0)*scale),axis=-1)
    chart_setup()
    fig,axs=plt.subplots(1,2,figsize=(9.1,4.0),constrained_layout=True)
    gp=xy(gps)
    for index,ax in enumerate(axs):
        ax.set_facecolor("#f4f5ed");ax.add_collection(LineCollection(xy(layers["roads"]),colors="#afbeb7",linewidths=.65))
        ax.add_collection(LineCollection(xy(ref),colors="#7055a6",linewidths=3.2,label="Reference"))
        for j,line in enumerate(layers["predicted"]):
            a=xy(line);ax.plot(a[:,0],a[:,1],ls="--",lw=1.5,color=TEAL,label="Predicted" if j==0 else None)
        ax.scatter(gp[:,0],gp[:,1],s=2,color=GOLD,alpha=.8,label="GPS observations")
        if index==0:
            span=max(np.ptp(gp[:,0]),np.ptp(gp[:,1]))*.55
            mid=(gp.min(axis=0)+gp.max(axis=0))*.5
            ax.set_title("Full trajectory")
        else:
            mid=gp[len(gp)//2];span=max(.12,.105*max(np.ptp(gp[:,0]),np.ptp(gp[:,1])))
            ax.set_title("Mid-trip detail")
        ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),aspect="equal",xlabel="East (km)",ylabel="North (km)")
    axs[0].legend(loc="lower left",fontsize=7,framealpha=.93)
    fig.savefig(out/"figures"/"route_map.png",dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data"));p.add_argument("--out",type=Path,default=Path("outputs"));p.add_argument("--trip",default="00000052")
    a=p.parse_args();charts(a.out);spatial(a.data,a.out,a.trip)
if __name__=="__main__": main()

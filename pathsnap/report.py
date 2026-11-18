"""Rebuild the consolidated PDF from the experiment outputs and captured demo figures."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from .models import DEFAULTS
W,H=A4
M=42
CW=W-2*M
INK="#193d37"; BG="#f2f5ef"; TEAL="#087a64"; SUB="#50645e"; GOLD="#ad572c"; LIGHT="#d4dfd5"
def color(s):return colors.HexColor(s)
def pc(v):return f"{100*v:.2f}%"
def f4(v):return f"{v:.4f}"
def num(v):return f"{int(v):,}"


class Page:
    def __init__(self,path):
        self.canvas=Canvas(str(path),pagesize=A4)
        self.canvas.setTitle("PathSnap | Map matching and route reconstruction")
        self.canvas.setAuthor("PathSnap project • AI-assisted experiment report")
        self.n=0
    def start(self,section,title,sub=None):
        self.n+=1;c=self.canvas
        c.setFillColor(color(INK));c.rect(0,H-12,W,12,stroke=0,fill=1)
        c.setFont("Helvetica-Bold",8.5);c.setFillColor(color(TEAL));c.drawString(M,H-42,"PATHSNAP  /  TECHNICAL REPORT")
        c.setFont("Helvetica",8.5);c.setFillColor(color(SUB));c.drawRightString(W-M,H-42,section.upper())
        self.y=H-65
        self.text(title,26,30,"Helvetica-Bold",INK,8)
        if sub:self.text(sub,11,15,"Helvetica",SUB,10)
    def text(self,s,size=10,lead=14,font="Helvetica",ink=INK,space=7,x=None,width=None):
        p=Paragraph(s,ParagraphStyle("body",fontName=font,fontSize=size,leading=lead,textColor=color(ink),alignment=TA_LEFT))
        w,h=p.wrap(width or CW,750)
        p.drawOn(self.canvas,M if x is None else x,self.y-h)
        self.y-=h+space
    def heading(self,s):self.text(s,13,16,"Helvetica-Bold",INK,4)
    def caption(self,s):self.text(s,8.5,11,"Helvetica",SUB,9)
    def table(self,heads,rows,widths,size=8.6):
        def para(v,head=False):
            return Paragraph(escape(str(v)),ParagraphStyle("t",fontName="Helvetica-Bold" if head else "Helvetica",
                              fontSize=size,leading=size+2.3,textColor=colors.white if head else color(INK)))
        contents=[[para(h,True) for h in heads]]+[[para(v) for v in r] for r in rows]
        tab=Table(contents,colWidths=[v*CW for v in widths],hAlign="LEFT")
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),color(INK)),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,color(BG)]),
                                 ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),
                                 ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),4)]))
        _,h=tab.wrap(CW,900);tab.drawOn(self.canvas,M,self.y-h);self.y-=h+10
    def fig(self,path,maxh,full=False):
        im=ImageReader(str(path));w,h=im.getSize();s=min(CW/w,maxh/h)
        dw,dh=w*s,h*s;self.canvas.drawImage(im,(W-dw)/2,self.y-dh,dw,dh,mask="auto");self.y-=dh+6
    def end(self):
        if self.y<33:raise ValueError(f"Report page {self.n} overflows: {self.y}")
        c=self.canvas;c.setStrokeColor(color(LIGHT));c.line(M,31,W-M,31);c.setFont("Helvetica",8);c.setFillColor(color(SUB))
        c.drawString(M,17,"PES UNIVERSITY  /  MACHINE LEARNING");c.drawRightString(W-M,17,f"{self.n:02d}  /  08");c.showPage()


def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,default=Path("outputs"));a.add_argument("--pdf",type=Path,default=Path("PathSnap_Technical_Report.pdf"))
    args=a.parse_args();out=args.out;fig=out/"figures";p=Page(args.pdf)
    inv=pd.read_csv(out/"inventory.csv",dtype={"trip":str}); t=pd.read_csv(out/"model_comparison.csv").set_index("model")
    tune=pd.read_csv(out/"tuning_comparison.csv");trials=pd.read_csv(out/"bayesian_trials.csv");dec=pd.read_csv(out/"decoder_summary.csv")
    det=pd.read_csv(out/"decoder_per_trip.csv",dtype={"trip":str});api=pd.read_csv(out/"api_checks.csv",dtype={"trip":str})
    sel=dec[dec.experiment=="Selected"].set_index("mode");best=sel.model.iloc[0]
    a,b=sel.loc["Independent"],sel.loc["Viterbi"]; train=inv[inv.split=="train"];test=inv[inv.split=="student_test"]
    config=json.loads((Path("models")/"config.json").read_text())
    p.start("01  /  Findings","GPS map matching and<br/>route reconstruction",
            "PES University machine-learning hackathon  /  18 November 2026")
    p.text("A GPS observation can fall between a pair of parallel roads. PathSnap first scores the nearby directed road segments, then uses the road network and the preceding observations to choose a route.",13,18,space=15)
    p.table(["Student-test measure","Independent "+best,best+" + Viterbi"],[
        ["Top-1 agreement with aligned reference",pc(a.top1),pc(b.top1)],["Strict adjacent-arc continuity",pc(a.continuity),pc(b.continuity)],
        ["Ordered route F1",f4(a.ordered_f1),f4(b.ordered_f1)]],[.46,.255,.285],10)
    p.text(f"The final system agreed with the derived road label for <b>{num(det[(det.experiment=='Selected')&(det['mode']=='Viterbi')].correct.sum())} of {num(test.observations.sum())} GPS observations</b>. The decoder improved Top-1 by {(b.top1-a.top1)*100:.2f} percentage points. The reference supplies the order in which roads were traversed, so each GPS-to-road label is an alignment estimate.",11,16,space=15)
    p.heading("Experiment scope")
    p.text("Five models were fitted: XGBoost, LightGBM, Random Forest, distance-weighted KNN and a Hidden Markov Model (HMM). The HMM includes a trained observation scorer. Its standalone decoded route and its pre-decoding scores are both reported.")
    p.text("Model selection and Bayesian tuning used grouped cross-validation on the 80 training trips. The ten supplied student-test trips provided development results. The ten organiser-held evaluator trips were unavailable.")
    p.heading("Application and authorship")
    p.text("Two local services reproduced the intended two-laptop workflow. The browser sends a trip ID, and FastAPI loads the model, predicts a route, then optionally scores it using server-side references.")
    p.text("AI assistance was used to implement, execute and document this experiment. The included data inventory, fold assignments, predictions, trials and request records allow the team to reproduce and examine the findings.")
    p.end()
    p.start("02  /  Dataset","What the supplied trips contain")
    rows=[]
    for name,extract in [
      ("Trip folders",lambda g:len(g)),("GPS observations",lambda g:num(g.observations.sum())),("Observations per trip",lambda g:f"{num(g.observations.min())} - {num(g.observations.max())}"),
      ("Road nodes / directed arcs",lambda g:f"{num(g.nodes.sum())} / {num(g.arcs.sum())}"),("Missing GPS cells / missing node cells",lambda g:f"{g.missing_track_values.sum()} / {g.missing_node_values.sum()}"),
      ("Unusable arcs excluded / isolated nodes",lambda g:f"{g.invalid_or_zero_arcs.sum()} / {g.isolated_nodes.sum()}"),
      ("Weak components per road network",lambda g:f"{g.weak_components.min()} - {g.weak_components.max()}"),
      ("One-second time steps / gaps above 30 s",lambda g:f"{pc(g.one_sec_steps.sum()/g.steps.sum())} / {g.gaps_over_30s.sum()}"),
      ("Largest time gap",lambda g:f"{num(g.max_gap_s.max())} s"),("Reference missing from top 8",lambda g:f"{num(g.candidate_misses.sum())} ({pc(g.candidate_misses.sum()/g.observations.sum())})")]:
        rows.append([name,extract(train),extract(test)])
    p.table(["Audit measure","Training","Student test"],rows,[.47,.265,.265],9)
    p.fig(fig/"eda.png",157)
    p.caption("Each dot represents one trip. Two training trips combine large reference offsets with frequent candidate misses. The inventory provides per-trip file sizes, geographic bounds and graph counts.")
    p.heading("Cleaning and geometry")
    p.text("Track rows store longitude, latitude and seconds. Node IDs and directed arc IDs are zero-based row indices. Original IDs are preserved when invalid arcs are excluded. The loader filters invalid coordinates and non-finite times, sorts valid observations by time and retains original row indices. All supplied observations survived. The audit found zero duplicate or reordered timestamps.")
    p.text("Distances use a local metric tangent-plane approximation centred on each trip. The longitude scale includes the cosine of its median latitude. Long time gaps split independent trajectory sections.")
    p.end()
    p.start("03  /  Features and references","How a GPS point gets a road label")
    p.heading("Candidate search and feature preparation")
    p.text("A KD-tree indexes midpoints of road pieces at most 35 m long. Each observation is projected onto nearby line segments. Search expands until an unseen segment cannot beat the eighth candidate. Both directions of a two-way road remain separate candidates. If a network has fewer than eight usable arcs, all are retained.")
    p.table(["Feature group","What the scorer can learn"],[
        ["Distance and excess over nearest","How close the observation lies to this segment in metres and how much closer another segment is."],
        ["Three heading cosines and speed","Incoming, outgoing and centred motion distinguish opposite directions. Up to four observations on either side stay inside the same time block."],
        ["Road length, endpoint margin, fraction","Projection position distinguishes a mid-road fix from one at a junction. Log transforms compress long-tailed distance, length and speed values."]],[.35,.65],9)
    p.heading("Deriving the development labels")
    p.text("The .route file lists traversed arcs in order. A dynamic program aligns observations to nondecreasing positions in that sequence, minimising the sum of log(1 + point-to-segment distance). Repeated observations may share a road, and a time gap may skip route entries. The alignment uses the complete reference sequence independently of the candidate shortlist.")
    p.text(f"Trips 54 and 33 account for {num(train.nlargest(2,'candidate_misses').candidate_misses.sum())} of {num(train.candidate_misses.sum())} training misses. Their median offsets exceed 120 m. All misses remain in the observation-level denominator. Apparent errors near junctions can also reflect an ambiguous alignment.")
    p.fig(fig/"folium.png",126)
    p.fig(fig/"route_map.png",142)
    p.caption("Student-test trip 00000052. The upper image shows the interactive Folium view. The lower plots show the full trajectory and enlarged detail from those same four layers. Purple shows the reference, green the decoded route, orange GPS and grey the roads. The interactive version is available in the private download or through the map-generation command.")
    p.end()
    p.start("04  /  Model comparison","Five models under the same split")
    p.text(f"Seed 817 assigns whole trips to three fixed folds (27, 27 and 26). Uniform sampling without replacement caps fitting at 260 observations per trip and validation at 450 per held-out trip. A final fit uses {num(config['fit_observations'])} observations across all 80 training trips. All candidates for an observation stay together. KNN scales features only on its fitting subset.")
    rows=[]
    for n,r in t.iterrows():
        rows.append([n if n!="HMM" else "HMM scorer",f4(r.cv_macro_f1),pc(r.top1),f4(r.macro_f1),f4(r.observation_macro_f1),pc(r.continuity),f4(r.log_loss),f"{r.inference_s:.3f}"])
    p.table(["Candidate model","CV macro F1","Test Top-1","Test cand. F1","Test arc F1","Strict cont.","Log loss","Score time (s)"],rows,[.17,.12,.105,.125,.12,.12,.105,.135],8)
    p.fig(fig/"comparison.png",155)
    p.caption("CV error bars show fold standard deviation. HMM* refers to the pre-decoding observation scorer. Score time covers candidate scoring across all ten student-test trips and excludes feature extraction and Viterbi.")
    p.heading("Which score answers which question?")
    p.text("<b>Candidate macro F1</b> averages match and nonmatch F1 at a fixed 0.5 threshold. It is the primary selection score. <b>Top-1</b> checks the highest-scoring arc per observation. <b>Test arc F1</b> macro-averages over local directed arc IDs within each trip, then averages the ten trip scores. Log loss uses the binary candidate probabilities.")
    p.text("<b>Strict continuity</b> counts consecutive observation pairs on the same arc or directly joined directed arcs, excluding time gaps over 30 s. <b>Ordered route F1</b> computes twice the exact longest common subsequence divided by the combined lengths of the two arc sequences after consecutive duplicates are collapsed.")
    p.fig(fig/"confusion.png",120)
    p.caption("Student-test candidate confusion matrices. Rows show alignment-based reference classes, columns show predictions. Raw counts are printed in each cell. Colour is normalised within each row.")
    p.end()
    p.start("05  /  Individual experiments","What each model learned")
    descriptions={
      "XGBoost":("Depth-limited histogram boosting","70 rounds, depth 5, learning rate 0.10. Binary logistic loss. Successive trees can correct local combinations of distance and heading.","Its test log loss was close to LightGBM. Depth-limited trees offered fast scoring, but local scores alone could switch between parallel roads."),
      "LightGBM":("Leaf-wise histogram boosting","70 rounds, 23 leaves, learning rate 0.09, minimum 45 rows per leaf. Binary log loss. Leaf-wise splits can concentrate on junctions where geometry and heading interact.","It had the highest mean CV candidate macro F1. Its small lead over XGBoost was within the fold variability. Random Forest edged it on student-test Top-1."),
      "Random Forest":("Bagged classification trees","40 trees, depth 15, minimum 6 samples per leaf, 80% of features considered at each split. Gini impurity governs splits. Averaging trees reduces instability.","It had the highest student-test candidate macro F1. Its larger fit-to-test gap is consistent with some memorisation of the fitting sample."),
      "KNN":("Distance-weighted neighbours","21 neighbours, Euclidean distance and training-fold standardisation. Voting by inverse distance estimates the positive fraction. The model stores examples and has no iterative loss function.","Training-sample queries include their own examples. Perfect in-sample F1 and lower development F1 reflect memorisation. Neighbour lookup was the slowest scoring method."),
      "HMM":("Probabilistic observation and transition model","20 quantile bins, smoothing 1.0, likelihood weight 0.8. Match and background histograms estimate distance, heading and endpoint evidence. A separate fitted distance-error scale defines transitions.","Its emission-only fit and test scores were close, but below the tree models. The simple marginal likelihoods miss feature interactions. Its standalone decoded result is reported on page 7.")}
    for n,r in t.iterrows():
        fam,setup,explain=descriptions[n];p.heading(n+"  /  "+fam)
        p.text(setup,9.5,13,space=4)
        p.text(f"Fit F1 <b>{r.train_macro_f1:.4f}</b> · Test candidate F1 <b>{r.macro_f1:.4f}</b> · Top-1 <b>{pc(r.top1)}</b> · Log loss <b>{r.log_loss:.4f}</b>.",9.4,13,space=3)
        p.text(explain,9.5,13,space=11)
    p.caption("Fit F1 is evaluated on the rows used to fit the model. Test F1 is evaluated on every student-test candidate. Detailed positive-class precision, recall, candidate accuracy and fit times are recorded in model_comparison.csv.")
    p.end()
    p.start("06  /  Bayesian optimisation","Did tuning improve LightGBM?")
    p.text("The search maximised mean candidate macro F1 on the same three held-out training-trip folds. The ten student-test trips played no role in the search objective, trial proposals or choice of the exported configuration.")
    p.table(["Parameter","Range sampled","Reason"],[
        ["Boosting rounds","40-140, integer","Controls the number of successive corrections."],["Number of leaves","12-55, integer","Controls the complexity of each tree."],
        ["Learning rate","0.035-0.18, log scale","Controls how much each new tree changes the scores."],["Minimum samples per leaf","18-100, integer","Controls how strongly rare road configurations can influence a split."]],[.32,.29,.39],9.3)
    p.text("Eight configurations include the initial model, two Sobol space-filling proposals and five proposals chosen by expected improvement under a Gaussian-process surrogate with a Matérn kernel. The baseline's existing CV score was reused. Each new trial retrained and evaluated three folds.")
    p.fig(fig/"bayes.png",142)
    p.caption(f"The seven additional configurations took {trials.seconds.sum():.1f} seconds of measured search time on this run. Another machine can produce different timings.")
    rows=[]
    for _,r in tune.iterrows():
        rows.append([r.stage,f4(r.cv_macro_f1),pc(r.top1),f4(r.macro_f1),f4(r.log_loss),f"{r.inference_s:.3f}"])
    p.table(["Configuration","CV macro F1","Test Top-1","Test macro F1","Log loss","Score time (s)"],rows,[.22,.16,.15,.17,.14,.16],9)
    p.heading("Selected configuration")
    p.text("The initial configuration retained first place: 70 rounds, 23 leaves, learning rate 0.09 and 45 minimum rows per leaf. Trial 6 approached its CV score. The final refit gave the same Top-1, F1 and log loss. Differences in measured time reflect runtime variation.")
    p.caption("Reproduction files: outputs/folds.json, outputs/bayesian_trials.csv, outputs/tuning_comparison.csv and models/config.json.")
    p.end()
    p.start("07  /  Route reconstruction","What Viterbi changed")
    p.text("A hidden state is one shortlisted directed arc at one GPS time. Emission scores come from the selected classifier, or from the HMM's own trained probability tables for the standalone experiment. Each set of candidate scores is normalised and expressed in log space.")
    p.text(f"A transition compares GPS displacement with projected travel along the previous arc, a directed shortest-path bridge and the current arc. A negative absolute distance error scaled by {config['beta_m']:.1f} m defines its log weight. The scale is the mean clipped training-route residual with a 6 m floor. Same-arc backward jitter up to 15 m is tolerated. Across candidates, admissible transition weights are normalised per source arc.")
    p.text("Viterbi keeps the best cumulative log score for each candidate and backtracks its predecessor. Infeasible transitions have zero probability. Gaps above 30 s or a step with no reachable successor start a new section. A bridge search is bounded at 400 m, with a tighter step-specific travel limit. The output inserts verified intermediate road arcs and preserves section breaks on the map.")
    table=[]
    for _,r in dec.iterrows():
        table.append([r.model+" scorer" if r.model=="HMM" else r.model,r["mode"],pc(r.top1),pc(r.continuity),f4(r.ordered_f1),f4(r.observation_macro_f1)])
    p.table(["Emission source","Decoder","Top-1","Strict continuity","Ordered F1","Arc macro F1"],table,[.21,.18,.13,.18,.15,.15],9)
    p.fig(fig/"decoder.png",156)
    p.heading("Two kinds of route error")
    p.text("Trips 7 and 40 lost point-level agreement after decoding. On trip 7, Top-1 fell from 96.16% to 89.00% while ordered F1 rose from 0.763 to 0.988. A model can recover the order of the roads while assigning a transition to the neighbouring timestamp. Trip 77 retained the lowest decoded Top-1, 81.95%. These results motivate separate pointwise and sequence metrics.")
    p.caption("Ordered F1 uses the actual graph bridges for decoded predictions. Missing references, road-graph gaps and course-provided route ambiguity limit the interpretation of any pointwise score.")
    p.end()
    p.start("08  /  System demonstration","From trip selection to API response")
    p.text(f"Two local services exercised the laptop roles. The browser served the trip selector separately from FastAPI. The API recomputed every route. All {len(api)} student-test requests returned HTTP 200, and every arc ID matched the offline prediction. An unknown trip returned HTTP 404.",10,14,space=8)
    p.fig(fig/"ui.png",229)
    p.caption("Working evaluator interface. The browser renders GPS observations, the network and the predicted route. It receives aggregate scores only. The reference overlay is available in the separate student-development Folium map.")
    p.fig(fig/"server.png",137)
    p.caption("Actual receiving-server terminal from the local run. Each request logs the trip ID, observation count, number of sections, scoring status, Top-1 and inference time.")
    p.heading("Two-laptop deployment")
    p.text("Laptop 1 starts Uvicorn with the exported model, trip folder and optional private references. Laptop 2 serves the static UI, connects to Laptop 1's LAN address and selects an ID. Both roles run offline with bundled map assets. The README gives the PowerShell commands, evaluator folder layout and live verification command.")
    p.heading("What remains for the on-site demonstration")
    p.text("The final live check needs the organisers' ten reserved trips and two physical laptops. The evidence here covers the supplied development split and separate local services. No hidden-set score is claimed.")
    p.caption("Project evidence: outputs/api_checks.csv, outputs/api_checks.txt, outputs/server_console.txt, the per-trip predictions and the map-generation source.")
    p.end()
    p.canvas.save()

if __name__=="__main__":main()

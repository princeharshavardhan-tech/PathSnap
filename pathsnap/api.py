import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .service import infer, load_model, map_layers, score_after_prediction, strict_continuity

log=logging.getLogger("uvicorn.error")
ROOT=Path(__file__).resolve().parents[1]


def build_api():
    data=Path(os.environ.get("PATHSNAP_TRIPS",ROOT/"data/student_test"))
    refs=Path(os.environ.get("PATHSNAP_REFERENCES",data))
    mode=os.environ.get("PATHSNAP_MODE","student_test")
    if mode not in ("student_test","evaluator"):
        raise ValueError("PATHSNAP_MODE must be student_test or evaluator")
    def listed():
        if not data.is_dir(): return []
        return sorted(p.name for p in data.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]+",p.name)
                      and all((p/(p.name+"."+ext)).is_file() for ext in ("nodes","arcs","track")))
    @asynccontextmanager
    async def lifespan(app):
        app.state.model,app.state.config=load_model(os.environ.get("PATHSNAP_MODEL_DIR",ROOT/"models"))
        if len(listed())!=10:
            raise RuntimeError(f"Expected ten complete {mode} trips in {data}; found {len(listed())}")
        yield
    app=FastAPI(title="PathSnap",version="1.0",lifespan=lifespan)
    app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET","POST"],allow_headers=["*"])
    @app.get("/health")
    def health():
        return {"status":"ready","model":app.state.config["model"],"split":mode}
    @app.get("/trips")
    def trips():
        return {"split":mode,"trips":listed()}
    @app.post("/match/{trip_id}")
    def match(trip_id:str):
        if trip_id not in listed(): raise HTTPException(404,"Unknown trip")
        start=time.perf_counter()
        try:
            tr,ids,chosen,breaks,paths=infer(data/trip_id,app.state.model,app.state.config)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        pred_seconds=time.perf_counter()-start
        ref=refs/trip_id/(trip_id+".route")
        if not ref.is_file(): ref=refs/(trip_id+".route")
        evaluation=None
        reference_status="unavailable"
        if ref.is_file():
            try:
                evaluation=score_after_prediction(tr,chosen,paths,ref)
                reference_status="scored"
            except ValueError:
                reference_status="invalid"
                log.warning("trip=%s reference could not be aligned",trip_id)
        log.info("trip=%s observations=%d segments=%d scoring=%s top1=%s inference_s=%.3f",
                 trip_id,len(chosen),len(paths),reference_status,
                 f"{evaluation['top1']:.4f}" if evaluation else "unavailable",pred_seconds)
        return dict(trip=trip_id,split=mode,model=app.state.config["model"],observations=len(chosen),
                    prediction_s=pred_seconds,segments=len(paths),strict_continuity=strict_continuity(tr,chosen),
                    evaluation=evaluation,reference_status=reference_status,source_rows=tr.source_rows.tolist(),
                    predicted_arc_ids=chosen.tolist(),break_rows=np.flatnonzero(breaks).tolist(),
                    layers=map_layers(tr,ids,paths))
    return app

app=build_api()

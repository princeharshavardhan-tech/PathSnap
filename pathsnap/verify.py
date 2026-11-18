"""Smoke-test the live server and optionally compare it to the offline predictions."""
import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen, Request


def call(base, path, method="GET"):
    with urlopen(Request(base.rstrip("/")+path,method=method),timeout=120) as response:
        return response.status,json.load(response)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--api",default="http://127.0.0.1:8000");p.add_argument("--offline",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    status,health=call(args.api,"/health"); assert status==200
    _,body=call(args.api,"/trips"); assert len(set(body["trips"]))==10
    print(f'{health["split"]} | {health["model"]} | {len(body["trips"])} trips',flush=True)
    rows=[]
    for id in body["trips"]:
        status,r=call(args.api,"/match/"+id,"POST")
        assert status==200 and r["trip"]==id
        assert len(r["source_rows"])==len(r["predicted_arc_ids"])==len(r["layers"]["gps"])==r["observations"]
        assert len(r["layers"]["predicted"])==r["segments"] and set(r["layers"])=={"roads","gps","predicted"}
        assert set(r["evaluation"] or {}) <= {"top1","ordered_f1"}
        if args.offline:
            expected=json.loads((args.offline/(id+".json")).read_text())
            assert r["predicted_arc_ids"]==expected["predicted_arc_ids"], id
        ev=r["evaluation"] or {}
        rows.append(dict(trip=id,status=status,observations=r["observations"],top1=ev.get("top1"),
                         ordered_f1=ev.get("ordered_f1"),continuity=r["strict_continuity"],sections=r["segments"],
                         prediction_s=round(r["prediction_s"],4)))
        score=f'{ev["top1"]:.2%}' if ev else "unavailable"
        print(f'POST /match/{id}   HTTP {status}   {r["observations"]:4d} GPS points   Top-1 {score}',flush=True)
    try:call(args.api,"/match/unknown","POST")
    except HTTPError as exc:assert exc.code==404
    else:raise AssertionError("Unknown trip should return 404")
    if args.out:
        with args.out.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print("Passed all ten requests and unknown-trip handling.",flush=True)

if __name__=="__main__":main()

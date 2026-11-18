"""Generate reproducible feature caches and a data-quality inventory from the supplied files."""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .data import read_inputs, candidates, GAP, valid_lonlat
from .reference import align


def process(args):
    part, folder, cache = args
    tr = read_inputs(folder)
    ids, x = candidates(tr)
    ref_path = folder / (folder.name + ".route")
    route, ref, dist, residual = align(tr, ref_path)
    edges = tr.edges[tr.edge_ids]
    graph = coo_matrix((np.ones(len(edges), bool), edges.T), shape=(len(tr.nodes),)*2)
    components = connected_components(graph, directed=True, connection="weak", return_labels=False)
    degree = np.bincount(edges.ravel(), minlength=len(tr.nodes))
    dt = np.diff(tr.track[:, 2])
    np.savez(cache/part/(folder.name+".npz"), ids=ids, x=x, ref=ref, route=route,
             t=tr.track[:, 2], endpoints=tr.edges[ids], residual_sum=np.minimum(100,residual).sum(),
             residual_count=len(residual))
    files = [folder/(folder.name + "."+ext) for ext in ("track","nodes","arcs","route")]
    return dict(split=part, trip=folder.name, observations=len(tr.track), raw_rows=tr.raw_count,
                dropped=tr.raw_count-len(tr.track), missing_track_values=tr.missing,
                missing_node_values=int((~np.isfinite(tr.nodes)).sum()), invalid_nodes=int((~valid_lonlat(tr.nodes)).sum()),
                invalid_or_zero_arcs=len(tr.edges)-len(tr.edge_ids), time_duplicates=int((dt==0).sum()),
                original_time_reorders=int((np.diff(tr.source_rows)<0).sum()), one_sec_steps=int((dt==1).sum()),
                steps=len(dt), gaps_over_30s=int((dt>GAP).sum()), max_gap_s=float(dt.max(initial=0)),
                nodes=len(tr.nodes), arcs=len(tr.edges), isolated_nodes=int((degree==0).sum()),
                weak_components=int(components), route_entries=len(route), usable_files=len(files),
                bytes_total=sum(p.stat().st_size for p in files), aligned_distance_p50=float(np.median(dist)),
                aligned_distance_p95=float(np.percentile(dist,95)), candidate_misses=int((~(ids==ref[:,None]).any(1)).sum()),
                reference_discontinuities=int((tr.edges[route[:-1],1]!=tr.edges[route[1:],0]).sum()),
                lon_min=tr.track[:,0].min(),lat_min=tr.track[:,1].min(),lon_max=tr.track[:,0].max(),lat_max=tr.track[:,1].max())


def folders(root, split):
    return sorted(p for p in (root/split).iterdir() if p.is_dir() and (p/(p.name+".track")).is_file())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True,help="Folder containing train and student_test")
    p.add_argument("--cache",type=Path,default=Path("cache"))
    p.add_argument("--out",type=Path,default=Path("outputs"))
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    jobs=[]
    for part,expected in [("train",80),("student_test",10)]:
        (args.cache/part).mkdir(parents=True,exist_ok=True)
        paths=folders(args.data,part)
        if len(paths)!=expected:
            raise ValueError(f"{part}: expected {expected} trips, found {len(paths)}")
        jobs.extend((part,path,args.cache) for path in paths)
    if {j[1].name for j in jobs[:80]} & {j[1].name for j in jobs[80:]}:
        raise ValueError("Trip IDs overlap between train and student_test")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[]
        for row in pool.map(process,jobs):
            rows.append(row)
            if len(rows)%10==0: print(f"Prepared {len(rows)}/{len(jobs)} trips",flush=True)
    pd.DataFrame(rows).to_csv(args.out/"inventory.csv",index=False)

if __name__=="__main__":
    main()

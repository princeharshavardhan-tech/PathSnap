"""A directed, distance-based HMM. Route bridges are recovered from graph edges."""
from collections import defaultdict
from heapq import heappush, heappop
from bisect import bisect_left
import numpy as np
from scipy.special import logsumexp
from .data import GAP


class RoadGraph:
    def __init__(self, trip):
        self.edges = trip.edges
        self.out = defaultdict(list)
        self.length = np.zeros(len(trip.edges))
        valid = trip.edge_ids
        e = trip.edges[valid]
        self.length[valid] = np.linalg.norm(trip.xy[e[:,1]]-trip.xy[e[:,0]],axis=1)
        for arc, (a,b) in zip(valid,e):
            self.out[int(a)].append((int(b),self.length[arc],int(arc)))
        self.cache = {}

    def within(self, start, radius=400.):
        if start in self.cache:
            return self.cache[start]
        d = {start:0.}
        back = {}
        todo = [(0.,start)]
        while todo:
            cost, node = heappop(todo)
            if cost != d[node]:
                continue
            for target, w, arc in self.out[node]:
                new = cost+w
                if new<=radius and new<d.get(target,np.inf):
                    d[target] = new
                    back[target] = (node,arc)
                    heappush(todo,(new,target))
        self.cache[start]=(d,back)
        return d,back

    def bridge(self, a, b):
        start, end = map(int,(self.edges[a,1],self.edges[b,0]))
        if start==end:
            return []
        d,back = self.within(start)
        if end not in d:
            return None
        route=[]
        while end!=start:
            end,edge=back[end]
            route.append(edge)
        return route[::-1]


def decode(trip, ids, features, probabilities, beta):
    """Return one state per valid GPS row, break flags, and intervening road arcs."""
    graph=RoadGraph(trip)
    n,k=ids.shape
    obs=np.arange(n)
    frac=features[:,:,-1].astype(float)
    lengths=graph.length[ids]
    dt=np.diff(trip.track[:,2])
    steps=np.linalg.norm(np.diff(trip.gps,axis=0),axis=1)
    emit=np.log(np.clip(probabilities,1e-7,1.))
    emit-=logsumexp(emit,axis=1)[:,None]
    parent=np.full((n,k),-1,np.int32)
    state=np.empty(n,np.int32)
    breaks=np.zeros(n,bool)
    breaks[0]=True
    value=emit[0].copy()
    max_arc=len(trip.edges)
    for begin in range(1,n,192):
        end=min(n,begin+192)
        prev,curr=ids[begin-1:end-1],ids[begin:end]
        # Encode the repeated arc pairs once per chunk, then look up bounded network distances.
        codes=prev[:,:,None].astype(np.int64)*max_arc+curr[:,None,:]
        pairs,inv=np.unique(codes.ravel(),return_inverse=True)
        p,c=np.divmod(pairs,max_arc)
        between=np.full(len(pairs),np.inf)
        for j,(pa,ca) in enumerate(zip(p,c)):
            u,v=int(trip.edges[pa,1]),int(trip.edges[ca,0])
            if u==v:
                between[j]=0.
            elif pa!=ca:
                between[j]=graph.within(u)[0].get(v,np.inf)
        net=((1-frac[begin-1:end-1])*lengths[begin-1:end-1])[:,:,None] + between[inv].reshape(codes.shape) + (frac[begin:end]*lengths[begin:end])[:,None,:]
        same=prev[:,:,None]==curr[:,None,:]
        forward=(frac[begin:end,None,:]-frac[begin-1:end-1,:,None])*lengths[begin:end,None,:]
        direct=np.where(forward>=-15.,np.maximum(0,forward),np.inf)
        net=np.where(same,direct,net)
        step=steps[begin-1:end-1,None,None]
        score=-np.abs(net-step)/beta
        score[net>np.minimum(400.,np.maximum(65.,3*step+25.))]=-np.inf
        norm=logsumexp(score,axis=2,keepdims=True)
        log_trans=score-np.where(np.isfinite(norm),norm,0.)
        for j,i in enumerate(range(begin,end)):
            transition=value[:,None]+log_trans[j]
            if dt[i-1]>GAP or not np.isfinite(transition).any():
                state[i-1]=np.argmax(value)
                breaks[i]=True
                value=emit[i].copy()
            else:
                parent[i]=np.argmax(transition,axis=0)
                value=emit[i]+np.max(transition,axis=0)
                value-=np.max(value)
    state[-1]=np.argmax(value)
    for i in range(n-1,0,-1):
        if not breaks[i]:
            state[i-1]=parent[i,state[i]]
    selected=ids[obs,state]
    segments=[]
    for i,arc in enumerate(selected):
        arc=int(arc)
        if breaks[i]:
            segments.append([arc])
        elif selected[i-1]!=arc:
            bridge=graph.bridge(int(selected[i-1]),arc)
            if bridge is None:
                raise RuntimeError("Decoded transition has no directed bridge")
            segments[-1].extend(bridge+[arc])
    return selected,breaks,segments


def collapse(seq):
    a=np.asarray(seq,dtype=np.int32)
    return a[np.r_[True,a[1:]!=a[:-1]]] if len(a) else a


def ordered_f1(pred, route):
    """Exact LCS for two directed arc sequences, using ranked matching positions."""
    a,b=collapse(pred),collapse(route)
    pos=defaultdict(list)
    for i,arc in enumerate(b):
        pos[int(arc)].append(i)
    tail=[]
    for arc in a:
        for idx in reversed(pos.get(int(arc),())):
            j=bisect_left(tail,idx)
            if j==len(tail):
                tail.append(idx)
            else:
                tail[j]=idx
    return 2*len(tail)/max(1,len(a)+len(b))

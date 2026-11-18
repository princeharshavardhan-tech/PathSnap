"use strict";
const el = id => document.getElementById(id);
const map = L.map("map", {preferCanvas:true, attributionControl:false}).setView([49,9],6);
L.control.scale({imperial:false}).addTo(map);
const road = L.layerGroup().addTo(map), gps = L.layerGroup().addTo(map), route = L.layerGroup().addTo(map);
L.control.layers(null, {"Road network":road,"GPS observations":gps,"Predicted route":route}, {collapsed:true}).addTo(map);
const apiInput=el("api"), trip=el("trip"), run=el("run");
apiInput.value=new URLSearchParams(location.search).get("api") || (location.protocol+"//"+(location.hostname || "localhost")+":8000");
let server="", lastTrip="", request=0;
const fmt = value => value===null || value===undefined ? "—" : (value*100).toFixed(1)+"%";
function text(id,value){el(id).textContent=value;}
function clearResult(){[road,gps,route].forEach(layer=>layer.clearLayers());["top1","continuity","ordered"].forEach(id=>text(id,"—"));el("details").hidden=true;text("mapid","");text("title","Trip map");}
async function connect(){
 const token=++request; clearResult();run.disabled=true;trip.disabled=true;el("status").className="";text("notice","");el("status").lastElementChild.textContent="Connecting";
 try{
  const address=new URL(apiInput.value);
  if(!["http:","https:"].includes(address.protocol))throw Error("Enter an HTTP or HTTPS server address.");
  const base=address.href.replace(/\/$/,"");
  const response=await fetch(base+"/trips");if(!response.ok)throw Error("The server returned "+response.status);
  const body=await response.json();if(token!==request)return;
  server=base;trip.options.length=0; body.trips.forEach(t=>trip.add(new Option(t,t)));
  text("split",body.trips.length+" "+(body.split==="evaluator"?"evaluator":"student-test")+" trips");
  trip.disabled=!body.trips.length;run.disabled=trip.disabled;el("status").className="ready";el("status").lastElementChild.textContent="Server connected";
 }catch(e){if(token!==request)return;trip.options.length=0; trip.add(new Option("No trips loaded",""));text("split","");el("status").lastElementChild.textContent="Offline";text("notice","Could not connect. Check the server address and connection.");}
}
async function match(){
 const token=++request;lastTrip=trip.value;run.disabled=true;trip.disabled=true;text("notice","Matching GPS observations to the road network…");
 clearResult();
 try{
  const res=await fetch(server+"/match/"+encodeURIComponent(lastTrip),{method:"POST"});if(!res.ok)throw Error("The match request returned "+res.status);
  const body=await res.json();if(token!==request)return;
  L.polyline(body.layers.roads,{color:"#a9b8b2",weight:1.4,opacity:.74,interactive:false}).addTo(road);
  L.polyline(body.layers.predicted,{color:"#067d64",weight:4.5,opacity:.92,interactive:false}).addTo(route);
  body.layers.gps.forEach(p=>L.circleMarker(p,{radius:1.5,stroke:false,fillColor:"#ca642d",fillOpacity:.73,interactive:false}).addTo(gps));
  map.fitBounds(L.latLngBounds(body.layers.gps),{padding:[23,23],maxZoom:17});
  text("title","Matched route");text("mapid","Trip "+body.trip);
  text("model",body.model+" + Viterbi");text("observations",body.observations.toLocaleString());
  text("time",body.prediction_s.toFixed(2)+" s");text("segments",body.segments);
  text("top1",fmt(body.evaluation?.top1));text("continuity",fmt(body.strict_continuity));text("ordered",fmt(body.evaluation?.ordered_f1));
  el("details").hidden=false;text("notice",body.evaluation?"Route matched and scored.":"Route matched. Reference scores unavailable.");
 }catch(e){if(token===request)text("notice",e.message);}
 finally{if(token===request){run.disabled=false;trip.disabled=false;}}
}
el("connect").addEventListener("click",connect);apiInput.addEventListener("keydown",e=>{if(e.key==="Enter")connect();});
run.addEventListener("click",match);trip.addEventListener("change",()=>{clearResult();text("notice","");});connect();

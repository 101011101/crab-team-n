/* ---------- arena ----------
   Drawn around the chassis, not around a point. Every gap is 0.90 m between
   walls, which leaves a 0.56 m planning band -- wide enough that two crabs can
   pass. The first version of this arena had 0.26 m bands and three rectangular
   robots jammed in it permanently. */
var WALLS=[[0,0,6,0],[6,0,6,4],[0,4,6,4],[0,0,0,4],
  [2.4,0.9,2.4,1.6],[2.4,2.5,2.4,2.6],[2.4,3.5,2.4,4.0],
  [2.4,0.9,5.1,0.9]];
/* Two ways through the divider: a side door at y 1.6-2.5, and the main door at
   y 2.6-3.5 which the box sits in. SEALED shuts the side door and the far end
   of the long way round, so the box becomes the only way in. */
var SEAL=[[2.4,1.6,2.4,2.5],[5.1,0.0,5.1,0.9]];
var sealed=false;
function activeWalls(){ return sealed?WALLS.concat(SEAL):WALLS; }
var AW=6.0, AH=4.0, CELL=0.05, SIGHT=0.80, FOV_DEG=45, FOV_HALF=FOV_DEG*Math.PI/360;
var NX=Math.round(AW/CELL), NY=Math.round(AH/CELL);
/* Robot is a rectangle, not a dot: 0.20 long by 0.15 wide. */
var BOT_L=0.20, BOT_W=0.15;
var BOT_RC=Math.hypot(BOT_L/2,BOT_W/2);          /* circumscribed radius */
var PLAN_PAD=BOT_RC+0.045;                        /* keep waypoints off the walls */
var GOAL={x:4.00,y:3.20};
var STARTS=[{x:0.60,y:3.40},{x:0.60,y:2.60},{x:0.90,y:1.90},{x:0.60,y:1.20},{x:1.50,y:3.10}];
var COLS=["#F0703C","#4FC4C4","#C9A227","#8E7CC3","#5FA8D3"];
var UNKNOWN=0, FREE=1, BLOCKED=2;
var BOX0={x:2.40,y:3.05};
/* 0.62 m across a 0.90 m doorway leaves 0.14 m either side -- narrower than the
   0.15 m chassis, so the box seals in physics exactly as it does in the field.
   At 0.34 m the planner called the doorway blocked while crabs squeezed past. */
var box={x:BOX0.x,y:BOX0.y,w:0.34,h:0.62,th:0};

/* ---------- grids ---------- */
var truth=new Uint8Array(NX*NY);      /* free ignoring the box */
var boxCell=new Uint8Array(NX*NY);    /* cells the box occupies right now */
var known=new Uint8Array(NX*NY);
var dist=new Float32Array(NX*NY);     /* wavefront from the goal, as believed */
var frontier=new Uint8Array(NX*NY);
var nFrontier=0, waveMs=0;

function cellFreeOfWalls(ix,iy){
  var x=(ix+0.5)*CELL, y=(iy+0.5)*CELL;
  if(x<PLAN_PAD||y<PLAN_PAD||x>AW-PLAN_PAD||y>AH-PLAN_PAD) return false;
  var AW_=activeWalls();
  for(var w=0;w<AW_.length;w++){
    var W=AW_[w];
    if(segDist(x,y,W[0],W[1],W[2],W[3])<PLAN_PAD) return false;
  }
  return true;
}
var phys=new Uint8Array(NX*NY);
var clear=new Float32Array(NX*NY);     /* metres from the nearest solid thing */
/* Two-pass chamfer transform. A crab with a 30 deg field of view sees almost
   nothing with its nose against a wall, so the planner is told to prefer the
   middle of a corridor: routes are costed through this, and hugging a wall is
   made expensive rather than forbidden. */
function buildClearance(){
  var i,ix,iy,BIG=1e6;
  for(i=0;i<NX*NY;i++) clear[i]= truth[i]===BLOCKED ? 0 : BIG;
  for(iy=0;iy<NY;iy++) for(ix=0;ix<NX;ix++){
    i=iy*NX+ix; if(clear[i]===0) continue;
    var v=clear[i];
    if(ix>0)            v=Math.min(v,clear[i-1]+1);
    if(iy>0)            v=Math.min(v,clear[i-NX]+1);
    if(ix>0&&iy>0)      v=Math.min(v,clear[i-NX-1]+1.4142);
    if(ix<NX-1&&iy>0)   v=Math.min(v,clear[i-NX+1]+1.4142);
    clear[i]=v;
  }
  for(iy=NY-1;iy>=0;iy--) for(ix=NX-1;ix>=0;ix--){
    i=iy*NX+ix; if(clear[i]===0) continue;
    var w=clear[i];
    if(ix<NX-1)           w=Math.min(w,clear[i+1]+1);
    if(iy<NY-1)           w=Math.min(w,clear[i+NX]+1);
    if(ix<NX-1&&iy<NY-1)  w=Math.min(w,clear[i+NX+1]+1.4142);
    if(ix>0&&iy<NY-1)     w=Math.min(w,clear[i+NX-1]+1.4142);
    clear[i]=w;
  }
  for(i=0;i<NX*NY;i++) clear[i]*=CELL;
}
function buildTruth(){
  var A=activeWalls();
  for(var iy=0;iy<NY;iy++) for(var ix=0;ix<NX;ix++){
    var i=iy*NX+ix;
    truth[i]= cellFreeOfWalls(ix,iy)?FREE:BLOCKED;
    var x=(ix+0.5)*CELL, y=(iy+0.5)*CELL, ok=(x>0.02&&y>0.02&&x<AW-0.02&&y<AH-0.02);
    for(var w=0;ok&&w<A.length;w++)
      if(segDist(x,y,A[w][0],A[w][1],A[w][2],A[w][3])<0.03) ok=false;
    phys[i]=ok?1:0;
  }
  buildClearance();
}
var COMFORT=0;   /* extra clearance demanded of an IMAGINED placement */
/* The box is planned round with the SAME margin as a wall. At BOT_RC+0.02 the
   box pad and the wall pad left a one-cell corridor beside a half-pushed box:
   the field called the reward reachable, UNBLOCK never re-opened, and a crab
   spent 100 s trying to thread a gap no chassis can drive. */
var BOX_PAD=PLAN_PAD;
var boxIdx=[], boxSeen=false;
function stampBox(){
  boxCell.fill(0); boxIdx=[];
  var pad=BOX_PAD+COMFORT;
  for(var iy=0;iy<NY;iy++) for(var ix=0;ix<NX;ix++){
    if(rectDist((ix+0.5)*CELL,(iy+0.5)*CELL,box)<pad){ boxCell[iy*NX+ix]=1; boxIdx.push(iy*NX+ix); }
  }
}
function cellState(i){ return truth[i]===BLOCKED?BLOCKED:(boxCell[i]?BLOCKED:FREE); }

/* ---------- observation ---------- */
function observe(b){
  if(b.seeAt!==undefined&&clock<b.seeAt) return;
  if(b.seeAt!==undefined) b.seeAt=clock+0.1;
  var cx=Math.floor(b.x/CELL), cy=Math.floor(b.y/CELL), R=Math.ceil(SIGHT/CELL);
  for(var iy=Math.max(0,cy-R); iy<Math.min(NY,cy+R+1); iy++){
    for(var ix=Math.max(0,cx-R); ix<Math.min(NX,cx+R+1); ix++){
      if(Math.hypot((ix-cx)*CELL,(iy-cy)*CELL)>SIGHT) continue;
      /* a 30 degree cone about the heading; the cell under the crab is always known */
      if((ix!==cx||iy!==cy)&&Math.abs(wrap(Math.atan2(iy-cy,ix-cx)-b.th))>FOV_HALF) continue;
      var steps=Math.max(Math.abs(ix-cx),Math.abs(iy-cy))||1;
      for(var s=1;s<=steps;s++){
        var sx=Math.round(cx+(ix-cx)*s/steps), sy=Math.round(cy+(iy-cy)*s/steps);
        var k=sy*NX+sx; known[k]=cellState(k); rewardSeen(k);
        if(known[k]===BLOCKED){ thicken(sx,sy); break; }  /* no seeing through solid things */
      }
    }
  }
}

/* Seeing a wall face tells a robot more than one cell: it knows its own width,
   so it knows the whole margin around that wall is untraversable. Without this
   the belief barrier is one cell thick while the planning margin is 3.4, and an
   optimistic field threads straight through the unobserved remainder -- a robot
   in a sealed dead end kept believing a route existed. */
var PADC=Math.ceil(PLAN_PAD/CELL);
function thicken(cx,cy){
  for(var dy=-PADC;dy<=PADC;dy++) for(var dx=-PADC;dx<=PADC;dx++){
    var x=cx+dx, y=cy+dy;
    if(x<0||y<0||x>=NX||y>=NY) continue;
    if(dx*dx+dy*dy>PADC*PADC) continue;
    var j=y*NX+x;
    if(truth[j]===BLOCKED) known[j]=BLOCKED;
  }
}

/* ---------- fields ---------- */
var DIRS=[[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1.4142],[1,-1,1.4142],[-1,1,1.4142],[-1,-1,1.4142]];
/* Dijkstra with a binary heap and a settled flag.

   A plain array queue re-pushes a cell every time its distance improves,
   and with diagonal moves that grows without bound -- it threw
   "Invalid array length" before this was a heap. */
function Heap(){ this.a=[]; }
Heap.prototype.push=function(i,d){
  var a=this.a; a.push([d,i]); var c=a.length-1;
  while(c>0){ var p=(c-1)>>1; if(a[p][0]<=a[c][0]) break;
    var t=a[p]; a[p]=a[c]; a[c]=t; c=p; }
};
Heap.prototype.pop=function(){
  var a=this.a, top=a[0], last=a.pop();
  if(a.length){ a[0]=last; var i=0;
    for(;;){ var l=2*i+1, r=l+1, m=i;
      if(l<a.length&&a[l][0]<a[m][0]) m=l;
      if(r<a.length&&a[r][0]<a[m][0]) m=r;
      if(m===i) break; var t=a[m]; a[m]=a[i]; a[i]=t; i=m; } }
  return top;
};
/* Comfort weight. 1.0 in open floor, up to CENTER_W where a crab would be
   scraping the margin. Applied to route SHAPE only -- every distance the robot
   reasons or reports with stays in honest metres. */
var CENTER_PREF=0.26, CENTER_W=3.2, useWeight=false;
function wOf(i){
  if(!useWeight) return 1;
  var c=clear[i]; if(c>=CENTER_PREF) return 1;
  return 1+CENTER_W*(CENTER_PREF-c)/CENTER_PREF;
}
var settled=new Uint8Array(NX*NY);
function flood(out, seedIdx, passable){
  out.fill(Infinity); settled.fill(0);
  if(seedIdx<0||seedIdx>=out.length||!passable(seedIdx)) return false;
  out[seedIdx]=0;
  var h=new Heap(); h.push(seedIdx,0);
  while(h.a.length){
    var top=h.pop(), i=top[1];
    if(settled[i]) continue;
    settled[i]=1;
    var ix=i%NX, iy=(i/NX)|0;
    for(var d=0;d<8;d++){
      var x=ix+DIRS[d][0], y=iy+DIRS[d][1];
      if(x<0||y<0||x>=NX||y>=NY) continue;
      var j=y*NX+x; if(settled[j]||!passable(j)) continue;
      var nd=out[i]+CELL*DIRS[d][2]*(useWeight?0.5*(wOf(i)+wOf(j)):1);
      if(nd<out[j]-1e-9){ out[j]=nd; h.push(j,nd); }
    }
  }
  return true;
}
/* ---------- the reward as a blob of possibility ----------
   The crabs know roughly where the reward is (it carries a tag), not exactly.
   rewardBelief[i] = 1 while cell i could still hold it. Seeing a cell of the
   blob and not seeing the star eliminates that cell. Seeing the star collapses
   the blob to one cell. Until then the team plans toward the CENTROID of the
   surviving cells, snapped to the nearest surviving cell believed passable --
   one seed for one flood; a nearest-to-each-crab target would need one flood
   per crab, and that cost 100 ms a cycle. */
var REWARD_R=1.2, rewardBelief=new Uint8Array(NX*NY), nRewardBelief=0, rewardFound=false;
var rewardTgt={x:GOAL.x,y:GOAL.y}, goalCell=Math.floor(GOAL.y/CELL)*NX+Math.floor(GOAL.x/CELL);
function seedRewardBelief(){
  rewardBelief.fill(0); nRewardBelief=0; rewardFound=false;
  for(var iy=0;iy<NY;iy++) for(var ix=0;ix<NX;ix++){
    var i=iy*NX+ix, x=(ix+0.5)*CELL, y=(iy+0.5)*CELL;
    if(truth[i]!==BLOCKED&&Math.hypot(x-GOAL.x,y-GOAL.y)<=REWARD_R){ rewardBelief[i]=1; nRewardBelief++; }
  }
  updateRewardTarget();
}
/* a crab has looked at cell k */
function rewardSeen(k){
  if(!rewardBelief[k]) return;
  if(k===goalCell){
    if(!rewardFound){ rewardFound=true; rewardBelief.fill(0); rewardBelief[k]=1; nRewardBelief=1;
      updateRewardTarget(); say("the reward is in sight"); }
    return;
  }
  rewardBelief[k]=0; nRewardBelief--;
}
function updateRewardTarget(){
  if(rewardFound||!nRewardBelief){ rewardTgt.x=GOAL.x; rewardTgt.y=GOAL.y; return; }
  var sx=0, sy=0, i;
  for(i=0;i<rewardBelief.length;i++) if(rewardBelief[i]){ sx+=(i%NX+0.5)*CELL; sy+=(((i/NX)|0)+0.5)*CELL; }
  var cx=sx/nRewardBelief, cy=sy/nRewardBelief, best=-1, bd=Infinity;
  for(i=0;i<rewardBelief.length;i++){
    if(!rewardBelief[i]||!imagined(i)) continue;
    var d=Math.hypot((i%NX+0.5)*CELL-cx,(((i/NX)|0)+0.5)*CELL-cy);
    if(d<bd){ bd=d; best=i; }
  }
  if(best<0){ rewardTgt.x=cx; rewardTgt.y=cy; return; }
  rewardTgt.x=(best%NX+0.5)*CELL; rewardTgt.y=(((best/NX)|0)+0.5)*CELL;
}
var goalIdx=function(){ return idxAt(rewardTgt.x,rewardTgt.y); };
function wavefrontOld(){
  var t0=(typeof performance!=="undefined"?performance.now():0);
  flood(dist, goalIdx(), function(i){ return known[i]===FREE; });
  frontier.fill(0); nFrontier=0;
  for(var iy=0;iy<NY;iy++) for(var ix=0;ix<NX;ix++){
    var i=iy*NX+ix; if(known[i]!==FREE) continue;
    for(var d=0;d<4;d++){
      var x=ix+DIRS[d][0], y=iy+DIRS[d][1];
      if(x<0||y<0||x>=NX||y>=NY) continue;
      if(known[y*NX+x]===UNKNOWN){ frontier[i]=1; nFrontier++; break; }
    }
  }
  waveMs=(typeof performance!=="undefined"?performance.now():0)-t0;
}
/* What the field WOULD be if the box were gone. This is how a robot decides
   the box is worth pushing: not by looking at the box, but by noticing that
   removing it makes an unreachable goal reachable. */
var distNoBox=new Float32Array(NX*NY);
function wavefrontWithoutBox(){
  flood(distNoBox, goalIdx(), function(i){
    return truth[i]===FREE && known[i]!==UNKNOWN;
  });
}

/* ---------- robots ---------- */
var VMAX=0.30, WMAX=2.4, ARRIVE=0.075, PUSH_GAIN=0.60;
/* A crab steers, it does not pivot. Yaw rate is capped at |v|/TURN_R, so with no
   speed there is no turn. To come round further than the steering allows it
   backs up, exactly as a car does. */
var TURN_R=0.34, KMAX=1/TURN_R, REV_AT=1.9, SPD_FLOOR=0.35, SPD_DIV=2.2;
var goalFound=false, reached=0, pushes=0, clock=0;

function wrap(a){ while(a>Math.PI)a-=2*Math.PI; while(a<-Math.PI)a+=2*Math.PI; return a; }
function idxAt(x,y){
  var ix=Math.max(0,Math.min(NX-1,Math.floor(x/CELL)));
  var iy=Math.max(0,Math.min(NY-1,Math.floor(y/CELL)));
  return iy*NX+ix;
}
function Bot(i,s){
  this.id=i; this.x=s.x; this.y=s.y; this.th=0; this.col=COLS[i%COLS.length];
  this.path=[]; this.wi=0; this.mode="EXPLORE"; this.note="looking for a way through";
  this.v=0; this.om=0; this.stuck=0; this.trail=[[s.x,s.y]];
  this.refx=s.x; this.refy=s.y; this.reft=0; this.evade=0; this.claim=-1;
  this.seeAt=0; this.goalBox=null; this.pushStart=-99; this.axis=null; this.job=null;
  this.orbit=0; this.wallHits=0; this.reversing=false;
  this.announced=false;
  this.cands=null; this.replanIn=0; this.done=false;
  this.plannedAt=0; this.latOk=false; this.stalled=false; this.ref2x=s.x; this.ref2y=s.y; this.ref2t=0;
  this.nEvProg=0; this.nEvWall=0; this.nEvBot=0; this.nUnstick=0; this.nStall=0; this.nUnstickEvade=0;
}
function evadeStats(){
  var p=0,w=0,o=0,u=0,st=0,ue=0;
  for(var i=0;i<bots.length;i++){ p+=bots[i].nEvProg; w+=bots[i].nEvWall; o+=bots[i].nEvBot; u+=bots[i].nUnstick; st+=bots[i].nStall; ue+=bots[i].nUnstickEvade; }
  return "evade prog="+p+" wall="+w+" bot="+o+" unstick="+u+"(evade "+ue+") stallReplan="+st+" slide="+slideOk+"/"+slideTry;
}
Bot.prototype.rect=function(x,y,th){
  return {x:x===undefined?this.x:x, y:y===undefined?this.y:y,
          w:BOT_L, h:BOT_W, th:th===undefined?this.th:th};
};

function rectHitsWalls(r){
  var c=rectCorners(r);
  for(var i=0;i<4;i++){
    if(c[i][0]<0||c[i][1]<0||c[i][0]>AW||c[i][1]>AH) return true;
  }
  var A=activeWalls();
  for(var w=0;w<A.length;w++) if(rectHitsSeg(r,A[w])) return true;
  return false;
}
/* Who has right of way. The crab clearing the doorway must never be the one
   that gives ground, or the whole team waits on a crab that yielded to a
   scout. Equal ranks break by serial so two crabs never both yield. */
function jobRank(b){
  return b.job==="UNBLOCK"?3 : b.job==="REACH"?2 : b.job==="STAGE"?1 : 0;
}
function blockerBot(r,self,bots){
  for(var i=0;i<bots.length;i++)
    if(bots[i]!==self&&rectsOverlap(r,bots[i].rect())) return bots[i];
  return null;
}
function botBlocked(r,self,bots){
  if(rectHitsWalls(r)) return true;
  for(var i=0;i<bots.length;i++)
    if(bots[i]!==self && rectsOverlap(r,bots[i].rect())) return true;
  return false;
}
function boxBlocked(nb,mover,bots){
  if(rectHitsWalls(nb)) return true;
  for(var i=0;i<bots.length;i++)
    if(bots[i]!==mover && rectsOverlap(nb,bots[i].rect())) return true;
  return false;
}

/* The push rule, exactly as specified: contact alone does nothing. The box
   moves only when the robot's motion this tick points into it. Brushing past
   it sideways leaves it where it is. */
function stepBot(b,bots,dt){
  if(b.done){ b.v=0; b.om=0; return; }
  /* Two crabs jostling each other never sit still long enough to trip a
     consecutive-blocked-tick counter, so measure real progress instead. */
  if(Math.hypot(b.x-b.refx,b.y-b.refy)>0.14){ b.refx=b.x; b.refy=b.y; b.reft=clock; }
  else if(clock-b.reft>3.5&&b.evade<clock){ b.evade=clock+1.1; b.evadeTurn=(b.id%2?1:-1)*1.3;
    b.reft=clock; b.replanIn=0; b.nEvProg++; }
  /* No progress for 2 s: plan again from THIS pose. The lattice includes
     reverse moves, so the plan it returns is an escape the crab can drive. */
  if(Math.hypot(b.x-b.ref2x,b.y-b.ref2y)>0.10){ b.ref2x=b.x; b.ref2y=b.y; b.ref2t=clock; b.stalled=false; }
  else if(clock-b.ref2t>2.0){ b.stalled=true; b.nStall++; b.ref2t=clock; planTimer=0; }
  if(b.evade>clock){
    b.v=-VMAX*0.55; b.om=b.evadeTurn;
    var eth=b.th+b.om*dt, ex=b.x+Math.cos(eth)*b.v*dt, ey=b.y+Math.sin(eth)*b.v*dt;
    var er=b.rect(ex,ey,eth);
    if(!botBlocked(er,b,bots)&&!rectsOverlap(er,box)){ b.x=ex; b.y=ey; b.th=eth; }
    else {
      /* Rotating in place is a move too. Turning without this test swung an
         evading crab's corner into a neighbour: 55 to 266 overlap ticks a run. */
      var rr=b.rect(b.x,b.y,eth);
      if(!botBlocked(rr,b,bots)&&!rectsOverlap(rr,box)) b.th=eth;
    }
    return;
  }
  var tgt=b.path[b.wi];
  if(!tgt){ b.v=0; b.om=0; return; }
  var dx=tgt.x-b.x, dy=tgt.y-b.y, d=Math.hypot(dx,dy);
  var tol=(b.mode==="PUSH"&&b.wi===b.path.length-1)?0.05:ARRIVE;
  if(d<tol){ b.wi++; b.orbit=0; return; }
  /* No waypoint skipping here. Dropping a waypoint when the heading error is
     large looks like it breaks an orbit, but it also throws away the one
     waypoint a lone crab needed to get through the doorway it had just opened. */

  /* Bias the heading away from any wall inside the margin. Following waypoints
     alone cuts corners, and a crab that ends up inside the margin is one the
     planner can no longer route. */
  var ux=dx/d, uy=dy/d, A=activeWalls(), rx=0, ry=0;
  /* Last leg of a push: steer by how far the crab has drifted OFF the push
     axis, not merely toward the end point. Chasing the end point alone lets the
     crab creep sideways and walk the box off line. */
  if(b.mode==="PUSH"&&b.axis&&b.wi===b.path.length-1){
    var ax=b.axis, off=(b.x-ax.x)*(-ax.dy)+(b.y-ax.y)*(ax.dx);
    var corr=Math.max(-0.9,Math.min(0.9,-3.4*off));
    ux=ax.dx-ax.dy*corr; uy=ax.dy+ax.dx*corr;
    var un=Math.hypot(ux,uy); ux/=un; uy/=un;
  }
  for(var w=0;w<A.length;w++){
    var W=A[w], wx=W[2]-W[0], wy=W[3]-W[1], LL=wx*wx+wy*wy;
    var tt=LL<1e-12?0:Math.max(0,Math.min(1,((b.x-W[0])*wx+(b.y-W[1])*wy)/LL));
    var ox=b.x-(W[0]+tt*wx), oy=b.y-(W[1]+tt*wy), od=Math.hypot(ox,oy);
    /* Stronger than the crab-to-crab term: a jostle in a 0.56 m band pushed a
       crab into the wall margin, where the planner cannot route it. */
    if(od<PLAN_PAD&&od>1e-6){ var f=3.0*(PLAN_PAD-od)/PLAN_PAD; rx+=ox/od*f; ry+=oy/od*f; }
  }
  ux+=rx; uy+=ry;
  /* and out of each other's way -- the same lateral bias, crab to crab. Not on
     the push leg: a crab parked beside the stand point deflected the pusher and
     the box went 0.19 m sideways. Right of way makes the other crab yield. */
  var onAxis=(b.mode==="PUSH"&&b.axis&&b.wi===b.path.length-1);
  for(var q=0;q<bots.length&&!onAxis;q++){
    var o=bots[q]; if(o===b||o.done) continue;
    var cxd=b.x-o.x, cyd=b.y-o.y, cd=Math.hypot(cxd,cyd);
    if(cd<0.55&&cd>1e-6){ var cf=2.2*(0.55-cd)/0.55; ux+=cxd/cd*cf; uy+=cyd/cd*cf; }
  }
  var e=wrap(Math.atan2(uy,ux)-b.th);
  /* The gear comes from the waypoint when the lattice laid it. Guessing the
     gear from heading error alone was the trap: a route behind the crab was
     reversed in its entirety, because backing straight along it keeps the
     error at 180 deg and nothing ever turns the crab round. */
  var rev;
  if(tgt.rev===true)       rev=Math.abs(e)>1.0;
  else if(tgt.rev===false) rev=Math.abs(e)>2.62;
  else                     rev=Math.abs(e)>REV_AT;   /* fallback path: it is behind us */
  var eff=rev?wrap(e-Math.PI):e;
  b.v=(rev?-0.55:1)*VMAX*Math.max(SPD_FLOOR,1-Math.abs(eff)/SPD_DIV);
  var lim=Math.abs(b.v)*KMAX;              /* no speed, no turn */
  b.om=Math.max(-lim,Math.min(lim,3.0*eff));
  b.reversing=rev;

  var nth=b.th+b.om*dt;
  var mx=Math.cos(nth)*b.v*dt, my=Math.sin(nth)*b.v*dt;
  var nx=b.x+mx, ny=b.y+my;
  var cand=b.rect(nx,ny,nth), moved=false;
  var hitWall=rectHitsWalls(cand);
  var hitBot=hitWall?null:blockerBot(cand,b,bots);
  if(hitBot&&(jobRank(hitBot)>jobRank(b)||
             (jobRank(hitBot)===jobRank(b)&&hitBot.id<b.id))){
    if(b.evade<clock){ b.evade=clock+0.5; b.evadeTurn=(b.id%2?1:-1)*1.2; b.nEvBot++; }
  }

  if(!hitWall&&!hitBot){
    if(rectsOverlap(cand,box)){
      /* moving INTO the box? dot of motion with the vector toward the box */
      if(mx*(box.x-b.x)+my*(box.y-b.y)>0){
        var nb={x:box.x+mx,y:box.y+my,w:box.w,h:box.h,th:box.th};
        if(!boxBlocked(nb,b,bots)){
          box.x=nb.x; box.y=nb.y; stampBox();
          b.x=nx; b.y=ny; b.th=nth; moved=true; pushed+=Math.hypot(mx,my);
        }
      }
      /* touching but not driving into it -> nothing moves */
    } else { b.x=nx; b.y=ny; b.th=nth; moved=true; }
  }
  /* Wall contact SLIDES. Rejecting the whole translation on any contact stopped
     the crab dead against the wall, spun it on the spot, and backed it off 6 cm
     after 45 ticks -- the "ramming and backing out" that was being watched. A
     chassis grazing a wall at a shallow angle keeps the tangential part of its
     motion. Only when the slide also collides is the crab genuinely wedged. */
  if(!moved&&hitWall){
    var nnx=rx, nny=ry, nn=Math.hypot(nnx,nny);
    if(nn<1e-6){                        /* normal from the closest wall segment */
      var bd=Infinity;
      for(var w2=0;w2<A.length;w2++){
        var W2=A[w2], wx2=W2[2]-W2[0], wy2=W2[3]-W2[1], LL2=wx2*wx2+wy2*wy2;
        var tt2=LL2<1e-12?0:Math.max(0,Math.min(1,((b.x-W2[0])*wx2+(b.y-W2[1])*wy2)/LL2));
        var ox2=b.x-(W2[0]+tt2*wx2), oy2=b.y-(W2[1]+tt2*wy2), od2=Math.hypot(ox2,oy2);
        if(od2<bd){ bd=od2; nnx=ox2; nny=oy2; }
      }
      nn=Math.hypot(nnx,nny);
    }
    if(nn>1e-6){
      nnx/=nn; nny/=nn;
      var into=mx*nnx+my*nny, tx_=mx-into*nnx, ty_=my-into*nny;
      if(Math.hypot(tx_,ty_)>1e-7){
        slideTry++;
        var sr=b.rect(b.x+tx_,b.y+ty_,nth);
        if(!rectHitsWalls(sr)&&!blockerBot(sr,b,bots)&&!rectsOverlap(sr,box)){
          b.x+=tx_; b.y+=ty_; b.th=nth; moved=true; slideOk++;
        }
      }
    }
  }
  if(!moved){
    /* turning in place is still allowed -- this is what frees a wedged corner */
    var rot=b.rect(b.x,b.y,nth);
    if(!botBlocked(rot,b,bots)&&!rectsOverlap(rot,box)) b.th=nth;
    b.stuck++;
    if(b.stuck>20){ b.replanIn=0; }
    if(b.stuck>45){                       /* pinned: give ground, then rethink */
      var bx=b.x-Math.cos(b.th)*0.06, by=b.y-Math.sin(b.th)*0.06;
      if(!botBlocked(b.rect(bx,by,b.th),b,bots)&&!rectsOverlap(b.rect(bx,by,b.th),box)){ b.x=bx; b.y=by; }
      b.stuck=0;
    }
  } else { b.stuck=0; }

  var t=b.trail[b.trail.length-1];
  if(Math.hypot(b.x-t[0],b.y-t[1])>0.05){ b.trail.push([b.x,b.y]); if(b.trail.length>420) b.trail.shift(); }
}
var pushed=0, slideTry=0, slideOk=0;

/* ---------- the field of possibilities ----------
   A robot never "sees a box in the way". It asks one question -- what does the
   reach field cost me from here? -- and then asks it again for each box
   placement it could create. If some imagined placement answers cheaper, that
   placement becomes the goal. Pushing is derived, never scripted. */
var scratch=new Float32Array(NX*NY);
/* Passable under an IMAGINED box: the cell must have been seen, must not be
   wall, and must not be under wherever the box is currently stamped. Reading
   `known` alone would keep the box at the place it was observed, so every
   imagined placement would score identically -- which is exactly the bug that
   made the first version of this search return a gain of zero. */
/* Optimism is what makes exploration and planning the same act. Unknown space
   is assumed open, so the route to the reward runs straight through it; driving
   that route is what discovers the walls. A cell is closed only if a robot has
   SEEN a wall there, or the box (as currently imagined) sits on it. */
function imagined(i){
  if(known[i]===BLOCKED && truth[i]===BLOCKED) return false;
  if(boxSeen && boxCell[i]) return false;
  return true;
}
function costFrom(b){
  var bi=idxAt(b.x,b.y);
  flood(scratch, goalIdx(), function(i){ return i===bi || imagined(i); });
  return scratch[bi];
}
var PUSHDIR=[[1,0],[-1,0],[0,1],[0,-1]];
var reachNow=new Float32Array(NX*NY);
/* Score the WHOLE plan -- walk to the pushing position, shove, then walk on --
   not just the world that would exist afterwards. Scoring only the aftermath
   picks pushes that require standing on the far side of the very doorway the
   box is sealing. */
/* Cheap gate before the expensive search: ask once what the world would cost
   with no box at all. If that is no better than the route the robot already
   has, no placement of the box can help, and the twenty-flood placement search
   is skipped entirely. Two floods instead of twenty, most ticks. */
/* Distance from a placed box to the route the crab would drive if the box did
   not exist. This is the gate that matters: a placement is only useful if it
   leaves that route wide enough to drive. */
function clearOfRoute(nb,route){
  var m=Infinity;
  for(var i=1;i<route.length;i++){
    var ax=route[i-1].x, ay=route[i-1].y, bx=route[i].x, by=route[i].y;
    var n=Math.max(1,Math.ceil(Math.hypot(bx-ax,by-ay)/0.03));
    for(var k=0;k<=n;k++){
      var t=k/n, d=rectDist(ax+(bx-ax)*t, ay+(by-ay)*t, nb);
      if(d<m) m=d;
    }
  }
  return m;
}
/* Has a crab SEEN the space this placement would sweep through? If not, a wall
   could be hiding behind the box. Certain placements are green, maybes yellow. */
/* Fraction of the swept cells nobody has looked at. 0 = certain. */
function sweepUnseen(sx,sy,nx,ny){
  var steps=Math.max(1,Math.ceil(Math.hypot(nx-sx,ny-sy)/0.09));
  var ri=Math.ceil((Math.max(box.w,box.h)/2+0.04)/CELL), tot=0, un=0;
  for(var s=1;s<=steps;s++){
    var t=s/steps, r={x:sx+(nx-sx)*t, y:sy+(ny-sy)*t, w:box.w, h:box.h, th:0};
    var ci=Math.floor(r.x/CELL), cj=Math.floor(r.y/CELL);
    /* every second cell: counting them all cost 13 ms a call, over the 5 ms bar */
    for(var dy=-ri;dy<=ri;dy+=2) for(var dx=-ri;dx<=ri;dx+=2){
      var X=ci+dx, Y=cj+dy;
      if(X<0||Y<0||X>=NX||Y>=NY) continue;
      if(rectDist((X+0.5)*CELL,(Y+0.5)*CELL,r)>0) continue;
      tot++; if(known[Y*NX+X]===UNKNOWN) un++;
    }
  }
  return tot?un/tot:0;
}
function sweepSeen(sx,sy,nx,ny){ return sweepUnseen(sx,sy,nx,ny)===0; }
/* How much room the box would have where it lands. The clearance grid already
   holds metres-to-the-nearest-solid-thing, so this is one lookup, not a disc. */
function freedomAt(nb){
  var i=idxAt(nb.x,nb.y);
  var c=clear[i];
  if(!isFinite(c)) return 0;
  return Math.min(1, c/0.55);
}
/* Certain beats maybe. Among equals, more room for the box, then cheaper. */
/* Then: fewer unseen cells in the sweep, and a push ACROSS the direction of
   travel over one along it. Driving the box ahead of yourself sends it into
   space you cannot see; shoving it sideways out of the gap does not. */
var UNSEEN_W=1.2, ALONG_W=0.9;
function score(a){ return a.total-a.freedom*1.6+a.unseen*UNSEEN_W+a.along*ALONG_W; }
function better(a,c){
  if(a.certain!==c.certain) return a.certain;
  return score(a)<score(c)-1e-6;
}
function pushStates(b){
  var sx=box.x, sy=box.y, base=costFrom(b), out=[], best=null;
  var bi=idxAt(b.x,b.y);
  var wasSeen=boxSeen; boxSeen=false;
  var ideal=costFrom(b);
  /* The route that would exist with no box at all -- the thing the box is in
     the way OF. Scoring placements by field cost alone fails here: with unknown
     space assumed open the route simply detours around the box through cells
     nobody has looked at, so every placement scores the same and the shortest,
     useless push wins. */
  boxSeen=wasSeen;
  /* The gap the box is sitting in: cells it denies that are otherwise open
     floor. A placement counts only if it vacates EVERY one of them -- "get it
     out of the doorway", which is local, physical, and does not care what the
     optimistic field believes about space nobody has looked at. */
  var throat=[];
  for(var t=0;t<boxIdx.length;t++) if(truth[boxIdx[t]]===FREE) throat.push(boxIdx[t]);
  if(!throat.length) return {base:base,best:null,cands:[],gain:0,ideal:ideal};
  /* Reachability of a PUSHING POSITION is judged on ground a crab has actually
     seen. Judging it optimistically offered stand points on the far side of the
     very doorway the box seals. */
  flood(reachNow, bi, function(i){ return i===bi || known[i]===FREE; });
  /* which way the crab wants to travel through the gap: downhill in the
     box-free field at the box */
  var bc=idxAt(sx,sy), tdx=0, tdy=0, tv=distNoBox[bc];
  for(var td=0;td<8;td++){
    var tX=bc%NX+DIRS[td][0], tY=((bc/NX)|0)+DIRS[td][1];
    if(tX<0||tY<0||tX>=NX||tY>=NY) continue;
    var tj=tY*NX+tX; if(distNoBox[tj]<tv){ tv=distNoBox[tj]; tdx=DIRS[td][0]; tdy=DIRS[td][1]; }
  }
  var tn=Math.hypot(tdx,tdy)||1; tdx/=tn; tdy/=tn;
  for(var d=0;d<4;d++){
    for(var k=1;k<=8;k++){
      var L=k*0.12;
      var nx=sx+PUSHDIR[d][0]*L, ny=sy+PUSHDIR[d][1]*L;
      var nb={x:nx,y:ny,w:box.w,h:box.h,th:box.th};
      if(rectHitsWalls(nb)) break;            /* the ray ends at the wall */
      /* where the robot must stand to push this way, and can it get there? */
      /* Stand clear of the box's own planning pad, not flush against it. At
         half+BOT_L/2+0.05 the gap was 5 mm; one nudge from a passing crab put
         every pushing position inside the pad and the search returned "no
         placement" for the rest of the run. */
      var half=(d<2?box.w:box.h)/2, off=half+BOX_PAD+0.06;
      var st={x:sx-PUSHDIR[d][0]*off, y:sy-PUSHDIR[d][1]*off};
      var si=idxAt(st.x,st.y);
      if(!imagined(si)||!isFinite(reachNow[si])) continue;   /* can I even get there? */
      var walk=reachNow[si];
      /* No wavefront per candidate. Vacating is a direct geometric test, and
         the leg after the shove is one lookup in the box-free field. Re-flooding
         for each of 32 placements cost 23 ms a call and a crab stood waiting. */
      var vacates=true, PADV=BOX_PAD+0.07;
      for(var q=0;q<throat.length;q++){
        var tq=throat[q];
        if(rectDist((tq%NX+0.5)*CELL,(((tq/NX)|0)+0.5)*CELL,nb)<PADV){ vacates=false; break; }
      }
      var onward=distNoBox[si], c=isFinite(onward)?onward:1e6;
      var un=sweepUnseen(sx,sy,nx,ny);
      var s={x:nx,y:ny,cost:c,total:walk+L+c,walk:walk,stand:st,dir:d,len:L,vacates:vacates,
             certain:un===0, unseen:un, along:Math.abs(PUSHDIR[d][0]*tdx+PUSHDIR[d][1]*tdy), freedom:freedomAt(nb)};
      out.push(s);
      if(vacates&&(!best||better(s,best))) best=s;
    }
  }
  /* Stop at the theoretical minimum and a centimetre of slip leaves the doorway
     still shut. Shove it clear and then some. */
  if(best){
    var over=0.16, ex={x:best.x+PUSHDIR[best.dir][0]*over, y:best.y+PUSHDIR[best.dir][1]*over,
                       w:box.w,h:box.h,th:box.th};
    if(!rectHitsWalls(ex)){ best.x=ex.x; best.y=ex.y; best.len+=over; best.total+=over; }
  }
  return {base:base, best:best, cands:out, ideal:ideal, atBx:sx, atBy:sy,
          gain:(best&&isFinite(best.total))?(base-best.total):0};
}

/* ---------- waypoints ----------
   Cells come out of the field one at a time; that is not a route a chassis can
   drive. Shortcut it against the padded grid -- the same padding the field was
   built with -- so a waypoint is never closer to a wall than the chassis is
   wide, and the corners come out as one clean turn. */
function clearLine(ax,ay,bx,by,passable){
  var n=Math.ceil(Math.hypot(bx-ax,by-ay)/(CELL*0.5));
  for(var i=0;i<=n;i++){
    var t=i/n;
    if(!passable(idxAt(ax+(bx-ax)*t, ay+(by-ay)*t))) return false;
  }
  return true;
}
function descend(field,fromIdx,passable){
  var pts=[], i=fromIdx, guard=0;
  if(!isFinite(field[i])) return null;
  while(field[i]>0&&guard++<20000){
    var ix=i%NX, iy=(i/NX)|0, bi=-1, bv=field[i];
    for(var d=0;d<8;d++){
      var x=ix+DIRS[d][0], y=iy+DIRS[d][1];
      if(x<0||y<0||x>=NX||y>=NY) continue;
      var j=y*NX+x; if(!passable(j)) continue;
      if(field[j]<bv-1e-9){ bv=field[j]; bi=j; }
    }
    if(bi<0) break;
    i=bi; pts.push({x:(i%NX+0.5)*CELL, y:(((i/NX)|0)+0.5)*CELL});
  }
  return pts;
}
function smooth(sx,sy,pts,passable){
  var out=[], cx=sx, cy=sy, i=0;
  while(i<pts.length){
    var j=pts.length-1;
    while(j>i && !clearLine(cx,cy,pts[j].x,pts[j].y,passable)) j--;
    out.push(pts[j]); cx=pts[j].x; cy=pts[j].y; i=j+1;
  }
  return out;
}

/* ---------- approach curve ----------
   A crab used to arrive at the pushing position on whatever heading the field
   path happened to end on, then pivot against the box -- which shoved it
   crooked. The last leg is now a cubic Bezier whose END TANGENT is the push
   direction itself, so the crab is already square to the box when it touches
   it. The curve is only accepted if every sample keeps a full chassis margin
   from the walls; otherwise the handles are shortened until it does. */
function bez(p0,p1,p2,p3,n){
  var out=[];
  for(var i=1;i<=n;i++){
    var t=i/n, u=1-t, a=u*u*u, b=3*u*u*t, c=3*u*t*t, d=t*t*t;
    out.push({x:a*p0.x+b*p1.x+c*p2.x+d*p3.x, y:a*p0.y+b*p1.y+c*p2.y+d*p3.y});
  }
  return out;
}
function wallClear(x,y){
  var A=activeWalls(), m=Math.min(x,y,AW-x,AH-y);
  for(var w=0;w<A.length;w++){
    var v=segDist(x,y,A[w][0],A[w][1],A[w][2],A[w][3]);
    if(v<m) m=v;
  }
  return m;
}
/* A curve the chassis cannot follow is not a route. Reject any cubic that bends
   tighter than the turning radius. */
function curveDrivable(p0,pts){
  var all=[p0].concat(pts);
  for(var i=2;i<all.length;i++){
    var a=all[i-2], m=all[i-1], c=all[i];
    var t1=Math.atan2(m.y-a.y,m.x-a.x), t2=Math.atan2(c.y-m.y,c.x-m.x);
    var len=Math.hypot(c.x-m.x,c.y-m.y);
    if(len>1e-6&&Math.abs(wrap(t2-t1))/len>KMAX*1.05) return false;
  }
  return true;
}
function curveClear(pts){
  for(var i=0;i<pts.length;i++)
    if(wallClear(pts[i].x,pts[i].y)<PLAN_PAD) return false;
  return true;
}
/* Waypoints are a spline, not a polyline.

   Two goal points -- where the crab is, and where it must end up -- and one
   cubic between each pair of key points. The final tangent is CLAMPED to the
   direction the crab must be facing on arrival, and each cubic's handles are
   shortened until every sample of it holds a full chassis margin off the walls.
   Routing to a point set back along the push axis produced huge hairpins; the
   curve does that job without the detour. */
function splineOf(sx,sy,sth,pts,endDir){
  var P=[{x:sx,y:sy}].concat(pts), n=P.length, i;
  if(n<2) return pts.slice();
  var tan=new Array(n);
  for(i=0;i<n;i++){
    var a=P[Math.max(0,i-1)], b=P[Math.min(n-1,i+1)];
    tan[i]={x:(b.x-a.x)*0.5, y:(b.y-a.y)*0.5};
  }
  var L0=Math.hypot(P[1].x-P[0].x,P[1].y-P[0].y);
  tan[0]={x:Math.cos(sth)*L0, y:Math.sin(sth)*L0};
  if(endDir){
    var LN=Math.hypot(P[n-1].x-P[n-2].x,P[n-1].y-P[n-2].y);
    tan[n-1]={x:endDir[0]*LN, y:endDir[1]*LN};
  }
  var out=[];
  for(i=0;i<n-1;i++){
    var leg=legOf(P[i],tan[i],P[i+1],tan[i+1]);
    if(!leg){
      /* One cubic cannot clear it: go round. Put a point off the chord, sweep
         the offset both ways, and lay two cubics through it. */
      var p0=P[i], p3=P[i+1], mx=(p0.x+p3.x)/2, my=(p0.y+p3.y)/2;
      var cx=p3.x-p0.x, cy=p3.y-p0.y, cl=Math.hypot(cx,cy)||1, px=-cy/cl, py=cx/cl;
      var OFF=[0.10,-0.10,0.20,-0.20,0.30,-0.30,0.45,-0.45];
      for(var o=0;o<OFF.length&&!leg;o++){
        var m={x:mx+px*OFF[o], y:my+py*OFF[o]};
        if(wallClear(m.x,m.y)<PLAN_PAD) continue;
        var tm={x:cx*0.5, y:cy*0.5};
        var a1=legOf(p0,tan[i],m,tm); if(!a1) continue;
        var a2=legOf(m,tm,p3,tan[i+1]); if(!a2) continue;
        leg=a1.concat(a2); splineSubdiv++;
      }
    }
    if(!leg){ splineNull++; return null; }   /* no drivable curve exists: say so */
    out=out.concat(leg);
  }
  return out;
}
var splineSubdiv=0, splineNull=0;
/* One cubic between two key points, handles shortened until it is both clear
   and drivable. The curvature test runs on 18 samples: on 6 it understated the
   curvature between samples and passed curves the chassis could not follow. */
function legOf(p0,t0,p3,t3){
  for(var sc=1.5; sc>=0.12; sc-=0.115){
    var p1={x:p0.x+t0.x*sc/3, y:p0.y+t0.y*sc/3};
    var p2={x:p3.x-t3.x*sc/3, y:p3.y-t3.y*sc/3};
    var fine=bez(p0,p1,p2,p3,18);
    if(curveClear(fine)&&curveDrivable(p0,fine)) return bez(p0,p1,p2,p3,6);
  }
  return null;
}
/* ---------- state lattice ----------
   Plan over POSES the chassis can reach, not over cells. A grid path followed
   by a curve is frequently not drivable from the crab's present heading once
   the chassis has a turning radius; the controller then thrashes. Measured: a
   PUSH route laid directly behind the crab put it in reverse for 30 s at half
   speed with the heading error pinned at 180 deg -- reverse is a stable trap,
   nothing in the controller turns the crab round.

   State (x, y, theta), theta continuous, visited set keyed on (cell, one of
   16 heading buckets). Six motion primitives: arc left, straight, arc right,
   forward and reverse, each drivable by construction. Cost is arc length times
   the centre-seeking weight, reverse 2.5x, and a gear change on top. The
   heuristic is the wavefront already flooded from the target. The output is a
   list of poses each reachable from the last, with the gear that gets there. */
var LAT_H=16, LAT_DTH=2*Math.PI/LAT_H, LAT_R=0.40, LAT_ARC=LAT_R*LAT_DTH, LAT_STR=0.15;
var LAT_REV_W=2.5, LAT_GEAR_W=0.30, LAT_BUDGET=6000, LAT_TOL=0.09, LAT_ATOL=0.03, LAT_LAT=0.05, LAT_SHOT=0.70;
/* Heuristic = max(wavefront distance, LAT_R * heading still to turn), inflated
   by LAT_EPS. Without the heading term the search reached the pushing position
   299 times facing the wrong way and ran out of budget (4001 nodes); with it,
   and the inflation, the same search closes in a few hundred. */
var LAT_EPS=1.25;
var LAT_CAP=LAT_BUDGET*6+64;
var latX=new Float32Array(LAT_CAP), latY=new Float32Array(LAT_CAP), latTh=new Float32Array(LAT_CAP), latG=new Float32Array(LAT_CAP);
var latPar=new Int32Array(LAT_CAP), latGear=new Int8Array(LAT_CAP);
var latSeen=new Int32Array(NX*NY*LAT_H), latGen=0;
var latMs=0, latNodes=0, latCalls=0, latFails=0, latMsMax=0, latNodesMax=0, latMsSum=0, latNodesSum=0;
var latP={x:0,y:0,th:0};
function latBucket(th){ var h=Math.round(th/LAT_DTH)%LAT_H; if(h<0) h+=LAT_H; return h; }
function latW(i){ var c=clear[i]; return c>=CENTER_PREF?1:1+CENTER_W*(CENTER_PREF-c)/CENTER_PREF; }
/* Pose a fraction f along one primitive. s = +1 forward, -1 reverse. a = signed
   heading change over the whole primitive, 0 for straight. */
function latStep(x,y,th,s,a,f,o,R){
  if(a===0){ o.x=x+s*LAT_STR*f*Math.cos(th); o.y=y+s*LAT_STR*f*Math.sin(th); o.th=th; return; }
  var af=a*f, rho=s*R*(a>0?1:-1);
  o.x=x+rho*(Math.sin(th+af)-Math.sin(th)); o.y=y-rho*(Math.cos(th+af)-Math.cos(th)); o.th=th+af;
}
/* A sample is legal on believed-open floor. Near the START only, it is also
   legal on real floor where the chassis physically clears the walls: that is
   how a crab inside the planning margin drives itself back out. */
var latEscape=false;   /* the search started inside the margin */
function latOk(x,y,th,sx,sy){
  if(x<0||y<0||x>=AW||y>=AH) return false;
  var i=idxAt(x,y);
  if(imagined(i)) return true;
  if(!latEscape||Math.hypot(x-sx,y-sy)>0.30||!phys[i]) return false;
  if(boxSeen&&boxCell[i]) return false;
  return !rectHitsWalls({x:x,y:y,w:BOT_L,h:BOT_W,th:th});
}
function latSearch(sx,sy,sth,tx,ty,endTh){
  var t0=performance.now();
  latGen++; latCalls++; latEscape=wallStuck(idxAt(sx,sy))||(boxSeen&&boxCell[idxAt(sx,sy)]===1);
  var n=1, expanded=0, h=new Heap(), goal=-1, shot=false;
  latX[0]=sx; latY[0]=sy; latTh[0]=sth; latG[0]=0; latPar[0]=-1; latGear[0]=0;
  var ex=endTh===undefined?0:Math.cos(endTh), ey=endTh===undefined?0:Math.sin(endTh);
  function hOf(x,y,th){
    var v=scratch[idxAt(x,y)]; if(!isFinite(v)) v=Math.hypot(x-tx,y-ty)+0.5;
    if(endTh!==undefined){ var hh=LAT_R*Math.abs(wrap(th-endTh)); if(hh>v) v=hh; }
    return LAT_EPS*v;
  }
  h.push(0,hOf(sx,sy,sth));
  var PR=[[1,LAT_DTH],[1,0],[1,-LAT_DTH],[-1,LAT_DTH],[-1,0],[-1,-LAT_DTH]];
  /* The first move also SNAPS the heading onto the lattice of the required end
     heading (at most 11.25 deg, a gentle arc), so every later heading is exactly
     endTh + k*22.5 deg and the approach can end square. Landing up to 11 deg
     off the push axis pushed the box 0.19 m sideways over a 0.9 m shove. */
  var snap=0;
  if(endTh!==undefined){ snap=wrap(endTh-sth); snap-=Math.round(snap/LAT_DTH)*LAT_DTH; if(Math.abs(snap)<1e-4) snap=0; }
  while(h.a.length){
    var top=h.pop(), c=top[1];
    var key=idxAt(latX[c],latY[c])*LAT_H+latBucket(latTh[c]);
    if(latSeen[key]===latGen) continue;
    latSeen[key]=latGen;
    if(++expanded>LAT_BUDGET) break;
    var cx=latX[c], cy=latY[c], cth=latTh[c], dxg=tx-cx, dyg=ty-cy, dg=Math.hypot(dxg,dyg);
    if(endTh===undefined){
      if(dg<=LAT_TOL){ goal=c; break; }
    } else {
      var along=dxg*ex+dyg*ey, lat=Math.abs(-dxg*ey+dyg*ex);
      var aligned=Math.abs(wrap(cth-endTh))<=LAT_ATOL;
      if(aligned&&lat<=LAT_LAT&&along>=-0.03&&along<=LAT_TOL){ goal=c; break; }
      /* straight shot: already on the axis and facing along it, and the run in is clear */
      if(aligned&&lat<=LAT_LAT&&along>0&&along<=LAT_SHOT&&latGear[c]>=0){
        var m=Math.ceil(along/CELL), okS=true;
        for(var q=1;q<=m&&okS;q++){ var f=q/m; if(!latOk(cx+dxg*f,cy+dyg*f,endTh,sx,sy)) okS=false; }
        if(okS){ goal=c; shot=true; break; }
      }
    }
    for(var p=0;p<6&&n<LAT_CAP;p++){
      var s=PR[p][0], a=PR[p][1], R=LAT_R, L=(a===0?LAT_STR:LAT_ARC), ok=true, wsum=0;
      if(c===0&&snap!==0){
        if(a===0){ a=snap; R=LAT_STR/Math.abs(snap); }
        else { a+=snap; L=LAT_R*Math.abs(a); }
      }
      for(var k=1;k<=3;k++){
        latStep(cx,cy,cth,s,a,k/3,latP,R);
        if(!latOk(latP.x,latP.y,latP.th,sx,sy)){ ok=false; break; }
        wsum+=latW(idxAt(latP.x,latP.y));
      }
      if(!ok) continue;
      var nk=idxAt(latP.x,latP.y)*LAT_H+latBucket(latP.th);
      if(latSeen[nk]===latGen) continue;
      var cost=L*(wsum/3)*(s<0?LAT_REV_W:1);
      if(latGear[c]!==0&&latGear[c]!==s) cost+=LAT_GEAR_W;
      latX[n]=latP.x; latY[n]=latP.y; latTh[n]=wrap(latP.th); latG[n]=latG[c]+cost; latPar[n]=c; latGear[n]=s;
      h.push(n,latG[n]+hOf(latP.x,latP.y,latP.th)); n++;
    }
  }
  latMs=performance.now()-t0; latNodes=expanded;
  latMsSum+=latMs; latNodesSum+=expanded;
  if(latMs>latMsMax) latMsMax=latMs; if(expanded>latNodesMax) latNodesMax=expanded;
  if(goal<0){ latFails++; return null; }
  var out=[];
  for(var i=goal;i>0;i=latPar[i]) out.push({x:latX[i],y:latY[i],th:latTh[i],rev:latGear[i]<0});
  out.reverse();
  if(shot) out.push({x:tx,y:ty,th:endTh,rev:false});
  /* already there: hand back the target itself, never an empty path. An empty
     path read as "finished" and re-triggered planning on every tick. */
  if(!out.length) out.push({x:tx,y:ty,th:endTh===undefined?sth:endTh,rev:false});
  return out;
}
function latStats(){
  return "lat calls="+latCalls+" fails="+latFails+" msAvg="+(latCalls?latMsSum/latCalls:0).toFixed(2)+
         " msMax="+latMsMax.toFixed(1)+" nodesAvg="+(latCalls?latNodesSum/latCalls:0).toFixed(0)+" nodesMax="+latNodesMax+" noEnd="+latNoEnd+" spline="+latFallbacks;
}
function approachPush(b,stand,dv,len){
  var r=routeTo(b,stand.x,stand.y,dv);
  if(!r||!r.length) return null;
  r.push({x:stand.x+dv[0]*(len+0.14), y:stand.y+dv[1]*(len+0.14), rev:false});
  return r;
}

/* ---------- planning ----------
   Both fields are flooded ONCE per planning cycle and every crab reads its own
   cell out of them. Flooding per crab instead made planning cost 100 ms, and an
   idle crab re-triggered it on every frame. */
var bots=[], pushClaim=null, lastStates=null, narration=[], planMs=0, planTimer=0;
function say(t){ narration.push({t:clock,s:t}); if(narration.length>90) narration.shift(); }
function knownPass(i){ return imagined(i); }

function fields(){
  var t0=performance.now(), i, ix, iy, d;
  updateRewardTarget();
  var gi=goalIdx();
  flood(dist, gi, imagined);                       /* the world as believed */
  var w=boxSeen; boxSeen=false;
  flood(distNoBox, gi, imagined);                  /* the world with no box */
  boxSeen=w;
  frontier.fill(0); nFrontier=0;
  for(iy=0;iy<NY;iy++) for(ix=0;ix<NX;ix++){
    i=iy*NX+ix; if(known[i]!==FREE) continue;
    for(d=0;d<4;d++){
      var x=ix+DIRS[d][0], y=iy+DIRS[d][1];
      if(x<0||y<0||x>=NX||y>=NY) continue;
      if(known[y*NX+x]===UNKNOWN){ frontier[i]=1; nFrontier++; break; }
    }
  }
  waveMs=performance.now()-t0;
}
/* A crab can sit in a cell the planner calls padded-solid; read the best
   neighbour rather than declaring the crab unreachable from itself. */
function ring(ix,iy,r,test){
  var out=-1, bd=Infinity;
  for(var dy=-r;dy<=r;dy++) for(var dx=-r;dx<=r;dx++){
    if(Math.max(Math.abs(dx),Math.abs(dy))!==r) continue;
    var X=ix+dx, Y=iy+dy;
    if(X<0||Y<0||X>=NX||Y>=NY) continue;
    var j=Y*NX+X; if(!test(j)) continue;
    var d=dx*dx+dy*dy; if(d<bd){ bd=d; out=j; }
  }
  return out;
}
/* A crab inside the planning margin has no field value of its own. Find one by
   walking OUTWARD OVER REAL FLOOR -- never by radius. A ring search here read a
   cell on the far side of a sealing wall and handed a walled-off crab a finite
   cost, which then elected it leader over the crab that could actually help. */
var bfsMark=new Int32Array(NX*NY), bfsGen=0, bfsQ=new Int32Array(NX*NY);
function at(field,b){
  var i=idxAt(b.x,b.y);
  if(isFinite(field[i])) return field[i];
  bfsGen++;
  var head=0, tail=0, LIM=Math.ceil(0.45/CELL);
  bfsQ[tail++]=i; bfsMark[i]=bfsGen;
  var depth=new Int32Array(0);
  var d=new Map(); d.set(i,0);
  while(head<tail){
    var c=bfsQ[head++], cd=d.get(c);
    if(isFinite(field[c])) return field[c]+cd*CELL;
    if(cd>=LIM) continue;
    var cx=c%NX, cy=(c/NX)|0;
    for(var k=0;k<8;k++){
      var x=cx+DIRS[k][0], y=cy+DIRS[k][1];
      if(x<0||y<0||x>=NX||y>=NY) continue;
      var j=y*NX+x;
      if(bfsMark[j]===bfsGen||!phys[j]) continue;
      bfsMark[j]=bfsGen; d.set(j,cd+1); bfsQ[tail++]=j;
    }
  }
  return Infinity;
}
/* A crab that has rounded a corner can end up inside the planning margin, where
   the planner calls its own cell solid and it can never route anywhere again.
   Walk it back out to open floor first, then plan normally. */
/* WALLS ONLY. Judging this with imagined() counted the box's planning pad as
   a margin, and a crab pushing the box is inside that pad by definition: 556
   of 557 "stuck" triggers in a 3-crab sealed run were the box, not a wall. */
function wallStuck(i){ return known[i]===BLOCKED&&truth[i]===BLOCKED; }
function nearestPassable(x,y){
  var i=idxAt(x,y);
  if(!wallStuck(i)) return null;
  var ix=i%NX, iy=(i/NX)|0;
  for(var r=1;r<=14;r++){
    var okCell=function(k){ return !wallStuck(k)&&phys[k]&&!(boxSeen&&boxCell[k]); };
    var j=ring(ix,iy,r,okCell);
    if(j>=0){
      /* Far enough OUT to matter. The nearest open cell can sit inside the
         arrival tolerance; the crab "arrived" without moving, its centre still
         in the margin, and was declared stuck again 0.3 s later -- for 150 s. */
      var jx=(j%NX+0.5)*CELL, jy=(((j/NX)|0)+0.5)*CELL, ux=jx-x, uy=jy-y, un=Math.hypot(ux,uy)||1;
      var far={x:x+ux/un*0.18, y:y+uy/un*0.18}, fi=idxAt(far.x,far.y);
      if(okCell(fi)&&far.x>0&&far.y>0&&far.x<AW&&far.y<AH) return far;
      return {x:jx, y:jy};
    }
  }
  return null;
}
function routeTo(b,tx,ty,endDir){
  var bi=idxAt(b.x,b.y);
  useWeight=true;                       /* shape only -- keeps off the walls */
  flood(scratch, idxAt(tx,ty), function(i){ return i===bi||imagined(i); });
  useWeight=false;
  if(!isFinite(scratch[bi])) return null;
  /* drivable by construction; the cell-and-curve method is only the fallback
     when the search runs out of budget */
  var lat=latSearch(b.x,b.y,b.th,tx,ty,endDir?Math.atan2(endDir[1],endDir[0]):undefined);
  if(lat&&lat.length){ b.latOk=true; return lat; }
  /* The constrained approach is out of budget: drive to the spot on any
     heading, and the next cycle plans the short, constrained final approach
     from there. What must never come back is a cell path laid behind the crab. */
  if(endDir){ lat=latSearch(b.x,b.y,b.th,tx,ty,undefined); if(lat&&lat.length){ b.latOk=true; latNoEnd++; return lat; } }
  b.latOk=false; latFallbacks++;
  var p=descend(scratch, bi, knownPass);
  if(!p||!p.length) return null;
  var key=smooth(b.x,b.y,p,knownPass);
  key.push({x:tx,y:ty});
  return splineOf(b.x,b.y,b.th,key,endDir);   /* may be null: no drivable curve */
}
var latFallbacks=0, latNoEnd=0;
function nearestFrontier(b){
  if(!nFrontier) return null;
  var bi=idxAt(b.x,b.y);
  flood(scratch, bi, function(i){ return i===bi||imagined(i); });
  var best=-1, bv=Infinity;
  for(var i=0;i<frontier.length;i++){
    if(!frontier[i]||!isFinite(scratch[i])) continue;
    var claimed=false;
    for(var j=0;j<bots.length;j++)
      if(bots[j]!==b&&bots[j].claim===i){ claimed=true; break; }
    if(!claimed&&scratch[i]<bv){ bv=scratch[i]; best=i; }
  }
  if(best<0) return null;
  b.claim=best;
  /* face the unknown: the mean direction of the unknown neighbours, so a crab
     with a 30 degree cone arrives looking at what it came to see */
  var bx=best%NX, by=(best/NX)|0, fx=0, fy=0;
  for(var d=0;d<8;d++){ var X=bx+DIRS[d][0], Y=by+DIRS[d][1];
    if(X<0||Y<0||X>=NX||Y>=NY) continue;
    if(known[Y*NX+X]===UNKNOWN){ fx+=DIRS[d][0]; fy+=DIRS[d][1]; } }
  var fn=Math.hypot(fx,fy);
  return {x:(bx+0.5)*CELL, y:(by+0.5)*CELL, face:fn>1e-6?[fx/fn,fy/fn]:null};
}

var psCache=null, psAt=-99, psFor=-1, psBx=0, psBy=0;
function pushStatesCached(b){
  if(psCache&&psFor===b.id&&clock-psAt<1.2&&psBx===box.x&&psBy===box.y) return psCache;
  psCache=pushStates(b); psAt=clock; psFor=b.id; psBx=box.x; psBy=box.y;
  return psCache;
}
/* ---------- the goal board ----------
   Three goals can be open at once, and every open goal must have an owner:

     UNBLOCK  the reward is unreachable and moving the box would change that
     EXPLORE  frontier remains
     REACH    go and take the reward

   Ownership is LATCHED. A crab keeps its goal until the goal stops existing or
   the crab dies. Re-electing a single leader each cycle let the pusher lose its
   job halfway through a shove and wander off scouting.

   A crab holding REACH while the reward is still walled off does not idle and
   does not scout. It STAGES: it drives as far along the route that WOULD exist
   once the box is gone as it can currently stand, and waits there. When the
   pusher clears the gap that crab is already at the door. */
function goalsOpen(live){
  var need={unblock:false, explore:nFrontier>0, reach:true};
  for(var i=0;i<live.length;i++){
    var b=live[i];
    if(!isFinite(b.cost)&&isFinite(b.ideal)) need.unblock=true;
    else if(b.ideal<b.cost-PUSH_GAIN) need.unblock=true;
  }
  return need;
}
/* Furthest point along the box-free route a crab can presently stand. */
var reachB=new Float32Array(NX*NY), staged=[];
/* Crabs waiting for the gap must wait in DIFFERENT places. Backing each one
   further along the same line still stacks them nose to tail in a corridor
   0.56 m wide. */
function freeOfStaged(p){
  for(var i=0;i<staged.length;i++)
    if(Math.hypot(p.x-staged[i].x,p.y-staged[i].y)<0.52) return false;
  return true;
}
function stagePoint(b,setback){
  var bi=idxAt(b.x,b.y);
  var w=boxSeen; boxSeen=false;
  flood(scratch, goalIdx(), imagined);
  var route=descend(scratch, bi, imagined);
  boxSeen=w;
  if(!route||!route.length) return null;
  flood(reachB, bi, function(i){ return i===bi||imagined(i); });
  var last=-1;
  for(var i=0;i<route.length;i++){
    if(isFinite(reachB[idxAt(route[i].x,route[i].y)])) last=i; else break;
  }
  if(last<0) return null;
  var j=last, d=0;
  while(j>0&&d<setback){ d+=Math.hypot(route[j].x-route[j-1].x,route[j].y-route[j-1].y); j--; }
  /* step further back until this spot is not someone else's spot */
  while(j>0&&!freeOfStaged(route[j])) j--;
  var pick=route[j];
  /* still crowded at the head of the queue: sidestep perpendicular to the route */
  if(!freeOfStaged(pick)&&last>0){
    var a=route[Math.max(0,j-1)], t={x:pick.x-a.x,y:pick.y-a.y};
    var tn=Math.hypot(t.x,t.y)||1, px=-t.y/tn, py=t.x/tn;
    for(var k=1;k<=4;k++) for(var sgn=-1;sgn<=1;sgn+=2){
      var q={x:pick.x+px*0.3*k*sgn, y:pick.y+py*0.3*k*sgn};
      if(wallClear(q.x,q.y)>=PLAN_PAD&&imagined(idxAt(q.x,q.y))&&freeOfStaged(q)) { pick=q; k=9; break; }
    }
  }
  staged.push(pick);
  return pick;
}

function planAll(){
  var t0=performance.now(), i, b;
  for(i=0;i<boxIdx.length;i++) if(known[boxIdx[i]]===BLOCKED){ boxSeen=true; break; }
  fields();
  var live=[];
  for(i=0;i<bots.length;i++){
    b=bots[i];
    if(b.done){ b.job=null; continue; }
    b.cost=at(dist,b); b.ideal=at(distNoBox,b); live.push(b);
  }
  if(!live.length){ planMs=performance.now()-t0; return; }
  var need=goalsOpen(live);
  staged.length=0;

  /* release goals that have closed, and any duplicate claim on a single-owner goal */
  var unb=null, rch=null;
  for(i=0;i<live.length;i++){
    b=live[i];
    if(b.job==="UNBLOCK"){
      /* A crab that has begun a shove is never released. That is what latched
         ownership means, and without it the job bounced 20-odd times a run. */
      var committed=b.goalBox&&(clock-b.pushStart)<40;
      if(committed&&!unb){ unb=b; need.unblock=true; }
      else if(need.unblock&&!unb) unb=b;
      else { b.job=null; b.goalBox=null; b.announced=false; }
    }
    else if(b.job==="REACH"){ if(!rch) rch=b; else b.job=null; }
    else if(b.job==="EXPLORE"&&!need.explore) b.job=null;
  }
  /* every open goal must end this function with an owner */
  if(need.unblock&&!unb){
    var nd=Infinity;
    for(i=0;i<live.length;i++){
      var d=Math.hypot(live[i].x-box.x,live[i].y-box.y);
      if(live[i].job==="REACH"&&live.length>1) continue;   /* leave REACH held */
      if(d<nd){ nd=d; unb=live[i]; }
    }
    if(unb) unb.job="UNBLOCK";
  }
  /* UNBLOCK is only an open goal if moving the box beats the best walk the TEAM
     already has. Opening it merely because the box is in the way of the SHORT
     route makes a crab shove a box that nobody needed moved. */
  if(need.unblock&&unb&&!(unb.goalBox&&(clock-unb.pushStart)<40)){
    /* Judge the trade against the OWNER's own route, not the team minimum.
       With unknown space assumed open, the team minimum is whatever the most
       deluded crab believes, and the trade never looks worth taking. */
    var cheapest=unb.cost;
    var q=pushStatesCached(unb);
    unb.cands=q; lastStates=q;
    var pays=q.best&&(!isFinite(cheapest)||q.best.total<cheapest-PUSH_GAIN);
    if(!pays){
      /* Give the job back. Nulling it here turned a lone crab's job into
         EXPLORE, and the next cycle re-elected it REACH: the job alternated
         every cycle and the crab shuttled between two routes. */
      need.unblock=false; unb.job=(unb===rch)?"REACH":null; unb.goalBox=null;
      unb.why=q.best?"walking round is cheaper than moving it":"no placement clears the gap";
      unb=null;
    } else if(!unb.announced){ unb.announced=true; say("crab "+(unb.id+1)+" takes UNBLOCK: "+
      (isFinite(cheapest)?cheapest.toFixed(1)+" m round vs "+q.best.total.toFixed(1)+" m through"
                         :"no route exists without moving the box")); }
  }
  if(!rch){
    var cv=Infinity, pick=null;
    for(i=0;i<live.length;i++){
      if(live[i]===unb) continue;
      var c=isFinite(live[i].cost)?live[i].cost:live[i].ideal+50;
      if(c<cv){ cv=c; pick=live[i]; }
    }
    if(pick){ pick.job="REACH"; rch=pick; }
    else if(unb&&live.length===1) unb.job="UNBLOCK";
  }
  for(i=0;i<live.length;i++) if(!live[i].job)
    live[i].job = need.explore ? "EXPLORE" : "STAGE";

  var queue=0;
  for(i=0;i<live.length;i++) route1(live[i], live[i]===unb?queue++:queue++);
  planMs=performance.now()-t0;
}

function route1(b,qi){
  var e,f; b.plannedAt=clock;
  /* a push, once begun, runs to completion */
  if(b.job==="UNBLOCK"&&b.goalBox&&b.mode==="PUSH"&&clock-b.pushStart<40){
    /* Remaining travel ALONG the push axis. A radius round the target point
       read a 3 cm sideways drift as "not there yet", and the crab was sent back
       to a pushing position computed before the box had moved. */
    var left=b.axis?((b.goalBox.x-box.x)*b.axis.dx+(b.goalBox.y-box.y)*b.axis.dy)
                   :Math.hypot(box.x-b.goalBox.x,box.y-b.goalBox.y);
    if(left>0.035&&b.wi<b.path.length&&!b.stalled) return;
    if(left<=0.035){
      say("crab "+(b.id+1)+" cleared the way: box moved "+
        Math.hypot(box.x-BOX0.x,box.y-BOX0.y).toFixed(2)+" m");
      /* Hand the job back. Falling through to the UNBLOCK branch here started
         a second latched shove with no "does it pay" judgement: the box went
         3.31 m down the arena instead of 0.87 m. Only planAll opens a shove. */
      b.goalBox=null; b.job=null; b.announced=false; b.mode="IDLE"; b.path=[]; b.wi=0;
      b.note="gap open -- awaiting the next goal"; return;
    }
  }
  b.stalled=false;
  /* Escaping a margin never pre-empts a shove in progress. */
  var esc=(b.job==="UNBLOCK"&&b.goalBox)?null:nearestPassable(b.x,b.y);
  if(esc){
    b.mode="UNSTICK"; b.wi=0; b.claim=-1; b.nUnstick++;
    b.note="inside a wall margin -- driving out";
    /* The search starts inside the margin, so it is allowed onto real floor
       near the start; what it returns is a way out the chassis can drive. */
    latEscape=true; var lp=latSearch(b.x,b.y,b.th,esc.x,esc.y,undefined);
    if(lp&&lp.length){ b.path=lp; return; }
    b.nUnstickEvade++;
    if(b.evade<clock){ b.evade=clock+0.9; b.evadeTurn=(b.id%2?1:-1)*1.1; }
    b.path=[esc]; return;
  }

  if(b.job==="UNBLOCK"){
    /* placements are only good for the box position they were scored at */
    var ps=(b.cands&&b.cands.atBx===box.x&&b.cands.atBy===box.y)?b.cands:pushStatesCached(b);
    if(ps.best){
      var dv=PUSHDIR[ps.best.dir];
      var r=approachPush(b,ps.best.stand,dv,ps.best.len);
      if(r){
        b.mode="PUSH"; b.path=r; b.wi=0; b.claim=-1;
        /* a re-approach onto the SAME placement is not a new push */
        if(!(b.goalBox&&Math.abs(b.goalBox.x-ps.best.x)<1e-6&&Math.abs(b.goalBox.y-ps.best.y)<1e-6)) b.pushStart=clock;
        b.goalBox={x:ps.best.x,y:ps.best.y};
        b.axis={x:ps.best.stand.x,y:ps.best.stand.y,dx:dv[0],dy:dv[1]};
        b.note=isFinite(b.cost)
          ? b.cost.toFixed(1)+" m around the box, "+ps.best.total.toFixed(1)+" m through it"
          : "no route exists -- moving the box makes one";
        return;
      }
      b.note="no clear approach onto the push axis";
    } else b.note="no placement clears the gap";
    b.mode="UNBLOCK?"; b.path=[]; return;
  }

  if(b.job==="EXPLORE"){
    f=nearestFrontier(b);
    if(f&&(e=routeTo(b,f.x,f.y,(Math.hypot(f.x-b.x,f.y-b.y)<0.6)?f.face:undefined))){
      b.mode="SCOUT"; b.path=e; b.wi=0; b.note="edge of the known world"; return;
    }
    b.job="STAGE";
  }

  if(b.job==="REACH"&&isFinite(b.cost)){
    if((e=routeTo(b,rewardTgt.x,rewardTgt.y))){
      b.mode="GOTO"; b.path=e; b.wi=0; b.claim=-1;
      b.note=(rewardFound?"route to the reward, ":"route to where the reward may be, ")+b.cost.toFixed(1)+" m"; return;
    }
  }

  /* STAGE -- queue at the door for the gap to open */
  var sp=stagePoint(b,0.70+0.58*qi);
  if(sp&&(e=routeTo(b,sp.x,sp.y))){
    b.mode="STAGE"; b.path=e; b.wi=0; b.claim=-1;
    b.note="waiting at the gap; "+(isFinite(b.ideal)?b.ideal.toFixed(1)+" m once it opens":"shorter once it opens");
    return;
  }
  f=nearestFrontier(b);
  if(f&&(e=routeTo(b,f.x,f.y,(Math.hypot(f.x-b.x,f.y-b.y)<0.6)?f.face:undefined))){
    b.mode="SCOUT"; b.path=e; b.wi=0; b.note="nothing to wait for -- scouting"; return;
  }
  b.mode="IDLE"; b.path=[]; b.claim=-1; b.note="nothing to do";
}

/* ---------- loop ---------- */
function reset(n,seal){
  if(seal!==undefined) sealed=!!seal;
  clock=0; goalFound=true; reached=0; pushes=0; pushed=0; pushClaim=null;
  narration=[]; lastStates=null; planTimer=0; boxSeen=false; psCache=null; psAt=-99;
  box.x=BOX0.x; box.y=BOX0.y; box.th=0;
  known.fill(UNKNOWN); buildTruth(); stampBox(); seedRewardBelief();
  bots=[]; for(var i=0;i<n;i++) bots.push(new Bot(i,STARTS[i%STARTS.length]));
  for(i=0;i<bots.length;i++) observe(bots[i]);
  planAll();
  say("crab team of "+n+" released; reward is tagged, the route is not known");
}
function tick(dt){
  clock+=dt;
  var i, need=false;
  for(i=0;i<bots.length;i++){
    var b=bots[i];
    stepBot(b,bots,dt); observe(b);
    if(!b.done&&Math.hypot(b.x-GOAL.x,b.y-GOAL.y)<0.18){
      b.done=true; b.path=[]; b.mode="AT REWARD"; b.note="arrived at "+clock.toFixed(1)+" s";
      reached++; say("crab "+(i+1)+" reached the reward at "+clock.toFixed(1)+" s");
      if(pushClaim===i) pushClaim=null;
      for(var j=0;j<bots.length;j++) if(!bots[j].done){
        bots[j].done=true; bots[j].path=[]; bots[j].mode="STAND DOWN";
        bots[j].note="reward claimed by crab "+(i+1);
      }
    }
    if(!b.done&&b.mode!=="IDLE"&&b.wi>=b.path.length&&clock-b.plannedAt>0.3) need=true;
  }
  planTimer-=dt;
  if(planTimer<=0) need=true;
  if(need&&reached===0){
    planAll(); planTimer=0.45;
  }
}
function explored(){
  var seen=0,tot=0;
  for(var i=0;i<truth.length;i++) if(truth[i]===FREE){ tot++; if(known[i]!==UNKNOWN) seen++; }
  return tot?seen/tot:0;
}

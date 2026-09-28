/* ---------- geometry ---------- */
function segDist(px,py,x1,y1,x2,y2){
  var dx=x2-x1, dy=y2-y1, L=dx*dx+dy*dy;
  if(L<1e-12) return Math.hypot(px-x1,py-y1);
  var t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/L));
  return Math.hypot(px-(x1+t*dx), py-(y1+t*dy));
}
/* Distance from a point to a rotated rectangle's surface, 0 inside. */
function rectDist(px,py,r){
  var ca=Math.cos(-r.th), sa=Math.sin(-r.th);
  var dx=px-r.x, dy=py-r.y;
  var lx=dx*ca-dy*sa, ly=dx*sa+dy*ca;
  var ox=Math.abs(lx)-r.w/2, oy=Math.abs(ly)-r.h/2;
  if(ox<=0&&oy<=0) return 0;
  return Math.hypot(Math.max(ox,0),Math.max(oy,0));
}
function rectCorners(r){
  var c=Math.cos(r.th), s=Math.sin(r.th), hw=r.w/2, hh=r.h/2, out=[];
  var P=[[hw,hh],[hw,-hh],[-hw,-hh],[-hw,hh]];
  for(var i=0;i<4;i++) out.push([r.x+P[i][0]*c-P[i][1]*s, r.y+P[i][0]*s+P[i][1]*c]);
  return out;
}
/* Separating-axis test between two rotated rectangles. */
function rectsOverlap(a,b){
  var A=rectCorners(a), B=rectCorners(b);
  var axes=[[Math.cos(a.th),Math.sin(a.th)],[-Math.sin(a.th),Math.cos(a.th)],
            [Math.cos(b.th),Math.sin(b.th)],[-Math.sin(b.th),Math.cos(b.th)]];
  for(var k=0;k<4;k++){
    var ax=axes[k][0], ay=axes[k][1];
    var a0=Infinity,a1=-Infinity,b0=Infinity,b1=-Infinity;
    for(var i=0;i<4;i++){ var p=A[i][0]*ax+A[i][1]*ay; if(p<a0)a0=p; if(p>a1)a1=p; }
    for(var j=0;j<4;j++){ var q=B[j][0]*ax+B[j][1]*ay; if(q<b0)b0=q; if(q>b1)b1=q; }
    if(a1<b0||b1<a0) return false;
  }
  return true;
}
/* A rectangle against a wall segment. Rotate the segment into the rectangle's
   own frame, where the rectangle is axis-aligned, then clip (Liang-Barsky).
   O(1). Sampling the segment every 3 cm instead cost ~75 million distance
   calls over one mission and ran the simulation at a third of real time. */
function rectHitsSeg(r,S){
  var ca=Math.cos(-r.th), sa=Math.sin(-r.th);
  var ax=S[0]-r.x, ay=S[1]-r.y, bx=S[2]-r.x, by=S[3]-r.y;
  var x0=ax*ca-ay*sa, y0=ax*sa+ay*ca;
  var x1=bx*ca-by*sa, y1=bx*sa+by*ca;
  var dx=x1-x0, dy=y1-y0, hw=r.w/2, hh=r.h/2;
  var t0=0, t1=1;
  var P=[-dx,dx,-dy,dy], Q=[x0+hw,hw-x0,y0+hh,hh-y0];
  for(var i=0;i<4;i++){
    if(Math.abs(P[i])<1e-12){ if(Q[i]<0) return false; continue; }
    var t=Q[i]/P[i];
    if(P[i]<0){ if(t>t1) return false; if(t>t0) t0=t; }
    else      { if(t<t0) return false; if(t<t1) t1=t; }
  }
  return t0<=t1;
}

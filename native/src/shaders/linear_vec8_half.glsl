#version 450 core
layout(local_size_x=16,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{vec4 d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{uint d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint ic;uint oc;}p;
#define KV 8
shared vec4 it[56*KV];shared vec4 wt[64*KV];
vec4 rw(uint v){vec2 a=unpackHalf2x16(w.d[v*2]),c=unpackHalf2x16(w.d[v*2+1]);
 return vec4(a,c);}
void main(){
 uint cb=gl_WorkGroupID.x*64+gl_LocalInvocationID.x*4;
 uint rb=gl_WorkGroupID.y*56+gl_LocalInvocationID.y*7;
 float s[7][4];for(uint r=0;r<7;++r)for(uint c=0;c<4;++c)s[r][c]=0;
 uint lane=gl_LocalInvocationID.y*16+gl_LocalInvocationID.x,iv=p.ic/4;
 for(uint kb=0;kb<iv;kb+=KV){
  for(uint n=lane;n<56*KV;n+=128){uint tr=n/KV,k=kb+n%KV,r=gl_WorkGroupID.y*56+tr;
   it[n]=r<p.rows&&k<iv?i.d[r*iv+k]:vec4(0);}
  for(uint n=lane;n<64*KV;n+=128){uint tc=n/KV,k=kb+n%KV,c=gl_WorkGroupID.x*64+tc;
   wt[n]=c<p.oc&&k<iv?rw(c*iv+k):vec4(0);}
  barrier();uint kc=min(KV,iv-kb);
  for(uint k=0;k<kc;++k){vec4 a[7],q[4];
   for(uint r=0;r<7;++r)a[r]=it[(gl_LocalInvocationID.y*7+r)*KV+k];
   for(uint c=0;c<4;++c)q[c]=wt[(gl_LocalInvocationID.x*4+c)*KV+k];
   for(uint r=0;r<7;++r)for(uint c=0;c<4;++c)s[r][c]+=dot(a[r],q[c]);}
  barrier();}
 for(uint r=0;r<7;++r){uint rr=rb+r;if(rr>=p.rows)continue;
  for(uint c=0;c<4;++c){uint cc=cb+c;if(cc<p.oc)o.d[rr*p.oc+cc]=s[r][c]+b.d[cc];}}
}

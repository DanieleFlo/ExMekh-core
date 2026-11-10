// CPU BF16 boundaries and ATen FP32 softmax reduction, bounded workspace.
inline void nm_cpu_attention(constant ds4_nm_args &a,device uchar *y,
    device const uchar *q,device const uchar *k,device const uchar *v,
    device const uint *bounds,uint g,uint tid,
    threadgroup float *scores,threadgroup float *prob,threadgroup float *stats) {
    uint dim=a.u[5],heads=a.u[3],kvh=a.u[4],keys=a.u[2],queries=a.u[1];
    uint h=g%heads,row=g/heads,bt=row/queries;
    uint lo=bounds[2*row],hi=bounds[2*row+1],warp=tid/32,lane=tid%32;
    ulong qi=ulong(g)*dim;
    float maximum=-INFINITY,denom[4]={},acc0[4]={},acc1[4]={};
    for(uint pass=0;pass<3;pass++) {
        for(uint base=lo;base<hi;base+=8) {
            uint key=base+warp;
            float z=-INFINITY;
            if(key<hi) {
                ulong ki=((ulong(bt)*keys+key)*kvh+h/(heads/kvh))*dim;
                float dot=0;
                for(uint d=lane;d<dim;d+=32)
                    dot=fma(rd(q,qi+d,a.t[2]),rd(k,ki+d,a.t[3]),dot);
                dot+=simd_shuffle_down(dot,16);
                dot+=simd_shuffle_down(dot,8);
                dot+=simd_shuffle_down(dot,4);
                dot=(simd_shuffle(dot,0)+simd_shuffle(dot,1))+
                    (simd_shuffle(dot,2)+simd_shuffle(dot,3));
                z=rnd(rnd(dot,a.t[2])*a.f[0],a.t[2]);
                if(a.f[1]>0)z=a.f[1]*stable_tanh(z/a.f[1]);
            }
            if(!lane)scores[warp]=z;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(pass==0) {
                if(!tid) {
                    for(uint j=0;j<8 && base+j<hi;j++)maximum=max(maximum,scores[j]);
                    stats[0]=maximum;
                }
            } else if(pass==1) {
                if(!tid) {
                    for(uint j=0;j<8 && base+j<hi;j++)denom[(base+j)%4]+=exp(scores[j]-stats[0]);
                }
            } else {
                if(tid<8)prob[tid]=base+tid<hi ? rnd(exp(scores[tid]-stats[0])*stats[1],a.t[2]):0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for(uint j=0;j<8 && base+j<hi;j++) {
                    ulong vi=((ulong(bt)*keys+base+j)*kvh+h/(heads/kvh))*dim;
                    uint s=base+j<keys/4*4 ? (base+j)%4:0;
                    if(tid<dim)acc0[s]+=prob[j]*rd(v,vi+tid,a.t[4]);
                    if(tid+256<dim)acc1[s]+=prob[j]*rd(v,vi+tid+256,a.t[4]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(pass==1 && !tid)stats[1]=1.0f/((denom[0]+denom[2])+(denom[1]+denom[3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<dim)wr(y,qi+tid,a.t[1],((acc0[0]+acc0[1])+acc0[2])+acc0[3]);
    if(tid+256<dim)wr(y,qi+tid+256,a.t[1],((acc1[0]+acc1[1])+acc1[2])+acc1[3]);
}

kernel void nm_softmax(constant ds4_nm_args &a [[buffer(0)]],device uchar *y [[buffer(1)]],
    device const uchar *x [[buffer(2)]],device const uint *bounds [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float stats[2];
    uint width=a.u[0],lo=bounds[2*row],hi=bounds[2*row+1];
    if(!tid) {
        float maximum=-INFINITY,sum[4]={};
        for(uint j=lo;j<hi;j++)maximum=max(maximum,rnd(rd(x,ulong(row)*width+j,a.t[2])*a.f[0],a.t[2]));
        for(uint j=lo;j<hi;j++)sum[j%4]+=nm_sleef_exp(rnd(rd(x,ulong(row)*width+j,a.t[2])*a.f[0],a.t[2])-maximum);
        stats[0]=maximum;stats[1]=1.0f/((sum[0]+sum[2])+(sum[1]+sum[3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<width;j+=256)wr(y,ulong(row)*width+j,a.t[1],j<lo || j>=hi ? 0:
        nm_sleef_exp(rnd(rd(x,ulong(row)*width+j,a.t[2])*a.f[0],a.t[2])-stats[0])*stats[1]);
}

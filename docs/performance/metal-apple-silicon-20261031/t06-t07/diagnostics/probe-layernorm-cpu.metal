// Independently expressed FP32 Welford order from pinned ATen CPU BF16.
// Four lanes represent the two halves of NEON's eight-BF16 vector.
struct nm_cpu_moment { uint count; float mean; float m2; };
inline nm_cpu_moment nm_cpu_moment_vector_join(nm_cpu_moment left,nm_cpu_moment right) {
    uint count=left.count+right.count;
    float c=count ? float(right.count)/float(count):0;
    float delta=right.mean-left.mean;
    float m2=fma(delta*float(left.count),c*delta,left.m2+right.m2);
    return {count,left.mean+c*delta,m2};
}
inline nm_cpu_moment nm_cpu_moment_scalar_join(nm_cpu_moment left,nm_cpu_moment right) {
    uint count=left.count+right.count;
    float c=count ? float(right.count)/float(count):0;
    float delta=right.mean-left.mean;
    return {count,left.mean+c*delta,left.m2+(right.m2+delta*delta*c*float(left.count))};
}
kernel void nm_bf16_layernorm(constant ds4_nm_args &a [[buffer(0)]],
    device uchar *y [[buffer(1)]],device const uchar *x [[buffer(2)]],
    device const uchar *weight [[buffer(3)]],device const uchar *bias [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    if(lane>=4)return;
    uint width=a.u[0],vectors=width/8,chunks=(vectors+15)/16;
    uint depth=chunks>1 ? 32-clz(chunks-1):0;
    ulong base=ulong(row)*width;
    nm_cpu_moment stack[16];
    for(uint i=0;i<16;i++)stack[i]={0,0,0};
    for(uint block=0;block<chunks;block++) {
        uint count=min(16u,vectors-block*16);
        nm_cpu_moment part0={count,0,0},part1={count,0,0};
        for(uint j=0;j<count;j++) {
            float c=1.0f/float(j+1);
            ulong at=base+ulong(block*16+j)*8+lane;
            float x0=rd(x,at,a.t[2]),x1=rd(x,at+4,a.t[2]);
            float d0=x0-part0.mean,d1=x1-part1.mean;
            part0.mean=fma(d0,c,part0.mean);part1.mean=fma(d1,c,part1.mean);
            part0.m2=fma(d0,x0-part0.mean,part0.m2);
            part1.m2=fma(d1,x1-part1.mean,part1.m2);
        }
        stack[0]=nm_cpu_moment_vector_join(stack[0],part0);
        stack[0]=nm_cpu_moment_vector_join(stack[0],part1);
        uint mask=block+1;
        for(uint level=1;level<depth && (mask&1)==0;level++) {
            stack[level]=nm_cpu_moment_vector_join(stack[level],stack[level-1]);
            stack[level-1]={0,0,0};mask>>=1;
        }
    }
    for(uint i=1;i<depth;i++)stack[0]=nm_cpu_moment_vector_join(stack[0],stack[i]);
    nm_cpu_moment total={0,0,0};
    for(uint j=vectors*8;j<width;j++) {
        float value=rd(x,base+j,a.t[2]),delta=value-total.mean;
        total.count++;total.mean+=delta/float(total.count);
        total.m2+=delta*(value-total.mean);
    }
    for(uint j=0;j<4;j++) {
        nm_cpu_moment part={vectors*2,simd_shuffle(stack[0].mean,j),simd_shuffle(stack[0].m2,j)};
        total=nm_cpu_moment_scalar_join(total,part);
    }
    float factor=1.0f/precise::sqrt(total.m2/float(width)+a.f[0]);
    float shift=-factor*total.mean;
    simdgroup_barrier(mem_flags::mem_device);
    for(uint j=lane;j<width;j+=4) {
        float z=rd(x,base+j,a.t[2])*factor+shift;
        if(a.u[2])z*=rd(weight,j,a.t[3])+(a.u[2]==2 ? 1.0f:0.0f);
        if(a.u[4])z+=rd(bias,j,a.t[4]);
        wr(y,base+j,a.t[1],z);
    }
}

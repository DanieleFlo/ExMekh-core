// Fixed BF16 RMS row reduction, independently expressed with SIMD shuffles.
// The cascade and final scalar order follow the pinned CPU oracle's sum.
kernel void nm_bf16_norm(constant ds4_nm_args &a [[buffer(0)]],
                        device uchar *y [[buffer(1)]],
                        device const uchar *x [[buffer(2)]],
                        device const uchar *weight [[buffer(3)]],
                        device const uchar *bias [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
    if (lane>=32) return;
    uint width=a.u[0];ulong base=ulong(row)*width;
    float levels[4]={};
    uint count=width/32;
    for(uint i=0;i<count;i++) {
        float value=rd(x,base+i*32+lane,a.t[2]);
        levels[0]+=value*value;
        if((i+1)%16==0) {
            for(uint level=1;level<4;level++) {
                levels[level]+=levels[level-1];levels[level-1]=0;
                if((i+1)&(15u<<(level*4))) break;
            }
        }
    }
    float value=((levels[0]+levels[1])+levels[2])+levels[3];
    uint pairs=(width/4-count*8)/2;
    if(lane<8) {
        for(uint i=0;i<pairs;i++) {
            uint j=(count*8+i*2+lane/4)*4+lane%4;
            float z=rd(x,base+j,a.t[2]);value+=z*z;
        }
    }
    float grouped=simd_shuffle(value,lane%8);
    grouped+=simd_shuffle(value,lane%8+8);
    grouped+=simd_shuffle(value,lane%8+16);
    grouped+=simd_shuffle(value,lane%8+24);
    float sum=0;
    for(uint j=(count*8+pairs*2)*4;j<width;j++) {
        float z=rd(x,base+j,a.t[2]);sum+=z*z;
    }
    for(uint j=0;j<8;j++) sum+=simd_shuffle(grouped,j);
    float factor=pow(sum/width+a.f[0],-0.5f);
    simdgroup_barrier(mem_flags::mem_device);
    for(uint j=lane;j<width;j+=32) {
        float z=rd(x,base+j,a.t[2])*factor;
        if(a.u[3]) z=rnd(z,a.t[2]);
        if(a.u[2]) z*=rd(weight,j,a.t[3])+(a.u[2]==2 ? 1.0f : 0.0f);
        if(a.u[4]) z+=rd(bias,j,a.t[4]);
        wr(y,base+j,a.t[1],z);
    }
}

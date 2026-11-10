kernel void nm_bf16_channel_rms(constant ds4_nm_args &a [[buffer(0)]],
    device uchar *y [[buffer(1)]],device const uchar *x [[buffer(2)]],
    device const uchar *weight [[buffer(3)]],uint p [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    if(lane>=4)return;
    uint channels=a.u[0],spatial=a.u[1];
    bool wide=p<spatial/32*32;
    uint streams=wide ? 1:4,count=channels/streams;
    uint power=max(4u,(count>1 ? 32u-clz(count-1):0u)/4u),step=1u<<power,mask=step-1;
    float levels[4]={};
    for(uint i=0;i<count && lane<streams;i++) {
        float value=rd(x,ulong(i*streams+lane)*spatial+p,a.t[2]);
        levels[0]+=value*value;
        if((i+1)%step==0)
            for(uint level=1;level<4;level++) {
                levels[level]+=levels[level-1];levels[level-1]=0;
                if((i+1)&(mask<<(level*power)))break;
            }
    }
    float part=((levels[0]+levels[1])+levels[2])+levels[3];
    if(!lane)
        for(uint c=count*streams;c<channels;c++) {
            float value=rd(x,ulong(c)*spatial+p,a.t[2]);part+=value*value;
        }
    float sum=wide ? simd_shuffle(part,0) : ((simd_shuffle(part,0)+simd_shuffle(part,1))+simd_shuffle(part,2))+simd_shuffle(part,3);
    float factor=1.0f/precise::sqrt(sum/channels+a.f[0]);
    for(uint c=lane;c<channels;c+=4) {
        float value=rd(x,ulong(c)*spatial+p,a.t[2])*factor;
        wr(y,ulong(p)*channels+c,a.t[1],value*rd(weight,c,a.t[3]));
    }
}

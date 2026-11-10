// Ordered BF16 dot: full NEON blocks, paired vector tail, then scalar tail.
inline float nm_bf16_attention_dot(device const uchar *q,device const uchar *k,
    ulong qi,ulong ki,uint dim,uint lane,bool four_streams) {
#pragma clang fp contract(off)
    float dot=0;
    if(four_streams) {
        float partial=0;
        if(lane<4)
            for(uint d=lane;d<dim;d+=4)
                partial=fma(rd(q,qi+d,30),rd(k,ki+d,30),partial);
        dot=((simd_shuffle(partial,0)+simd_shuffle(partial,1))+
             simd_shuffle(partial,2))+simd_shuffle(partial,3);
    } else {
        uint aligned=dim/32*32,aligned8=dim/8*8;
        for(uint d=lane;d<aligned;d+=32)
            dot=fma(rd(q,qi+d,30),rd(k,ki+d,30),dot);
        dot+=simd_shuffle_down(dot,16);
        dot+=simd_shuffle_down(dot,8);
        dot+=simd_shuffle_down(dot,4);
        dot=(simd_shuffle(dot,0)+simd_shuffle(dot,1))+
            (simd_shuffle(dot,2)+simd_shuffle(dot,3));
        float tail=0;
        if(lane<4)
            for(uint d=aligned;d<aligned8;d+=8)
                tail=fma(rd(q,qi+d+lane+4,30),rd(k,ki+d+lane+4,30),
                     fma(rd(q,qi+d+lane,30),rd(k,ki+d+lane,30),tail));
        dot+=(simd_shuffle(tail,0)+simd_shuffle(tail,1))+
             (simd_shuffle(tail,2)+simd_shuffle(tail,3));
        if(!lane)
            for(uint d=aligned8;d<dim;d++)
                dot=fma(rd(q,qi+d,30),rd(k,ki+d,30),dot);
        dot=simd_shuffle(dot,0);
    }
    return dot;
}

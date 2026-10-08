#!/usr/bin/env python3
"""Measure exact QSA selection ideas before changing production dispatch.

The oracle is extracted verbatim from the current DS4 selector. The candidate
is an independent implementation of Strata's register-resident key idea;
it retains DS4's greater-then-equal stable order, including zero-score ties.
CUDA event timings are primitive diagnostics, not whole-forward benchmarks.
"""
import argparse
from pathlib import Path
import subprocess
from perf_qwen_flash_strata import ROOT, digest, write_json

CUDA = r'''
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstdint>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(1); }} while(0)

ORACLE

template<unsigned NT, unsigned PER>
__global__ __launch_bounds__(NT) void cached_select(int *out, const float *score, unsigned N, unsigned K) {
    const unsigned tid=threadIdx.x, t=blockIdx.x, lane=tid&31u, warp=tid/32u;
    __shared__ unsigned hist[NT/32][256], gt[NT/32], eq[NT/32], threshold, need;
    const unsigned chunk=(N+NT-1)/NT, begin=min(N,tid*chunk), end=min(N,begin+chunk);
    unsigned keys[PER];
    #pragma unroll
    for(unsigned j=0;j<PER;j++) keys[j]=begin+j<end ? __float_as_uint(fmaxf(score[(uint64_t)t*N+begin+j],0)) : 0;
    if(!tid) { threshold=0; need=K; }
    __syncthreads();
    for(unsigned pass=0;pass<4;pass++) {
        unsigned shift=24-8*pass, mask=pass ? (0xffffffffu<<(shift+8)) : 0;
        for(unsigned i=tid;i<(NT/32)*256;i+=NT) ((unsigned*)hist)[i]=0;
        __syncthreads();
        #pragma unroll
        for(unsigned j=0;j<PER;j++) if(begin+j<end && (keys[j]&mask)==threshold)
            atomicAdd(&hist[warp][(keys[j]>>shift)&255],1u);
        __syncthreads();
        if(tid<256) {
            unsigned total=0;
            for(unsigned w=0;w<NT/32;w++) total+=hist[w][tid];
            hist[0][tid]=total;
        }
        __syncthreads();
        if(!tid) for(int d=255;d>=0;d--) {
            if(hist[0][d]>=need) { threshold|=(unsigned)d<<shift; break; }
            need-=hist[0][d];
        }
        __syncthreads();
    }
    unsigned ng=0,ne=0;
    #pragma unroll
    for(unsigned j=0;j<PER;j++) if(begin+j<end) { ng+=keys[j]>threshold; ne+=keys[j]==threshold; }
    unsigned pg=ng,pe=ne;
    for(unsigned off=1;off<32;off*=2) {
        unsigned g=__shfl_up_sync(0xffffffffu,pg,off), e=__shfl_up_sync(0xffffffffu,pe,off);
        if(lane>=off) { pg+=g; pe+=e; }
    }
    if(lane==31) { gt[warp]=pg; eq[warp]=pe; }
    __syncthreads();
    if(tid<32) {
        unsigned g=tid<NT/32 ? gt[tid] : 0, e=tid<NT/32 ? eq[tid] : 0, oldg=g, olde=e;
        for(unsigned off=1;off<32;off*=2) {
            unsigned a=__shfl_up_sync(0xffffffffu,g,off), b=__shfl_up_sync(0xffffffffu,e,off);
            if(lane>=off) { g+=a; e+=b; }
        }
        if(tid<NT/32) { gt[tid]=g-oldg; eq[tid]=e-olde; }
    }
    __syncthreads();
    unsigned g=gt[warp]+pg-ng, e=eq[warp]+pe-ne;
    #pragma unroll
    for(unsigned j=0;j<PER;j++) if(begin+j<end) {
        if(keys[j]>threshold) out[(uint64_t)t*K+g++]=begin+j;
        else if(keys[j]==threshold) { if(e<need) out[(uint64_t)t*K+K-need+e]=begin+j; e++; }
    }
}
void launch(unsigned mode,unsigned T,int* out,float* score,unsigned N,unsigned K) {
    if(mode==0) idx_select<<<T,256>>>(out,score,N,K);
    if(mode==1) cached_select<1024,33><<<T,1024>>>(out,score,N,K);
    if(mode==2) cached_select<512,64><<<T,512>>>(out,score,N,K);
}
int main() {
    std::mt19937 rng(42); cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    puts("["); bool first=true;
    for(unsigned T: {1u,3u}) for(unsigned N: {513u,2048u,16384u,32768u}) for(unsigned pattern=0;pattern<3;pattern++) {
        unsigned K=512; std::vector<float> scores((uint64_t)T*N);
        for(unsigned t=0;t<T;t++) for(unsigned i=0;i<N;i++) {
            float value=pattern==0 ? (rng()%100000)/997.f : pattern==1 ? (float)(i%7) : 0;
            scores[(uint64_t)t*N+i]=i>=N-t-4 ? -3e38f : value;
        }
        float* dscore; int* out; CK(cudaMalloc(&dscore,scores.size()*4)); CK(cudaMalloc(&out,T*K*4));
        CK(cudaMemcpy(dscore,scores.data(),scores.size()*4,cudaMemcpyHostToDevice));
        std::vector<int> ref(T*K),got(T*K); launch(0,T,out,dscore,N,K); CK(cudaGetLastError());
        CK(cudaMemcpy(ref.data(),out,T*K*4,cudaMemcpyDeviceToHost));
        double us[3];
        for(unsigned mode=0;mode<3;mode++) {
            launch(mode,T,out,dscore,N,K); CK(cudaGetLastError());
            CK(cudaMemcpy(got.data(),out,T*K*4,cudaMemcpyDeviceToHost));
            if(ref!=got) { fprintf(stderr,"ordered IDs differ T=%u N=%u pattern=%u mode=%u\n",T,N,pattern,mode); return 1; }
            for(unsigned i=0;i<20;i++) launch(mode,T,out,dscore,N,K);
            std::vector<float> samples;
            for(unsigned s=0;s<21;s++) {
                CK(cudaEventRecord(a)); for(unsigned i=0;i<100;i++) launch(mode,T,out,dscore,N,K);
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b)); samples.push_back(ms*10);
            }
            std::sort(samples.begin(),samples.end()); us[mode]=samples[10];
        }
        printf("%s{\"rows\":%u,\"blocks\":%u,\"pattern\":%u,\"ordered_ids_exact\":true,\"original_us\":%.3f,\"cached1024_us\":%.3f,\"cached512_us\":%.3f}",first ? "" : ",\n",T,N,pattern,us[0],us[1],us[2]); first=false;
        CK(cudaFree(out)); CK(cudaFree(dscore));
    }
    puts("\n]"); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "ds4_qwen4_cuda.cuh"
    source = path.read_text()
    oracle = source[source.index("__global__ void idx_select("):source.index("__global__ void idx_expand(")]
    generated = a.output / "probe.cu"; generated.write_text(CUDA.replace("ORACLE", oracle))
    binary = a.output / "probe"
    command = ["/usr/local/cuda/bin/nvcc", "-O3", "--use_fast_math", "-arch=sm_86", str(generated), "-o", str(binary)]
    subprocess.run(command, check=True)
    with (a.output / "measurements.json").open("w") as log:
        subprocess.run([str(binary)], stdout=log, check=True)
    write_json(a.output / "identity.json", {"source_sha256": digest(path), "generated_sha256": digest(generated),
        "binary_sha256": digest(binary), "command": command, "kind": "primitive_only; no end-to-end promotion"})


if __name__ == "__main__":
    main()

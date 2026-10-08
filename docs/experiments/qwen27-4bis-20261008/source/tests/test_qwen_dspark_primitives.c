/* Independent float64 oracles for the DSpark-specific equations. No target
 * checkpoint is needed. Attention reads real distant 64K cells from RAM. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); exit(1); } } while (0)
static uint16_t half(float x) { _Float16 h = x; uint16_t v; memcpy(&v,&h,2); return v; }
static double unhalf(uint16_t v) { _Float16 h; memcpy(&h,&v,2); return (double)h; }

static double norm_oracle(const float *x, const float *y, const float *w,
        unsigned heads, unsigned rows, unsigned position) {
    const double low=floor(128*log(8192.0/(32*2*M_PI))/(2*log(1e7)));
    const double high=ceil(128*log(8192.0/(2*M_PI))/(2*log(1e7)));
    double worst=0;
    for (unsigned r=0;r<rows;r++) for (unsigned h=0;h<heads;h++) {
        const float *p=x+((uint64_t)r*heads+h)*128;
        const float *q=y+((uint64_t)r*heads+h)*128;
        double sum=0; for (unsigned d=0;d<128;d++) sum+=(double)p[d]*p[d];
        const double inv=1/sqrt(sum/128+1e-6);
        for (unsigned d=0;d<64;d++) {
            const double ramp=fmin(1,fmax(0,(d-low)/(high-low)));
            const double freq=((1-ramp)+ramp/32)/pow(1e7,2.0*d/128);
            const double angle=(position+r)*freq, scale=1+.1*log(32.0);
            const double a=p[d]*w[d]*inv,b=p[d+64]*w[d+64]*inv;
            worst=fmax(worst,fabs(scale*(a*cos(angle)-b*sin(angle))-q[d]));
            worst=fmax(worst,fabs(scale*(b*cos(angle)+a*sin(angle))-q[d+64]));
        }
    }
    return worst;
}

static double attention_oracle(const float *q,const float *nk,const float *nv,
        const uint16_t *k,const uint16_t *v,const float *out,unsigned context,unsigned rows) {
    double worst=0;
    /* Four full-context independent queries, covering different GQA groups,
     * the first/last query and every output dimension. */
    for (unsigned ri=0;ri<2;ri++) for (unsigned hi=0;hi<2;hi++) {
        const unsigned row=ri?rows-1:0, head=hi?31:0, kvh=head/4;
        const float *query=q+((uint64_t)row*32+head)*128;
        double *scores=malloc((context+rows)*sizeof(double)),maximum=-INFINITY;
        CHECK(scores);
        for (unsigned cell=0;cell<context+rows;cell++) {
            const uint64_t off=((uint64_t)(cell<context?cell:cell-context)*8+kvh)*128;
            double score=0;
            for (unsigned d=0;d<128;d++)
                score+=query[d]*(cell<context?unhalf(k[off+d]):nk[off+d]);
            scores[cell]=score/sqrt(128.0); maximum=fmax(maximum,scores[cell]);
        }
        double sum=0,accum[128]={0};
        for (unsigned cell=0;cell<context+rows;cell++) {
            const double weight=exp(scores[cell]-maximum);
            const uint64_t off=((uint64_t)(cell<context?cell:cell-context)*8+kvh)*128;
            sum+=weight;
            for (unsigned d=0;d<128;d++)
                accum[d]+=weight*(cell<context?unhalf(v[off+d]):nv[off+d]);
        }
        for (unsigned d=0;d<128;d++)
            worst=fmax(worst,fabs(accum[d]/sum-out[((uint64_t)row*32+head)*128+d]));
        free(scores);
    }
    return worst;
}

int main(void) {
    enum {MAX_ROWS=15,CTX=65536};
    float weights[1024]={0},embedded[256],features[3*25600];
    for (unsigned i=0;i<128;i++) weights[i]=.8f+(i%17)*.01f;
    for (unsigned i=128;i<1024;i++) weights[i]=(float)((int)(i%101)-50)/32;
    uint16_t *hw=(uint16_t *)(weights+512);
    for (unsigned i=0;i<512;i++) hw[i]=half((float)((int)(i%101)-50)/32);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(weights,sizeof(weights)));
    ds4_gpu_tensor *single=ds4_gpu_tensor_alloc(256*4);
    CHECK(single && ds4_gpu_embed_token_quant_tensor(single,weights,sizeof(weights),128*4,0,3,2,128));
    CHECK(ds4_gpu_tensor_read(single,0,embedded,128*4));
    CHECK(!memcmp(embedded,weights+128+2*128,128*4));
    CHECK(ds4_gpu_embed_token_quant_tensor(single,weights,sizeof(weights),512*4,1,4,3,128));
    CHECK(ds4_gpu_tensor_read(single,0,embedded,128*4));
    for (unsigned i=0;i<128;i++) CHECK(embedded[i]==unhalf(hw[3*128+i]));
    CHECK(!ds4_gpu_embed_token_quant_tensor(single,weights,sizeof(weights),512*4,1,4,4,128));
    CHECK(!ds4_gpu_embed_token_quant_tensor(single,weights,sizeof(weights),sizeof(weights)-16,1,4,0,128));
    ds4_gpu_tensor_free(single);

    float source[3*5120]; for (unsigned i=0;i<3*5120;i++) source[i]=(float)i;
    ds4_gpu_tensor *src=ds4_gpu_tensor_alloc(sizeof(source)),*dst=ds4_gpu_tensor_alloc(sizeof(features));
    CHECK(src&&dst&&ds4_gpu_tensor_write(src,0,source,sizeof(source)));
    for (unsigned slot=0;slot<5;slot++) CHECK(ds4_gpu_qwen_dspark_features_tensor(dst,src,3,slot));
    CHECK(ds4_gpu_tensor_read(dst,0,features,sizeof(features)));
    for (unsigned row=0;row<3;row++) for (unsigned slot=0;slot<5;slot++)
        CHECK(!memcmp(features+row*25600+slot*5120,source+row*5120,5120*4));
    CHECK(!ds4_gpu_qwen_dspark_features_tensor(dst,src,3,5));
    ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst);

    float *q=malloc(MAX_ROWS*4096*4),*actual=malloc(MAX_ROWS*4096*4);
    float *nk=malloc(MAX_ROWS*1024*4),*nv=malloc(MAX_ROWS*1024*4);
    uint16_t *k=malloc((uint64_t)CTX*1024*2),*v=malloc((uint64_t)CTX*1024*2);
    CHECK(q&&actual&&nk&&nv&&k&&v);
    for (unsigned i=0;i<MAX_ROWS*4096;i++) q[i]=2*sinf(i*.013f);
    for (unsigned i=0;i<MAX_ROWS*1024;i++) { nk[i]=cosf(i*.017f);nv[i]=sinf(i*.021f)+.7f; }
    for (uint64_t i=0;i<(uint64_t)CTX*1024;i++) {
        k[i]=half((float)((int)((i*17)%127)-63)/64);
        v[i]=half((float)((int)((i*31)%251)-125)/128);
    }
    ds4_gpu_tensor *qt=ds4_gpu_tensor_alloc(MAX_ROWS*4096*4),*kt=ds4_gpu_tensor_alloc(MAX_ROWS*1024*4);
    ds4_gpu_tensor *vt=ds4_gpu_tensor_alloc(MAX_ROWS*1024*4),*out=ds4_gpu_tensor_alloc(MAX_ROWS*4096*4);
    CHECK(qt&&kt&&vt&&out);
    const unsigned positions[]={0,4096,65520};
    for (unsigned pi=0;pi<3;pi++) for (unsigned heads=8;heads<=32;heads+=24) {
        CHECK(ds4_gpu_tensor_write(qt,0,q,MAX_ROWS*4096*4));
        CHECK(ds4_gpu_qwen_dspark_norm_rope_tensor(qt,weights,sizeof(weights),0,heads,MAX_ROWS,positions[pi]));
        CHECK(ds4_gpu_tensor_read(qt,0,actual,MAX_ROWS*4096*4));
        double error=norm_oracle(q,actual,weights,heads,MAX_ROWS,positions[pi]);
        /* Phase uses float64; the same float32 normalization tolerance now
         * applies even at distant positions. */
        CHECK(error<3e-5);
        printf("PASS learned head norm/YaRN heads=%u pos=%u max_abs=%.9g\n",heads,positions[pi],error);
    }
    CHECK(ds4_gpu_tensor_write(qt,0,q,MAX_ROWS*4096*4));
    CHECK(ds4_gpu_tensor_write(kt,0,nk,MAX_ROWS*1024*4)&&ds4_gpu_tensor_write(vt,0,nv,MAX_ROWS*1024*4));
    void *workspace=ds4_gpu_qwen_dspark_workspace_alloc(CTX,MAX_ROWS); CHECK(workspace);
    const unsigned contexts[]={0,1,1023,8191,8192,8193,CTX};
    for (unsigned ci=0;ci<7;ci++) for (unsigned wi=0;wi<2;wi++) {
        const unsigned rows=wi?15:7,context=contexts[ci];
        CHECK(ds4_gpu_qwen_dspark_attention_tensor(out,qt,kt,vt,k,v,context,rows,workspace));
        CHECK(ds4_gpu_tensor_read(out,0,actual,(uint64_t)rows*4096*4));
        for (unsigned i=0;i<rows*4096;i++) CHECK(isfinite(actual[i]));
        const double error=attention_oracle(q,nk,nv,k,v,actual,context,rows);
        CHECK(error<3e-5);
        printf("PASS GQA whole context=%u bidirectional rows=%u max_abs=%.9g\n",context,rows,error);
    }
    CHECK(!ds4_gpu_qwen_dspark_attention_tensor(out,qt,kt,vt,k,v,CTX+1,7,workspace));
    ds4_gpu_qwen_dspark_workspace_free(workspace);
    ds4_gpu_tensor_free(qt);ds4_gpu_tensor_free(kt);ds4_gpu_tensor_free(vt);ds4_gpu_tensor_free(out);
    free(q);free(actual);free(nk);free(nv);free(k);free(v);ds4_gpu_cleanup();
    puts("PASS Qwen DSpark independent primitive qualification"); return 0;
}

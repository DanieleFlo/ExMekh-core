/* Real DSpark/target integration. Force every verifier rollback boundary,
 * check the entire target payload against scalar evaluation, then exercise
 * checkpoint identity/truncation, cancellation/RNG and session recreation. */
#include "../ds4.c"
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#c,err); exit(1); } } while (0)

static void save(ds4_session *s, ds4_session_snapshot *snap, char *err) {
    FILE *fp=tmpfile(); CHECK(fp);
    CHECK(ds4_session_save_payload(s,fp,err,256)==0);
    snap->len=ds4_session_payload_bytes(s);snap->ptr=malloc(snap->len);CHECK(snap->ptr);
    CHECK(!fseek(fp,0,SEEK_SET)&&fread(snap->ptr,1,snap->len,fp)==snap->len);fclose(fp);
}
static int load(ds4_session *s,const ds4_session_snapshot *snap,char *err) {
    FILE *fp=fmemopen(snap->ptr,snap->len,"rb");CHECK(fp);
    int rc=ds4_session_load_payload(s,fp,snap->len,err,256);fclose(fp);return rc;
}
static void target_save(ds4_session *s,ds4_session_snapshot *snap,char *err) {
    ds4_qwen_gpu_graph *g=&s->qwen_graph;ds4_qwen_dspark_graph *d=g->dspark;
    const bool mtp=g->mtp_ready;g->dspark=NULL;g->mtp_ready=false;
    save(s,snap,err);g->dspark=d;g->mtp_ready=mtp;
}
typedef struct { int drafts[15],keep,abort_row,calls; } sampler;
static int controlled(void *ud,const float *logits,int vocab,const int *prefix,int len,uint64_t *rng) {
    (void)logits;(void)prefix;sampler *p=ud;p->calls++;*rng+=123;
    if (p->abort_row==len) return -1;
    if (len<=p->keep) return p->drafts[len-1];
    return len<=15?(p->drafts[len-1]+1)%vocab:sample_argmax(logits,vocab);
}

int main(int argc,char **argv) {
    if (argc!=3) return 2;
    char err[256]={0};ds4_engine *e=NULL;ds4_session *s=NULL;
    ds4_engine_options opt={.model_path=argv[1],.mtp_path=argv[2],
        .backend=DS4_BACKEND_CUDA,.context_size=512,.prefill_chunk=64,.mtp_draft_tokens=15};
    CHECK(ds4_engine_open(&e,&opt)==0&&e->support_kind==DS4_SUPPORT_QWEN_DSPARK);
    CHECK(ds4_session_create(&s,e,512)==0);
    CHECK(s->qwen_graph.dspark&&s->qwen_graph.mtp_ready);
    ds4_tokens pattern={0},prompt={0};
    ds4_tokenize_text(e,"Write a C function that clamps an integer and explain boundary values.\n",&pattern);
    CHECK(pattern.len>0);
    for (unsigned i=0;i<64;i++) ds4_tokens_push(&prompt,pattern.v[i%pattern.len]);
    CHECK(ds4_session_sync(s,&prompt,err,sizeof(err))==0);
    CHECK(s->qwen_graph.dspark->frontier==64&&s->qwen_graph.mtp_cache_len==64);
    ds4_session_snapshot initial={0};save(s,&initial,err);
    for (int keep=0;keep<=15;keep++) {
        CHECK(load(s,&initial,err)==0);
        sampler p={.keep=keep};int drafted=0,first=ds4_session_argmax(s);
        CHECK(qwen_dspark_propose(&s->qwen_graph,&e->model,&e->weights,first,64,15,0,p.drafts,&drafted));
        CHECK(drafted==15);
        int accepted[16],next=-1;uint64_t rng=42;
        const int n=ds4_session_eval_speculative_constrained_sample(s,first,0,0,1,0,&rng,16,-1,
            accepted,16,&next,controlled,&p,err,sizeof(err));
        CHECK(n==keep+1&&accepted[0]==first&&s->qwen_graph.mtp_ready);
        CHECK(s->qwen_graph.dspark->frontier==64u+n&&s->qwen_graph.mtp_cache_len==64u+n);
        CHECK(s->qwen_mtp_fallbacks==0&&s->qwen_mtp_cycles>0);
        ds4_session_snapshot spec={0},serial={0};target_save(s,&spec,err);
        CHECK(load(s,&initial,err)==0);
        for (int i=0;i<n;i++) CHECK(ds4_session_eval(s,accepted[i],err,sizeof(err))==0);
        target_save(s,&serial,err);
        CHECK(spec.len==serial.len&&!memcmp(spec.ptr,serial.ptr,spec.len));
        ds4_session_snapshot_free(&spec);ds4_session_snapshot_free(&serial);
        fprintf(stderr,"PASS DSpark verifier accepted=%d: full target payload scalar bit-exact\n",n);
    }
    /* A declined constrained row discards every tentative sample, including
     * its RNG consumption. Target and drafter stop at the seed frontier. */
    const int abort_rows[]={1,4,16};
    for (unsigned i=0;i<3;i++) {
        CHECK(load(s,&initial,err)==0);
        sampler p={.keep=15,.abort_row=abort_rows[i]};int drafted=0,first=ds4_session_argmax(s);
        CHECK(qwen_dspark_propose(&s->qwen_graph,&e->model,&e->weights,first,64,15,0,p.drafts,&drafted));
        uint64_t rng=42;int accepted[16],next=-1;
        CHECK(ds4_session_eval_speculative_constrained_sample(s,first,0,0,1,0,&rng,16,-1,
            accepted,16,&next,controlled,&p,err,sizeof(err))==1);
        CHECK(rng==42&&next==-1&&s->qwen_graph.dspark->frontier==65);
        fprintf(stderr,"PASS DSpark cancellation row=%d and RNG rollback\n",abort_rows[i]);
    }
    CHECK(load(s,&initial,err)==0);
    /* A deliberately certain cutoff exercises a zero-length proposal block
     * without marking inference failed or reading a nonexistent bonus row. */
    e->dspark_confidence_threshold_set=true;e->dspark_confidence_threshold=1;
    int confidence_tokens[16];
    CHECK(ds4_session_eval_speculative_argmax(s,ds4_session_argmax(s),16,-1,
        confidence_tokens,16,err,sizeof(err))==1);
    CHECK(s->qwen_graph.mtp_ready&&s->qwen_graph.dspark->frontier==65&&s->qwen_mtp_fallbacks==0);
    e->dspark_confidence_threshold_set=false;
    CHECK(load(s,&initial,err)==0);
    ds4_session_snapshot roundtrip={0};save(s,&roundtrip,err);
    CHECK(initial.len==roundtrip.len&&!memcmp(initial.ptr,roundtrip.ptr,initial.len));
    ds4_session_snapshot_free(&roundtrip);
    FILE *compact=tmpfile();uint64_t written=0;CHECK(compact);
    CHECK(ds4_session_save_skill_state(s,compact,&written,NULL,err,sizeof(err))==0);
    ds4_session_device_frontier *device=NULL;
    CHECK(ds4_session_capture_device_frontier(s,&device,err,sizeof(err))==0);
    int child=ds4_session_argmax(s);CHECK(ds4_session_eval(s,child,err,sizeof(err))==0);
    CHECK(!fseek(compact,0,SEEK_SET));
    CHECK(ds4_session_load_skill_state(s,compact,&prompt,NULL,err,sizeof(err))==0);
    CHECK(s->qwen_graph.dspark->frontier==64&&s->qwen_graph.mtp_ready);
    save(s,&roundtrip,err);CHECK(initial.len==roundtrip.len&&!memcmp(initial.ptr,roundtrip.ptr,initial.len));
    ds4_session_snapshot_free(&roundtrip);
    CHECK(ds4_session_eval(s,child,err,sizeof(err))==0);
    CHECK(ds4_session_restore_device_frontier(s,device,&prompt,err,sizeof(err))==0);
    save(s,&roundtrip,err);CHECK(initial.len==roundtrip.len&&!memcmp(initial.ptr,roundtrip.ptr,initial.len));
    ds4_session_snapshot_free(&roundtrip);ds4_session_device_frontier_free(device);fclose(compact);
    fprintf(stderr,"PASS DSpark compact and device frontiers preserve full logical payload\n");
    const uint64_t fingerprint=e->qwen_dspark_weights.fingerprint;
    e->qwen_dspark_weights.fingerprint^=1;
    CHECK(load(s,&initial,err)!=0);e->qwen_dspark_weights.fingerprint=fingerprint;
    CHECK(load(s,&initial,err)==0);
    ds4_session_snapshot truncated=initial;truncated.len--;
    CHECK(load(s,&truncated,err)!=0);CHECK(load(s,&initial,err)==0);
    /* Second session owns its own draft frontiers and RAM arrays. */
    ds4_session *other=NULL;CHECK(ds4_session_create(&other,e,512)==0);
    CHECK(other->qwen_graph.dspark->keys[0]!=s->qwen_graph.dspark->keys[0]);
    CHECK(load(other,&initial,err)==0);
    int token=ds4_session_argmax(other);CHECK(ds4_session_eval(other,token,err,sizeof(err))==0);
    CHECK(other->qwen_graph.dspark->frontier==65&&s->qwen_graph.dspark->frontier==64);
    ds4_session_free(other);
    ds4_session_invalidate(s);CHECK(s->qwen_graph.dspark->frontier==0);
    CHECK(ds4_session_sync(s,&prompt,err,sizeof(err))==0);
    CHECK(s->qwen_graph.dspark->frontier==64&&s->qwen_graph.mtp_ready);
    ds4_session_free(s);s=NULL;CHECK(ds4_session_create(&s,e,512)==0);
    CHECK(load(s,&initial,err)==0);save(s,&roundtrip,err);
    CHECK(initial.len==roundtrip.len&&!memcmp(initial.ptr,roundtrip.ptr,initial.len));
    ds4_session_snapshot_free(&roundtrip);ds4_session_snapshot_free(&initial);
    ds4_tokens_free(&pattern);ds4_tokens_free(&prompt);ds4_session_free(s);ds4_engine_close(e);
    fprintf(stderr,"PASS DSpark checkpoint, identity, truncation, session isolation and recreation\n");return 0;
}

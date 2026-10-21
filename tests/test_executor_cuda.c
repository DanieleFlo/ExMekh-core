#include "ds4.h"
#include "ds4_gpu.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct { ds4_gpu_tensor *tensor; ds4_session *session; unsigned executed, committed; } component;
static ds4_status load(void *opaque) {
    component *c=opaque;
    if (!c->tensor) c->tensor=ds4_gpu_tensor_alloc(16u*1024u*1024u);
    return c->tensor ? DS4_STATUS_OK : DS4_STATUS_NOMEM;
}
static ds4_status release(void *opaque) {
    component *c=opaque; ds4_gpu_tensor_free(c->tensor); c->tensor=NULL; return DS4_STATUS_OK;
}
static ds4_status execute(void *opaque,uint32_t step) {
    component *c=opaque; assert(step==1 && c->tensor);
    c->executed++;
    return ds4_gpu_tensor_fill_f32(c->tensor,17.25f,16u*1024u*1024u/4u) ? DS4_STATUS_OK : DS4_STATUS_INFERENCE;
}
static ds4_status drain(void *opaque) { (void)opaque; return ds4_gpu_quiesce_status(); }
static ds4_status commit(void *opaque) {
    component *c=opaque; ds4_session_memory memory;
    assert(!ds4_session_memory_query(c->session,&memory) && !memory.parked && memory.weights_resident);
    c->committed++; return DS4_STATUS_OK;
}
static void abort_result(void *opaque) { (void)opaque; assert(0); }
int main(int argc,char **argv) {
    assert(argc==2);
    ds4_engine_options options={.model_path=argv[1],.backend=DS4_BACKEND_CUDA,.context_size=256,.prefill_chunk=32};
    ds4_engine *engine=NULL; ds4_session *session=NULL;
    assert(!ds4_engine_open(&engine,&options) && !ds4_session_create(&session,engine,256));
    ds4_tokens tokens={0}; char error[256]={0};
    ds4_tokenize_text(engine,"Explain the colors of a red square and blue circle.",&tokens);
    assert(!ds4_session_sync(session,&tokens,error,sizeof(error)));
    int vocabulary=ds4_engine_vocab_size(engine), token=ds4_session_argmax(session);
    float *expected=malloc((size_t)vocabulary*sizeof(float)), *actual=malloc((size_t)vocabulary*sizeof(float));
    assert(expected && actual);
    ds4_session_snapshot snapshot={0};
    assert(!ds4_session_save_snapshot(session,&snapshot,error,sizeof(error)));
    assert(!ds4_session_eval(session,token,error,sizeof(error)) && ds4_session_copy_logits(session,expected,vocabulary)==vocabulary);
    ds4_resources *resources=ds4_engine_resources(engine);
    ds4_resource_budget original; assert(!ds4_resources_get_budget(resources,DS4_MEMORY_DEVICE,&original));
    for (unsigned mode=0;mode<3;mode++) {
        assert(!ds4_session_load_snapshot(session,&snapshot,error,sizeof(error)));
        ds4_engine_reclaim *adapter=NULL;
        assert(!ds4_engine_reclaim_create(&adapter,engine,"/tmp"));
        size_t count=0;
        const ds4_execution_victim *source=ds4_engine_reclaim_victims(adapter,&count);
        assert(count==2);
        ds4_execution_victim victims[2]; memcpy(victims,source,sizeof(victims));
        if (mode==2) { victims[0].cost=100; victims[1].cost=1; }
        ds4_resource_totals totals; ds4_resources_totals(resources,0,&totals);
        ds4_resource_budget budget=original;
        if (mode) { budget.limit=totals.allocated[DS4_MEMORY_DEVICE]+4u*1024u*1024u; budget.global_known=false; }
        assert(!ds4_resources_set_budget(resources,DS4_MEMORY_DEVICE,&budget));
        component c={.session=session};
        ds4_execution_group group={.memory.bytes={[DS4_MEMORY_DEVICE]=16u*1024u*1024u},.context=&c,.load=load,.release=release};
        ds4_execution_step step = {.id = 1, .groups = 1};
        ds4_execution_variant variant = {.steps = &step, .step_count = 1};
        ds4_execution_request request={.groups=&group,.group_count=1,.variants=&variant,.variant_count=1,.victims=victims,.victim_count=count};
        ds4_execution_ops ops={.context=&c,.execute=execute,.drain=drain,.commit=commit,.abort=abort_result};
        ds4_execution *job=NULL;
        assert(!ds4_execution_create(&job,ds4_engine_runtime(engine),resources,900,&request,&ops));
        ds4_status status=ds4_execution_run(job);
        if (status) fprintf(stderr,"executor CUDA mode=%u status=%u state=%u\n",mode,status,ds4_execution_get_state(job));
        assert(!status && c.executed==1 && c.committed==1);
        const ds4_execution_plan *plan=ds4_execution_get_plan(job);
        assert(plan->victims==(mode==0 ? 0u : mode==1 ? 1u : 2u));
        assert(!ds4_execution_close(job) && !ds4_engine_reclaim_close(adapter));
        assert(!ds4_resources_set_budget(resources,DS4_MEMORY_DEVICE,&original));
        assert(!ds4_session_eval(session,token,error,sizeof(error)) && ds4_session_copy_logits(session,actual,vocabulary)==vocabulary);
        assert(!memcmp(expected,actual,(size_t)vocabulary*sizeof(float)));
        printf("executor CUDA mode=%u continuation bit-exact: PASS\n",mode); fflush(stdout);
    }
    ds4_session_snapshot_free(&snapshot); free(expected); free(actual); ds4_tokens_free(&tokens);
    ds4_session_free(session); ds4_engine_close(engine);
    ds4_resource_totals totals; ds4_resources_totals(resources,0,&totals);
    assert(!totals.roots && !totals.views);
    for (unsigned i=0;i<DS4_MEMORY_COUNT;i++) assert(!totals.allocated[i] && !totals.reserved[i]);
    return 0;
}

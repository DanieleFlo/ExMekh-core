#include "ds4_bundle.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ds4_resources *resources;
    ds4_resource *allocation;
    bool fail_write, fail_fence, fail_release;
    const char *mutate_file;
} backend;
static ds4_status allocate(void *opaque, uint64_t bytes, void **out) {
    backend *b=opaque;
    ds4_resource_spec spec={.owner=42,.requested=bytes,.allocated=bytes,.live=bytes,.location=DS4_MEMORY_DEVICE};
    ds4_status status=ds4_resource_register(b->resources,&spec,&b->allocation);
    if (status) return status;
    *out=malloc((size_t)bytes);
    if (!*out) { ds4_resource_release(b->allocation); b->allocation=NULL; return DS4_STATUS_NOMEM; }
    return DS4_STATUS_OK;
}
static ds4_status write_tensor(void *opaque, void *memory, uint64_t offset, const void *source, uint64_t bytes) {
    backend *b=opaque;
    if (b->fail_write) return DS4_STATUS_INFERENCE;
    if (b->mutate_file) {
        FILE *file = fopen(b->mutate_file, "ab"); assert(file);
        assert(fputs("changed", file) >= 0 && !fclose(file));
        b->mutate_file = NULL;
    }
    memcpy((uint8_t *)memory+offset,source,(size_t)bytes); return DS4_STATUS_OK;
}
static ds4_status release(void *opaque, void *memory) {
    backend *b=opaque;
    if (b->fail_release) return DS4_STATUS_INFERENCE;
    ds4_status status=ds4_resource_release(b->allocation);
    if (!status) { b->allocation=NULL; free(memory); }
    return status;
}
static ds4_status fence(void *opaque) {
    return ((backend *)opaque)->fail_fence ? DS4_STATUS_INFERENCE : DS4_STATUS_OK;
}
int main(int argc,char **argv) {
    assert(argc==4);
    ds4_resources *r=NULL; assert(!ds4_resources_create(&r));
    bool host_deny = !strcmp(argv[3], "host-deny");
    if (host_deny) {
        ds4_resource_budget budget = {.limit=1};
        assert(!ds4_resources_set_budget(r, DS4_MEMORY_HOST, &budget));
    }
    ds4_bundle *b=NULL;
    ds4_status status=ds4_bundle_open(&b,argv[1],argv[2],r,19);
    if (host_deny) { assert(status == DS4_STATUS_NOMEM && !b); goto end; }
    if (!strcmp(argv[3],"reject")) { assert(status && !b); goto end; }
    assert(!status && b);
    assert(ds4_bundle_model(b) && ds4_bundle_revision(b) && ds4_bundle_profile(b));
    ds4_resource_totals accounted; ds4_resources_totals(r,19,&accounted);
    assert(accounted.allocated[DS4_MEMORY_HOST] && accounted.allocated[DS4_MEMORY_MMAP]);
    assert(!ds4_bundle_dump(b,stdout));
    if (!strcmp(argv[3],"dump")) { assert(!ds4_bundle_close(b)); goto end; }
    {
        ds4_bundle_hint *hint = NULL;
        assert(ds4_bundle_hint_start(&hint, b, 0, (16u << 20) + 1) == DS4_STATUS_INVALID && !hint);
        assert(!ds4_bundle_hint_start(&hint, b, 0, 4096));
        assert(ds4_bundle_close(b) == DS4_STATUS_BUSY);
        assert(!ds4_bundle_hint_close(hint));
    }
    for (size_t component=0;component<ds4_bundle_component_count(b);component++) {
        size_t count=0;
        ds4_weight_tensor *expected=calloc(ds4_bundle_tensor_count(b),sizeof(*expected)); assert(expected);
        for (size_t i=0;i<ds4_bundle_tensor_count(b);i++) {
            const ds4_weight_tensor *t=ds4_bundle_tensor(b,i);
            if (t->component==component) expected[count++]=*t;
        }
        assert(!ds4_bundle_validate_component(b,component,expected,count));
        expected[0].type^=1;
        assert(ds4_bundle_validate_component(b,component,expected,count)==DS4_STATUS_INVALID);
        expected[0].type^=1;
        assert(ds4_bundle_validate_component(b,component,expected,count-1)==DS4_STATUS_INVALID);
        if (expected[0].rank) {
            expected[0].shape[0]++;
            assert(ds4_bundle_validate_component(b,component,expected,count)==DS4_STATUS_INVALID);
        }
        free(expected);
    }
    {
        /* Execution groups can split storage groups; tensor zero must not be
         * accidentally selected just because its packed offset is zero. */
        const ds4_weight_tensor *t=ds4_bundle_tensor(b,0);
        size_t ids[]={0,0};
        backend stub={.resources=r};
        ds4_weight_backend ops={&stub,allocate,write_tensor,fence,release};
        ds4_weight_group *g=NULL;
        assert(ds4_weight_group_create_selected(&g,b,t->component,ids,2,&ops)==DS4_STATUS_INVALID);
        g=(ds4_weight_group *)&stub;
        assert(ds4_weight_group_create_selected(&g,b,t->component,NULL,0,&ops)==DS4_STATUS_INVALID);
        assert(!g);
        ids[0]=ds4_bundle_tensor_count(b);
        assert(ds4_weight_group_create_selected(&g,b,t->component,ids,1,&ops)==DS4_STATUS_INVALID);
        ids[0]=0;
        assert(!ds4_weight_group_create_selected(&g,b,t->component,ids,1,&ops));
        assert(!ds4_weight_group_load(g));
        const void *storage=NULL; uint64_t offset=0;
        assert(!ds4_weight_group_tensor(g,t->name,&storage,&offset));
        assert(!memcmp((const uint8_t*)storage+offset,ds4_bundle_tensor_data(b,0),t->bytes));
        for(size_t i=1;i<ds4_bundle_tensor_count(b);i++)
            assert(ds4_weight_group_tensor(g,ds4_bundle_tensor(b,i)->name,&storage,&offset)==DS4_STATUS_INVALID);
        assert(!ds4_weight_group_close(g));
    }
    for (size_t i=0;i<ds4_bundle_tensor_count(b);i++) {
        const ds4_weight_tensor *t=ds4_bundle_tensor(b,i);
        bool seen=false;
        for (size_t j=0;j<i;j++) {
            const ds4_weight_tensor *p=ds4_bundle_tensor(b,j);
            if (t->component==p->component && t->group==p->group) seen=true;
        }
        if (seen) continue;
        backend stub={.resources=r,.fail_write=true};
        ds4_weight_backend ops={&stub,allocate,write_tensor,fence,release};
        ds4_weight_group *g=NULL;
        ds4_resource_totals before; ds4_resources_totals(r,19,&before);
        assert(!ds4_weight_group_create(&g,b,t->component,t->group,&ops));
        ds4_resources_totals(r,19,&accounted);
        assert(accounted.allocated[DS4_MEMORY_HOST] > before.allocated[DS4_MEMORY_HOST]);
        assert(ds4_bundle_close(b)==DS4_STATUS_BUSY);
        assert(ds4_weight_group_load(g)==DS4_STATUS_INFERENCE && !ds4_weight_group_resident(g));
        assert(!ds4_weight_group_release(g)); stub.fail_write=false;
        if (!strcmp(argv[3], "mutate")) {
            char mutation_path[4096];
            assert(snprintf(mutation_path, sizeof(mutation_path), "%s/config.json", argv[2]) < (int)sizeof(mutation_path));
            stub.mutate_file = mutation_path;
            assert(ds4_weight_group_load(g) == DS4_STATUS_INVALID && !ds4_weight_group_resident(g));
            assert(!ds4_weight_group_close(g) && !ds4_bundle_close(b));
            goto end;
        }
        for (unsigned repeat=0;repeat<2;repeat++) {
            assert(!ds4_weight_group_load(g) && ds4_weight_group_resident(g));
            for (size_t j=0;j<ds4_bundle_tensor_count(b);j++) {
                const ds4_weight_tensor *p=ds4_bundle_tensor(b,j);
                if (t->component!=p->component || t->group!=p->group) continue;
                const void *storage=NULL; uint64_t offset=0;
                assert(!ds4_weight_group_tensor(g,p->name,&storage,&offset));
                assert(!memcmp((const uint8_t *)storage+offset,ds4_bundle_tensor_data(b,j),(size_t)p->bytes));
            }
            stub.fail_fence = true;
            assert(ds4_weight_group_release(g) == DS4_STATUS_INFERENCE && !ds4_weight_group_resident(g));
            assert(ds4_weight_group_load(g) == DS4_STATUS_NOT_READY);
            assert(ds4_weight_group_close(g) == DS4_STATUS_INFERENCE);
            stub.fail_fence = false; stub.fail_release = true;
            assert(ds4_weight_group_release(g) == DS4_STATUS_INFERENCE && !ds4_weight_group_resident(g));
            stub.fail_release = false;
            assert(!ds4_weight_group_release(g) && !ds4_weight_group_release(g));
        }
        assert(!ds4_weight_group_close(g));
        ds4_resources_totals(r,19,&accounted);
        assert(accounted.allocated[DS4_MEMORY_HOST] == before.allocated[DS4_MEMORY_HOST]);
    }
    assert(!ds4_bundle_close(b));
end:
    ds4_resource_totals totals; ds4_resources_totals(r,0,&totals);
    assert(!totals.roots && !totals.views);
    for (unsigned i=0;i<DS4_MEMORY_COUNT;i++) assert(!totals.allocated[i] && !totals.reserved[i]);
    assert(!ds4_resources_close(r)); return 0;
}

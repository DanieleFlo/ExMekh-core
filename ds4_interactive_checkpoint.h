#ifndef DS4_INTERACTIVE_CHECKPOINT_H
#define DS4_INTERACTIVE_CHECKPOINT_H

/* Owns the live rollback, portable pause/rollback payloads and their budget.
 * The protocol lifecycle asks for capture, staging, publication or release;
 * it does not interpret resident graph state or checkpoint-list storage. */
typedef struct {
    server *owner;
    server_slot *slot;
    interactive_entry *entry;
    char *error;
    size_t error_capacity;
} interactive_checkpoint_operation;

static bool interactive_checkpoint_reserve_locked(server *s, int slot_id,
        const char *context_id, uint64_t bytes, bool *pending) {
    interactive_entry *owner = NULL;
    uint64_t used = 0;
    for (interactive_entry *entry = s->interactive_entries; entry; entry = entry->next) {
        used += entry->checkpoint_bytes + entry->rollback.reserved_bytes;
        if (entry->slot_id == slot_id && !strcmp(entry->context_id, context_id) && interactive_active(entry))
            owner = entry;
    }
    *pending = owner != NULL;
    if (!owner) return true;
    if (bytes > INTERACTIVE_CHECKPOINT_BUDGET ||
        used - owner->checkpoint_bytes > INTERACTIVE_CHECKPOINT_BUDGET - bytes) return false;
    owner->checkpoint_bytes = bytes;
    return true;
}

static void interactive_drop_checkpoint_locked(server *s, interactive_entry *e) {
    e->checkpoint_bytes = 0;
    if (e->slot_id < 0 || e->slot_id >= s->slot_count) return;
    server_slot *slot = &s->slots[e->slot_id];
    agent_context_checkpoint *cp = agent_context_checkpoint_find_locked(slot, e->context_id);
    if (cp) agent_context_checkpoint_remove_locked(slot, cp, true);
}

static void interactive_rollback_clear(interactive_rollback *rollback) {
    resident_kv_frontier_clear(&rollback->resident);
    live_tool_state_free(&rollback->binding);
    agent_context_checkpoint_free(rollback->checkpoint, true);
    memset(rollback, 0, sizeof(*rollback));
}

/* Publish ownership only. The slot worker restores this payload before reuse;
 * a control/client thread never mutates a model session. */
static void interactive_publish_rollback_locked(server *s, interactive_entry *entry) {
    if (!entry->rollback.checkpoint && entry->slot_id >= 0 &&
        entry->slot_id < s->slot_count &&
        agent_context_checkpoint_restore_pending_locked(
            &s->slots[entry->slot_id], entry->context_id)) return;
    interactive_drop_checkpoint_locked(s, entry);
    if (!entry->rollback.checkpoint) return;
    server_slot *slot = &s->slots[entry->slot_id];
    agent_context_checkpoint *checkpoint = entry->rollback.checkpoint;
    checkpoint->restore_pending = true;
    checkpoint->next = slot->agent_contexts;
    slot->agent_contexts = checkpoint;
    entry->rollback.checkpoint = NULL;
    entry->rollback.reserved_bytes = 0;
}

static bool interactive_capture_rollback(const interactive_checkpoint_operation *operation,
        const request *request) {
    server *s = operation->owner;
    server_slot *slot = operation->slot;
    interactive_entry *entry = operation->entry;
    char *err = operation->error;
    size_t errlen = operation->error_capacity;
    if (!entry || request->interactive.resumed || entry->rollback.resident.valid) return true;
    slot_reuse reuse = slot_probe_reuse(s, slot, request);
    if (reuse.kind != REUSE_RESPONSES_VISIBLE && reuse.kind != REUSE_RESPONSES_TOOL_OUTPUT)
        return true;
    if (!server_resident_frontier_capture(s, slot, &entry->rollback.resident, true, err, errlen))
        return false;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_copy_locked(&entry->rollback.binding, &slot->responses_live);
    entry->rollback.reserved_bytes = resident_kv_frontier_bytes(&entry->rollback.resident);
    pthread_mutex_unlock(&s->tool_mu);
    return true;
}

static bool interactive_stage_rollback(const interactive_checkpoint_operation *operation) {
    server *s = operation->owner;
    server_slot *slot = operation->slot;
    interactive_entry *entry = operation->entry;
    char *err = operation->error;
    size_t errlen = operation->error_capacity;
    interactive_rollback *rollback = &entry->rollback;
    if (!rollback->resident.valid) return true;
    pthread_mutex_lock(&s->tool_mu);
    agent_context_checkpoint *paused = agent_context_checkpoint_take_locked(slot, entry->context_id);
    entry->checkpoint_bytes = 0;
    /* Reserve the detached pause while the rollback payload is written. */
    rollback->reserved_bytes = paused ? paused->checkpoint_bytes : 0;
    pthread_mutex_unlock(&s->tool_mu);
    if (!paused) { snprintf(err, errlen, "%s", interactive_rollback_pause_error); return false; }

    bool saved = server_resident_frontier_restore(s, slot, &rollback->resident, err, errlen);
    if (saved) {
        pthread_mutex_lock(&s->tool_mu);
        live_tool_state_copy_locked(&slot->responses_live, &rollback->binding);
        pthread_mutex_unlock(&s->tool_mu);
        saved = agent_context_checkpoint_save(s, slot, entry->context_id, err, errlen);
    }
    pthread_mutex_lock(&s->tool_mu);
    rollback->checkpoint = agent_context_checkpoint_take_locked(slot, entry->context_id);
    rollback->reserved_bytes = rollback->checkpoint ? rollback->checkpoint->checkpoint_bytes : 0;
    paused->next = slot->agent_contexts;
    slot->agent_contexts = paused;
    entry->checkpoint_bytes = paused->checkpoint_bytes;
    pthread_mutex_unlock(&s->tool_mu);

    /* Resume always keeps the paused generation. Cancellation instead publishes
     * the completed-turn payload, so no half-written invocation enters history. */
    char restore_error[INTERACTIVE_ERROR_BYTES] = {0};
    bool restored = agent_context_checkpoint_restore(s, slot, paused, NULL, restore_error, sizeof(restore_error));
    resident_kv_frontier_clear(&rollback->resident);
    live_tool_state_free(&rollback->binding);
    if (!restored) snprintf(err, errlen, "%s", restore_error);
    return saved && restored;
}

#endif

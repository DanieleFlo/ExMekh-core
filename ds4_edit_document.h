#ifndef DS4_EDIT_DOCUMENT_H
#define DS4_EDIT_DOCUMENT_H

/* Two-marker framing around literal bodies. Probes only read state; path
 * authorizations and generated IDs are committed by the runtime, never by
 * vocabulary/trie probes. */
typedef struct {
    char *path;
    char identifier[EDIT_ID_DIGITS + 1];
    edit_constraint *snapshot;
} edit_document_item;

/* The runtime only appends to the accepted transcript. Pending delimiters and
 * UTF-8 scalars stay after this validated restart; probes never advance it. */
typedef struct {
    size_t offset, index, characters, logical_bytes;
    bool old_body, new_body, separator_id, started;
} edit_document_checkpoint;

typedef struct {
    edit_document_item item[INTERACTIVE_MAX_RECORDS];
    size_t count;
    edit_constraint *snapshots[INTERACTIVE_MAX_RECORDS];
    size_t snapshot_count;
    const char *native_end;
    edit_document_checkpoint checkpoint;
    edit_selection_frontier selection;
} edit_document;

static edit_document *edit_document_new(const char *native_end) {
    edit_document *document = xmalloc(sizeof(*document));
    memset(document, 0, sizeof(*document));
    document->native_end = native_end;
    return document;
}

static void edit_document_free(edit_document *document) {
    if (!document) return;
    edit_selection_frontier_free(&document->selection);
    for (size_t i = 0; i < document->count; i++) free(document->item[i].path);
    /* Items borrow snapshots; the document owns each unique allocation. */
    for (size_t i = 0; i < document->snapshot_count; i++) {
        edit_constraint_free(document->snapshots[i]);
        free(document->snapshots[i]);
    }
    free(document);
}

static const char *edit_document_native_end(const edit_document *document) {
    return document->native_end;
}

static edit_document_checkpoint edit_document_restart(const edit_document *document,
                                                       size_t n, size_t committed) {
    if (!document || document->checkpoint.offset > n ||
        document->checkpoint.offset > committed) return (edit_document_checkpoint){0};
    return document->checkpoint;
}

/* Safe tokens cannot complete protocol/native delimiters or a pending UTF-8 scalar. */
static bool edit_document_plain_replacement(const void *opaque, size_t n) {
    const edit_document *document = opaque;
    return document && document->checkpoint.new_body && document->checkpoint.offset == n;
}

static void edit_document_commit_restart(edit_document *document,
        edit_document_checkpoint checkpoint, const buf *selected) {
    if (checkpoint.old_body) {
        if (!document->checkpoint.old_body || document->checkpoint.index != checkpoint.index)
            edit_selection_frontier_init(&document->selection,
                document->item[checkpoint.index].snapshot);
        edit_selection_frontier_advance(&document->selection,
            selected->ptr ? selected->ptr : "", selected->len);
    } else edit_selection_frontier_free(&document->selection);
    document->checkpoint = checkpoint;
}

static bool edit_document_begin_input(edit_document *document, const char *path) {
    if (!path || !path[0] || document->count >= INTERACTIVE_MAX_RECORDS ||
        (document->count && !document->item[document->count - 1].snapshot)) return false;
    document->item[document->count++].path = xstrdup(path);
    return true;
}

static edit_constraint *edit_document_shared_snapshot(const edit_document *document) {
    if (!document || !document->count) return NULL;
    const char *path = document->item[document->count - 1].path;
    for (size_t i = 0; i + 1 < document->count; i++)
        if (!strcmp(document->item[i].path, path)) return document->item[i].snapshot;
    return NULL;
}

static bool edit_document_reuses_snapshot(const edit_document *document) {
    return edit_document_shared_snapshot(document) != NULL;
}

static size_t edit_document_snapshot_memory(size_t length) {
    return edit_constraint_memory_bound(length) + sizeof(edit_document);
}

static bool edit_document_assign_identifier(const edit_document *document,
        const char *text, char identifier[EDIT_ID_DIGITS + 1]) {
    unsigned random;
    if (!random_bytes(&random, sizeof(random))) return false;
    for (unsigned offset = 0; offset < EDIT_ID_SPACE; offset++) {
        char id[EDIT_ID_DIGITS + 1], spelling[sizeof(EDIT_ID_ATTRIBUTE_FORMAT) + EDIT_ID_DIGITS];
        snprintf(id, sizeof(id), EDIT_ID_FORMAT, (random % EDIT_ID_SPACE + offset) % EDIT_ID_SPACE);
        snprintf(spelling, sizeof(spelling), EDIT_ID_ATTRIBUTE_FORMAT, id);
        bool used = strstr(text, spelling) != NULL;
        for (size_t i = 0; i + 1 < document->count; i++)
            used |= !strcmp(document->item[i].identifier, id);
        if (used) continue;
        memcpy(identifier, id, sizeof(id));
        return true;
    }
    return false;
}

/* Reject inconsistent snapshots before publishing any authorization state. */
static bool edit_document_provide(edit_document *document, const char *text) {
    if (!document || !document->count || !text) return false;
    edit_document_item *item = &document->item[document->count - 1];
    if (item->snapshot) return false;
    edit_constraint *snapshot = edit_document_shared_snapshot(document);
    if (snapshot && !edit_constraint_matches(snapshot, text)) return false;
    char identifier[EDIT_ID_DIGITS + 1];
    if (!edit_document_assign_identifier(document, text, identifier)) return false;
    if (!snapshot) {
        snapshot = xmalloc(sizeof(*snapshot));
        edit_constraint_init(snapshot, text);
        document->snapshots[document->snapshot_count++] = snapshot;
    }
    memcpy(item->identifier, identifier, sizeof(identifier));
    item->snapshot = snapshot;
    return true;
}

typedef enum {
    EDIT_DOCUMENT_STRUCTURE, EDIT_DOCUMENT_AWAIT,
    EDIT_DOCUMENT_OLD, EDIT_DOCUMENT_NEW, EDIT_DOCUMENT_DONE
} edit_document_phase;

typedef struct {
    bool valid, speculative, payload_start, native_escape;
    edit_document_phase phase;
    size_t index;
    char *path;
    edit_document_checkpoint checkpoint;
} edit_document_view;

typedef struct {
    const char *text;
    size_t length, pos, committed;
} edit_xml_cursor;

static void edit_xml_skip_space(edit_xml_cursor *cursor) {
    while (cursor->pos < cursor->length) {
        char ch = cursor->text[cursor->pos];
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
        cursor->pos++;
    }
}

/* Literal body parsing; only the two protocol markers have framing meaning. */
static bool edit_xml_unicode(const char *text, bool complete) {
    return interactive_utf8_valid((const unsigned char *)text, strlen(text), complete);
}

static const char *edit_document_selection_candidate(const edit_document *document,
        edit_document_checkpoint checkpoint, edit_document_checkpoint restart,
        const buf *selected, size_t *cursor, bool *unique) {
    const char *value = selected->ptr ? selected->ptr : "";
    if (restart.old_body && restart.index == checkpoint.index)
        return edit_selection_frontier_candidate(&document->selection, value, selected->len, cursor, unique);
    return edit_selection_candidate(document->item[checkpoint.index].snapshot,
        value, selected->len, cursor, unique);
}

static bool edit_document_native_escape_possible(const edit_document *document,
        edit_document_checkpoint checkpoint, edit_document_checkpoint restart, const buf *selected,
        const char *pending, size_t n, const char *end) {
    if (!document || !checkpoint.old_body || checkpoint.index >= document->count) return false;
    size_t cursor = 0;
    const char *suffix;
    while ((suffix = edit_document_selection_candidate(document, checkpoint, restart, selected, &cursor, NULL))) {
        buf encoded = {0};
        append_tool_body_text(&encoded, suffix, end);
        bool possible = encoded.len >= n && !memcmp(encoded.ptr, pending, n);
        buf_free(&encoded);
        if (possible) return true;
    }
    return false;
}

static void edit_document_draft_suffix(const edit_document *document,
        edit_document_checkpoint checkpoint, edit_document_checkpoint restart,
        const buf *selected, buf *out, size_t limit) {
    if (!document || !checkpoint.old_body || checkpoint.index >= document->count) return;
    size_t cursor = 0;
    bool unique = false;
    const char *suffix = edit_document_selection_candidate(document, checkpoint, restart, selected, &cursor, &unique);
    if (!suffix || !unique || !checkpoint.started) return;
    size_t n = strlen(suffix);
    if (n > limit) {
        n = limit;
        while (n && ((unsigned char)suffix[n] & 0xc0) == 0x80) n--;
    }
    buf_append(out, suffix, n);
}

static bool edit_document_plain_piece(const char *piece, size_t n) {
    return piece && n && !memchr(piece, 0, n) && !memchr(piece, '<', n) &&
        !memchr(piece, '&', n) && strlen(piece) == n && edit_xml_unicode(piece, true);
}

/* 1: consumed, 0: viable partial spelling, -1: incompatible. */
static int edit_xml_take(edit_xml_cursor *c, const char *literal) {
    size_t n = strlen(literal), left = c->length - c->pos;
    if (memcmp(c->text + c->pos, literal, left < n ? left : n)) return -1;
    if (left < n) return 0;
    c->pos += n;
    return 1;
}

static bool edit_xml_prefix(const edit_xml_cursor *c, const char *literal) {
    size_t left = c->length - c->pos, n = strlen(literal);
    return !memcmp(c->text + c->pos, literal, left < n ? left : n);
}

/* Entities are decoded only in the path attribute, never in file bodies. */
static void edit_xml_utf8(buf *out, unsigned c) {
    if (c < 0x80) buf_putc(out, (char)c);
    else if (c < 0x800) {
        buf_putc(out, (char)(0xc0 | (c >> 6)));
        buf_putc(out, (char)(0x80 | (c & 63)));
    } else if (c < 0x10000) {
        buf_putc(out, (char)(0xe0 | (c >> 12)));
        buf_putc(out, (char)(0x80 | ((c >> 6) & 63)));
        buf_putc(out, (char)(0x80 | (c & 63)));
    } else {
        buf_putc(out, (char)(0xf0 | (c >> 18)));
        buf_putc(out, (char)(0x80 | ((c >> 12) & 63)));
        buf_putc(out, (char)(0x80 | ((c >> 6) & 63)));
        buf_putc(out, (char)(0x80 | (c & 63)));
    }
}

static int edit_xml_entity(edit_xml_cursor *c, buf *out) {
    static const char *names[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    static const char values[] = {'&', '<', '>', '"', '\''};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        if (!edit_xml_prefix(c, names[i])) continue;
        int result = edit_xml_take(c, names[i]);
        if (result == 1) buf_putc(out, values[i]);
        return result;
    }
    size_t i = c->pos + 1;
    if (i == c->length) return 0;
    if (c->text[i++] != '#') return -1;
    bool hex = i < c->length && c->text[i] == 'x';
    if (hex) i++;
    size_t start = i;
    unsigned value = 0, base = hex ? 16 : 10;
    for (; i < c->length && c->text[i] != ';'; i++) {
        unsigned char ch = (unsigned char)c->text[i];
        unsigned digit = ch >= '0' && ch <= '9' ? (unsigned)(ch - '0') :
            hex && ch >= 'a' && ch <= 'f' ? (unsigned)(ch - 'a' + 10) :
            hex && ch >= 'A' && ch <= 'F' ? (unsigned)(ch - 'A' + 10) : base;
        if (digit >= base || value > (0x10ffff - digit) / base) return -1;
        value = value * base + digit;
    }
    if (i == c->length) return 0;
    if (i == start || !value || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return -1;
    edit_xml_utf8(out, value);
    c->pos = i + 1;
    return 1;
}

static int edit_xml_text_byte(edit_xml_cursor *c, buf *out) {
    unsigned char ch = (unsigned char)c->text[c->pos];
    if (ch == '&') return edit_xml_entity(c, out);
    if (!ch) return -1;
    buf_putc(out, c->text[c->pos++]);
    return 1;
}

static bool edit_document_literal(const edit_document_item *item, const buf *text,
        bool complete, bool *speculative, const edit_selection_frontier *frontier) {
    if (!item || !item->snapshot) return false;
    if (edit_constraint_empty(item->snapshot)) return text->len == 0;
    size_t candidates = 0;
    const char *value = text->ptr ? text->ptr : "";
    bool valid = frontier ? edit_selection_frontier_prefix(frontier, value, text->len, complete, &candidates) :
        edit_selection_prefix(item->snapshot, value, text->len, complete, &candidates);
    if (speculative) *speculative = candidates == 1 && (frontier ?
        edit_selection_frontier_prefix(frontier, value, text->len, true, NULL) :
        edit_selection_prefix(item->snapshot, value, text->len, true, NULL));
    return valid;
}

/* A bare marker already present in the selected source is literal when the
 * complete spelling can still continue that source. Partial spellings stay
 * provisional so an explicit ID can disambiguate the separator. */
static bool edit_document_marker_literal(const edit_document_item *item, const buf *body,
        const char *marker, const edit_selection_frontier *frontier) {
    size_t cursor = 0, length = strlen(marker);
    const char *value = body->ptr ? body->ptr : "", *next;
    while ((next = frontier ?
        edit_selection_frontier_candidate(frontier, value, body->len, &cursor, NULL) :
        edit_selection_candidate(item->snapshot, value, body->len, &cursor, NULL)))
        if (!strncmp(next, marker, length)) return true;
    return false;
}

static void edit_document_body_restart(edit_document_view *view, size_t offset, size_t logical_bytes) {
    view->checkpoint.offset = offset;
    view->checkpoint.index = view->index;
    view->checkpoint.old_body = view->phase == EDIT_DOCUMENT_OLD;
    view->checkpoint.new_body = view->phase == EDIT_DOCUMENT_NEW;
    view->checkpoint.started = true;
    view->checkpoint.logical_bytes = view->checkpoint.old_body ? logical_bytes : 0;
}

static int edit_document_scalar(edit_xml_cursor *c, buf *out) {
    unsigned char lead = (unsigned char)c->text[c->pos];
    if (!lead) return -1;
    size_t need = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
    size_t left = c->length - c->pos, n = left < need ? left : need;
    if (!interactive_utf8_valid((const unsigned char *)c->text + c->pos, n, left >= need)) return -1;
    buf_append(out, c->text + c->pos, n);
    c->pos += n;
    return left >= need ? 1 : 0;
}

typedef struct {
    const char *text;
    size_t length, committed;
    bool complete;
    edit_document_checkpoint restart;
} edit_document_input;

static edit_document_view edit_document_scan(const edit_document *state,
        const edit_document_input *input, buf *selected) {
    const char *text = input->text;
    size_t n = input->length, committed = input->committed;
    bool complete = input->complete;
    edit_document_checkpoint restart = input->restart;
    edit_document_view view = {.valid = true, .phase = EDIT_DOCUMENT_STRUCTURE,
        .index = restart.index, .checkpoint = restart};
    view.checkpoint.offset = view.checkpoint.logical_bytes = 0;
    edit_xml_cursor cursor = {.text = text, .length = n, .committed = committed};
    edit_xml_cursor *c = &cursor;
    buf path = {0}, body = {0};
    const edit_document_item *item = NULL;
    const edit_selection_frontier *frontier = NULL;
    int result = 0;
    char opening[sizeof(EDIT_OPEN_SUFFIX_FORMAT) + EDIT_ID_DIGITS];
    char separator[sizeof(EDIT_SEPARATOR_FORMAT) + EDIT_ID_DIGITS];
#define EDIT_TAKE(s) do { result = edit_xml_take(c, (s)); if (result != 1) goto stop; } while (0)
    if (restart.old_body || restart.new_body) {
        if (!state || view.index >= state->count) { result = -1; goto stop; }
        item = &state->item[view.index];
        if (!item->snapshot) { result = -1; goto stop; }
        if (restart.old_body) { frontier = &state->selection; goto selection; }
        goto replacement;
    }
    for (;;) {
        edit_xml_skip_space(c);
        if (c->pos == n) {
            view.valid = !complete;
            break;
        }
header:
        if (view.index >= INTERACTIVE_MAX_RECORDS) { result = -1; goto stop; }
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.speculative = view.native_escape = false;
        EDIT_TAKE(EDIT_OPEN_PREFIX);
        path.len = 0;
        if (path.ptr) path.ptr[0] = 0;
        while (c->pos < n && text[c->pos] != '"') {
            if (text[c->pos] == '<' || text[c->pos] == '\r' || text[c->pos] == '\n' || text[c->pos] == '\t') {
                result = -1; goto stop;
            }
            result = edit_xml_text_byte(c, &path);
            if (result != 1) goto stop;
            if (path.len > INTERACTIVE_MAX_VALUES) { result = -1; goto stop; }
        }
        if (c->pos == n) { result = 0; goto stop; }
        c->pos++;
        if (!path.len || !edit_xml_unicode(path.ptr, true)) { result = -1; goto stop; }
        item = state && view.index < state->count ? &state->item[view.index] : NULL;
        if (!item || !item->snapshot) {
            view.phase = EDIT_DOCUMENT_AWAIT;
            if (!item) view.path = xstrdup(path.ptr);
            view.valid = !complete && c->pos == n;
            break;
        }
        if (strcmp(path.ptr, item->path)) { result = -1; goto stop; }
        snprintf(opening, sizeof(opening), EDIT_OPEN_SUFFIX_FORMAT, item->identifier);
        EDIT_TAKE(opening);
        view.checkpoint = (edit_document_checkpoint){.offset = c->pos, .index = view.index, .old_body = true};
        body.len = 0;
        if (body.ptr) body.ptr[0] = 0;
        frontier = NULL;
selection:
        view.phase = EDIT_DOCUMENT_OLD;
        view.native_escape = true;
        snprintf(separator, sizeof(separator), EDIT_SEPARATOR_FORMAT, item->identifier);
        while (c->pos < n) {
            const char *marker = edit_xml_prefix(c, separator) ? separator :
                edit_xml_prefix(c, EDIT_SEPARATOR) ? EDIT_SEPARATOR : NULL;
            bool literal_marker = marker && n - c->pos >= strlen(marker) &&
                edit_document_marker_literal(item, &body, marker, frontier);
            if (marker && !literal_marker && edit_document_literal(item, &body, true, NULL, frontier)) {
                result = edit_xml_take(c, marker);
                if (result != 1) { view.native_escape = false; goto stop; }
                restart = (edit_document_checkpoint){0};
                view.checkpoint = (edit_document_checkpoint){.offset = c->pos,
                    .index = view.index, .new_body = true,
                    .separator_id = marker == separator};
                view.phase = EDIT_DOCUMENT_STRUCTURE;
                view.native_escape = false;
                /* Body sampling starts after its structural separator. */
                if (c->pos > committed && c->pos != n) { result = -1; goto stop; }
                goto replacement;
            }
            result = edit_document_scalar(c, &body);
            if (result == 0 && !edit_document_literal(item, &body, false, NULL, frontier)) result = -1;
            if (result != 1) { view.native_escape = false; goto stop; }
            if (!edit_document_literal(item, &body, false, &view.speculative, frontier)) { result = -1; goto stop; }
            edit_document_body_restart(&view, c->pos, body.len);
        }
        if (!edit_document_literal(item, &body, false, &view.speculative, frontier)) { result = -1; goto stop; }
        result = 0;
        goto stop;
replacement:
        view.phase = EDIT_DOCUMENT_NEW;
        view.speculative = view.native_escape = true;
        view.payload_start = !restart.started && c->pos == n;
        body.len = 0;
        if (body.ptr) body.ptr[0] = 0;
        snprintf(separator, sizeof(separator), EDIT_SEPARATOR_FORMAT, item->identifier);
        while (c->pos < n) {
            if (edit_xml_prefix(c, EDIT_OPEN_PREFIX)) {
                size_t start = c->pos, available = n - start;
                const char *prefix = EDIT_OPEN_PREFIX;
                if (available < strlen(prefix) && !complete) {
                    view.speculative = view.native_escape = false;
                    result = 0; goto stop;
                }
                if (available >= strlen(prefix)) {
                    view.index++;
                    view.checkpoint = (edit_document_checkpoint){.offset = start, .index = view.index};
                    restart = (edit_document_checkpoint){0};
                    goto header;
                }
            }
            const char *reserved = separator;
            if (!view.checkpoint.separator_id && !edit_xml_prefix(c, reserved))
                reserved = EDIT_SEPARATOR;
            if (edit_xml_prefix(c, reserved)) {
                edit_xml_cursor probe = *c;
                int taken = edit_xml_take(&probe, reserved);
                if (taken == 1) { result = -1; goto stop; }
                if (!complete && taken == 0) { result = 0; view.native_escape = false; goto stop; }
            }
            result = edit_document_scalar(c, &body);
            if (result != 1) goto stop;
            edit_document_body_restart(&view, c->pos, 0);
        }
        view.valid = !complete || (state && view.index + 1 == state->count);
        if (complete && view.valid) view.phase = EDIT_DOCUMENT_DONE;
        break;
    }
    goto done;
stop:
    view.valid = result == 0 && !complete;
    if (view.phase == EDIT_DOCUMENT_STRUCTURE) view.speculative = false;
done:
    if (selected && view.checkpoint.old_body && view.checkpoint.logical_bytes)
        buf_append(selected, body.ptr, view.checkpoint.logical_bytes);
    buf_free(&path); buf_free(&body);
    return view;
#undef EDIT_TAKE
}


#endif
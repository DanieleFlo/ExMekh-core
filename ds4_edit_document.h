#ifndef DS4_EDIT_DOCUMENT_H
#define DS4_EDIT_DOCUMENT_H

/* The final new_text close precedes </edit>; earlier bare closes are text.
 * Probes only read state. Path authorizations are committed by the runtime,
 * never by a vocabulary/trie probe. */
typedef struct {
    char *path;
    edit_constraint *snapshot;
} edit_document_item;

/* The runtime only appends to the accepted transcript. Pending delimiters and
 * UTF-8 scalars stay after this validated restart; probes never advance it. */
typedef struct {
    size_t offset, index, characters, logical_bytes;
    bool old_body, new_body, cdata, all, started;
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

/* Safe tokens cannot complete XML/native delimiters or a pending UTF-8 scalar. */
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

/* Repeated paths share the initial snapshot; a contradictory reply never
 * mutates authorization state or replaces the earlier snapshot. */
static bool edit_document_provide(edit_document *document, const char *text, bool create) {
    if (!document || !document->count) return false;
    edit_document_item *item = &document->item[document->count - 1];
    if (item->snapshot) return false;
    edit_constraint *snapshot = edit_document_shared_snapshot(document);
    if (snapshot && !edit_constraint_matches(snapshot, text, create)) return false;
    if (!snapshot) {
        snapshot = xmalloc(sizeof(*snapshot));
        edit_constraint_init(snapshot, text, create);
        document->snapshots[document->snapshot_count++] = snapshot;
    }
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

static bool edit_xml_character(unsigned c) {
    return c == 9 || c == 10 || c == 13 || (c >= 32 && c <= 0xd7ff) ||
        (c >= 0xe000 && c <= 0xfffd) || (c >= 0x10000 && c <= 0x10ffff);
}

static bool edit_xml_unicode(const char *text, bool complete) {
    size_t length = strlen(text);
    if (!interactive_utf8_valid((const unsigned char *)text, length, complete)) return false;
    const unsigned char *p = (const unsigned char *)text;
    const unsigned char *end = p + length;
    while (*p) {
        unsigned value = *p++, need = 0;
        if (value >= 0xf0) { need = 3; value &= 7; }
        else if (value >= 0xe0) { need = 2; value &= 15; }
        else if (value >= 0xc0) { need = 1; value &= 31; }
        if ((size_t)(end - p) < need) return !complete;
        while (need--) value = (value << 6) | (*p++ & 63);
        if (!edit_xml_character(value)) return false;
    }
    return true;
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

/* A pending native escape cannot authorize arbitrary amp; repetition in OLD.
 * Its eventual decoded spelling must still occur in an authorized snapshot.
 * Ordinary XML/entity prefixes are handled by the document recognizer first. */
static bool edit_document_native_escape_possible(const edit_document *document,
        edit_document_checkpoint checkpoint, edit_document_checkpoint restart, const buf *selected,
        const char *pending, size_t n, const char *end) {
    if (!document || !checkpoint.old_body || checkpoint.index >= document->count) return false;
    size_t cursor = 0;
    const char *suffix;
    while ((suffix = edit_document_selection_candidate(document, checkpoint, restart, selected, &cursor, NULL))) {
        buf encoded = {0};
        if (!strncmp(suffix, end, strlen(end))) {
            buf_puts(&encoded, checkpoint.cdata ? "&lt;" : "&amp;lt;");
            buf_puts(&encoded, end + 1);
        } else if (ds4_tool_text_escaped_close(suffix, end)) {
            buf_puts(&encoded, checkpoint.cdata ? "&amp;" : "&amp;amp;");
            buf_append(&encoded, suffix + 1, strnlen(suffix + 1, n));
        }
        bool possible = encoded.len >= n && !memcmp(encoded.ptr, pending, n);
        buf_free(&encoded);
        if (possible) return true;
    }
    return false;
}

/* The document owns snapshot lookup and literal selection. Callers receive
 * only an XML continuation; EOF and delimiters return control to the sampler. */
static void edit_document_draft_suffix(const edit_document *document,
        edit_document_checkpoint checkpoint, edit_document_checkpoint restart,
        const buf *selected, buf *out, size_t limit) {
    if (!document || !checkpoint.old_body || checkpoint.index >= document->count) return;
    const char *value = selected->ptr ? selected->ptr : "", *suffix = NULL;
    size_t cursor = 0;
    bool unique = false;
    if (restart.old_body && restart.index == checkpoint.index) {
        if (!edit_selection_frontier_prefix(&document->selection, value, selected->len, true, NULL)) return;
    } else {
        const edit_constraint *snapshot = document->item[checkpoint.index].snapshot;
        if (!edit_selection_prefix(snapshot, value, selected->len, true, NULL)) return;
    }
    suffix = edit_document_selection_candidate(document, checkpoint, restart, selected, &cursor, &unique);
    if (!suffix || !unique || !checkpoint.started) return;
    while (*suffix && out->len < limit) {
        if (checkpoint.cdata && !strncmp(suffix, "]]>", 3)) break;
        unsigned char ch = (unsigned char)*suffix;
        size_t n = ch < 0x80 ? 1 : ch < 0xe0 ? 2 : ch < 0xf0 ? 3 : 4;
        char scalar[5] = {0};
        memcpy(scalar, suffix, n);
        if (!edit_xml_unicode(scalar, true)) break;
        const char *piece = scalar;
        if (!checkpoint.cdata) {
            if (ch == '<') piece = "&lt;";
            else if (ch == '&') piece = "&amp;";
            /* Escaping ']' also handles a preceding raw ']' at the boundary. */
            else if (ch == ']') piece = "&#93;";
        }
        size_t bytes = strlen(piece);
        if (bytes > limit - out->len) break;
        buf_append(out, piece, bytes);
        suffix += n;
    }
}

/* Context-independent tokens safe in both XML text and CDATA. */
static bool edit_document_plain_piece(const char *piece, size_t n) {
    if (!piece || !n || memchr(piece, 0, n) || memchr(piece, '<', n) ||
        memchr(piece, '&', n) || memchr(piece, ']', n)) return false;
    return strlen(piece) == n && edit_xml_unicode(piece, true);
}

/* 1: consumed, 0: viable fragment, -1: incompatible. */
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
    if (i == start || !edit_xml_character(value)) return -1;
    edit_xml_utf8(out, value);
    c->pos = i + 1;
    return 1;
}

static int edit_xml_text_byte(edit_xml_cursor *c, buf *out) {
    unsigned char ch = (unsigned char)c->text[c->pos];
    if (ch == '&') return edit_xml_entity(c, out);
    if (ch < 0x80 && !edit_xml_character(ch)) return -1;
    buf_putc(out, c->text[c->pos++]);
    return 1;
}

/* Checkpoint whole scalars, keeping a fragmented UTF-8 scalar in the tail.
 * Entity decoding also produces whole scalars. Physical CR/LF are preserved. */
static int edit_xml_scalar(edit_xml_cursor *c, buf *out) {
    unsigned char lead = (unsigned char)c->text[c->pos];
    if (lead < 0x80) {
        if (!edit_xml_character(lead)) return -1;
        buf_putc(out, c->text[c->pos++]);
        return 1;
    }
    size_t need = lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
    size_t left = c->length - c->pos, n = left < need ? left : need;
    if (!interactive_utf8_valid((const unsigned char *)c->text + c->pos, n, left >= need)) return -1;
    buf_append(out, c->text + c->pos, n);
    c->pos += n;
    if (left < need) return 0;
    /* UTF-8 validation alone admits XML-forbidden noncharacters. */
    unsigned value = lead & (need == 2 ? 31 : need == 3 ? 15 : 7);
    for (size_t i = 1; i < n; i++) value = (value << 6) | ((unsigned char)c->text[c->pos - n + i] & 63);
    return edit_xml_character(value) ? 1 : -1;
}

/* Preserve physical CR/LF bytes, unlike a general XML parser. */
static void edit_document_body_restart(edit_document_view *view, size_t offset,
                                       bool cdata, size_t logical_bytes) {
    if (!view || (view->phase != EDIT_DOCUMENT_NEW && view->phase != EDIT_DOCUMENT_OLD)) return;
    view->checkpoint.offset = offset;
    view->checkpoint.index = view->index;
    view->checkpoint.old_body = view->phase == EDIT_DOCUMENT_OLD;
    view->checkpoint.new_body = view->phase == EDIT_DOCUMENT_NEW;
    view->checkpoint.started = true;
    view->checkpoint.logical_bytes = view->checkpoint.old_body ? logical_bytes : 0;
    view->checkpoint.cdata = cdata;
}

static int edit_xml_cdata_content(edit_xml_cursor *c, buf *out,
                                  edit_document_view *view) {
    while (c->pos < c->length) {
        if (edit_xml_prefix(c, "]]>") ) {
            int result = edit_xml_take(c, "]]>");
            if (result == 1) edit_document_body_restart(view, c->pos, false, out->len);
            return result;
        }
        int result = edit_xml_scalar(c, out);
        if (result != 1) return result;
        edit_document_body_restart(view, c->pos, true, out->len);
    }
    return 0;
}

static int edit_xml_cdata(edit_xml_cursor *c, buf *out, edit_document_view *view) {
    int result = edit_xml_take(c, "<![CDATA[");
    if (result != 1) return result;
    edit_document_body_restart(view, c->pos, true, out->len);
    return edit_xml_cdata_content(c, out, view);
}

/* Ranges are lexical strings: no integer conversion or file-range policy. */
static int edit_xml_number(edit_xml_cursor *c, size_t maximum, bool bounded) {
    size_t start = c->pos, value = 0;
    while (c->pos < c->length && c->text[c->pos] >= '0' && c->text[c->pos] <= '9') {
        unsigned digit = (unsigned)(c->text[c->pos] - '0');
        if (bounded && ((c->pos == start && !digit) ||
            value > maximum / 10 || (value == maximum / 10 && digit > maximum % 10))) return -1;
        if (bounded) value = value * 10 + digit;
        c->pos++;
    }
    if (c->pos == c->length) return 0;
    return c->pos > start && c->text[c->pos] == '"' ? 1 : -1;
}

static bool edit_document_barrier(const edit_xml_cursor *c) {
    return c->pos > c->committed && c->pos != c->length;
}

static bool edit_document_literal(const edit_document_item *item, const buf *text,
        bool complete, bool *speculative, const edit_selection_frontier *frontier) {
    size_t candidates = 0;
    if (!item || !item->snapshot) return false;
    const char *value = text->ptr ? text->ptr : "";
    bool valid = frontier ? edit_selection_frontier_prefix(frontier, value, text->len, complete, &candidates) :
        edit_selection_prefix(item->snapshot, value, text->len, complete, &candidates);
    if (speculative) *speculative = candidates == 1 && (frontier ?
        edit_selection_frontier_prefix(frontier, value, text->len, true, NULL) :
        edit_selection_prefix(item->snapshot, value, text->len, true, NULL));
    return valid;
}

static unsigned edit_document_next_scalar(const char *text) {
    const unsigned char *p = (const unsigned char *)text;
    unsigned value = *p++, need = 0;
    if (value >= 0xf0) { value &= 7; need = 3; }
    else if (value >= 0xe0) { value &= 15; need = 2; }
    else if (value >= 0xc2) { value &= 31; need = 1; }
    else if (value >= 0x80) return 0;
    while (need--) value = (value << 6) | (*p++ & 63);
    return value;
}

static bool edit_document_literal_has_next(const edit_document_item *item,
        const buf *logical, const edit_selection_frontier *frontier) {
    size_t cursor = 0;
    const char *value = logical->ptr ? logical->ptr : "", *next;
    while ((next = frontier ? edit_selection_frontier_candidate(frontier, value, logical->len, &cursor, NULL) :
            edit_selection_candidate(item->snapshot, value, logical->len, &cursor, NULL)))
        if (*next && edit_xml_character(edit_document_next_scalar(next))) return true;
    return false;
}

static bool edit_document_entity_scalar(unsigned scalar, const char *raw, size_t n) {
    const char *names[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    const unsigned values[] = {'&', '<', '>', '"', '\''};
    for (size_t i=0; i<sizeof(names)/sizeof(names[0]); i++)
        if (scalar == values[i] && n <= strlen(names[i]) && !memcmp(raw,names[i],n)) return true;
    if (!n || (n == 1 && raw[0] == '&')) return edit_xml_character(scalar);
    if (n < 2 || memcmp(raw,"&#",2) || !edit_xml_character(scalar)) return false;
    if (n == 2) return true;
    bool hex = raw[2] == 'x';
    size_t start = hex ? 3 : 2;
    while (start < n && raw[start] == '0') start++;
    char digits[16]; snprintf(digits,sizeof(digits),hex ? "%x" : "%u",scalar);
    if (n-start > strlen(digits)) return false;
    for (size_t i=start; i<n; i++) {
        unsigned ch=(unsigned char)raw[i];
        if (hex && ch >= 'A' && ch <= 'F') ch += 'a'-'A';
        if (ch != (unsigned char)digits[i-start]) return false;
    }
    return true;
}

/* A partial entity/native escape must still match a possible next scalar.
 * In particular an escape cannot be appended at EOF, or inside a closing
 * delimiter. This tests snapshot bytes, never future tokenizer paths. */
static bool edit_document_escape_candidate(const edit_document_item *item, const buf *logical,
        const char *entity, size_t n, const char *native_end, const edit_selection_frontier *frontier) {
    size_t cursor = 0;
    const char *next;
    const char *value = logical->ptr ? logical->ptr : "";
    while ((next = frontier ? edit_selection_frontier_candidate(frontier, value, logical->len, &cursor, NULL) :
            edit_selection_candidate(item->snapshot, value, logical->len, &cursor, NULL))) {
        if (!*next) continue;
        if (entity ? edit_document_entity_scalar(edit_document_next_scalar(next),entity,n) :
                     ds4_tool_text_needs_escape(next,native_end)) return true;
    }
    return false;
}

/* Body parser returns at its end delimiter. Text inside CDATA never has
 * command semantics, including a byte-exact closing delimiter. */
static int edit_document_body(edit_xml_cursor *c, const edit_document_item *item,
                               bool old, bool *all, edit_document_view *view, const char *native_end,
        const edit_document_checkpoint *restart, const edit_selection_frontier *frontier, buf *selected) {
    const char *closing = old ? "</old_text>" : "</new_text>";
    buf logical = {0};
    int result = 0;
    bool command = false, unfinished_markup = false;
    /* Content is owned by the transcript. The sentinel only preserves the
     * rule that a delete command must be the entire original body. */
    if (!old && restart && restart->started) buf_putc(&logical, 'x');
    if (restart && restart->cdata) {
        result = edit_xml_cdata_content(c, &logical, view);
        if (result != 1) goto done;
    }
    if (old && (!restart || !restart->started) && edit_xml_prefix(c, "<all/>")) {
        command = true; view->phase = EDIT_DOCUMENT_STRUCTURE;
        result = edit_xml_take(c, "<all/>");
        if (result != 1) goto done;
        *all = true;
    } else if (old && (!restart || !restart->started) && edit_xml_prefix(c, "<line number=\"")) {
        command = true; view->phase = EDIT_DOCUMENT_STRUCTURE;
        result = edit_xml_take(c, "<line number=\"");
        if (result != 1) goto done;
        result = edit_xml_number(c, edit_constraint_line_count(item->snapshot) + 1, true);
        if (result != 1) goto done;
        result = edit_xml_take(c, "\"/>");
        if (result != 1) goto done;
    } else if (!old && (!restart || !restart->started) &&
               c->length - c->pos >= 2 && edit_xml_prefix(c, "<delete/>")) {
        if (!*all || edit_constraint_is_creation(item->snapshot)) { result = -1; goto done; }
        command = true; view->phase = EDIT_DOCUMENT_STRUCTURE;
        result = edit_xml_take(c, "<delete/>");
        if (result != 1) goto done;
    }
    if (command) view->phase = EDIT_DOCUMENT_STRUCTURE;
    while (c->pos < c->length) {
        if (!command && c->length - c->pos == 1 && c->text[c->pos] == '<') {
            unfinished_markup = true;
            result = 0; goto done;
        }
        /* A bare close inside the replacement is literal unless it ends the
         * edit. Empty/partial lookahead keeps streaming prefixes viable. */
        edit_xml_cursor tail = *c;
        size_t closing_length = strlen(closing);
        if (c->length - c->pos >= closing_length) tail.pos += closing_length;
        edit_xml_skip_space(&tail);
        if (edit_xml_prefix(c, closing) && (old ||
            c->length - c->pos < closing_length || edit_xml_prefix(&tail, "</edit>"))) {
            view->native_escape = false;
            if (!old && !command && c->length - c->pos >= closing_length &&
                tail.pos == c->length) {
                /* This can still be a literal close followed by more text.
                 * Preserve body sampling; defer speculation until resolved. */
                view->speculative = false;
                unfinished_markup = true;
                result = 0; goto done;
            }
            if (old && !command && !edit_document_literal(item, &logical, true, NULL, frontier)) {
                result = -1; goto done;
            }
            if (!command && !edit_xml_unicode(logical.ptr ? logical.ptr : "", true)) {
                result = -1; goto done;
            }
            result = edit_xml_take(c, closing);
            view->phase = EDIT_DOCUMENT_STRUCTURE;
            goto done;
        }
        if (command) { result = -1; goto done; }
        if (c->text[c->pos] == '<') {
            if (edit_xml_prefix(c, "<![CDATA[")) {
                /* At snapshot EOF only the OLD close can make progress.
                 * Empty CDATA wrappers otherwise admit an unbounded loop. */
                if (old && !edit_document_literal_has_next(item, &logical, frontier)) {
                    result = -1; goto done;
                }
                size_t opening = c->pos;
                result = edit_xml_cdata(c, &logical, view);
                unfinished_markup = !result && c->pos == opening;
            } else if (!old && edit_xml_prefix(c, "</new_text>")) {
                result = edit_xml_take(c, "</new_text>");
                unfinished_markup = !result;
                if (result == 1) buf_puts(&logical, "</new_text>");
            } else { result = -1; goto done; }
        } else {
            if (c->text[c->pos] == ']' && edit_xml_prefix(c, "]]>") ) {
                result = edit_xml_take(c, "]]>");
                if (result == 1) result = -1;
                else if (!result)
                    /* A pending ] or ]] is already literal content. Include
                     * it in snapshot matching: dropping it admits dead old_text
                     * prefixes when the snapshot's next byte differs. */
                    buf_append(&logical, c->text + c->pos, c->length - c->pos);
                goto done;
            }
            bool entity = c->text[c->pos] == '&';
            size_t entity_start = c->pos;
            result = entity ? edit_xml_entity(c, &logical) : edit_xml_scalar(c, &logical);
            if (old && entity && !result && !edit_document_escape_candidate(item, &logical,
                    c->text + entity_start, c->length - entity_start, NULL, frontier)) result = -1;
            unfinished_markup = entity && !result;
            if (result == 1) edit_document_body_restart(view, c->pos, false, logical.len);
        }
        if (result < 0) goto done;
        if (!result) goto done;
    }
    result = 0;
done:
    if (!edit_xml_unicode(logical.ptr ? logical.ptr : "", false)) result = -1;
    if (!result && old && !command && view->phase == EDIT_DOCUMENT_OLD &&
        !edit_document_literal(item, &logical, false, &view->speculative, frontier)) result = -1;
    if (!result && !command && view->phase != EDIT_DOCUMENT_STRUCTURE)
        /* A transport escape cannot finish an XML opener/entity or UTF-8
         * scalar. Accepting it there leaves no possible next token. */
        view->native_escape = !unfinished_markup &&
            edit_xml_unicode(logical.ptr ? logical.ptr : "", true) &&
            (!old || edit_document_escape_candidate(item,&logical,NULL,0,native_end,frontier));
    if (selected && old && view->checkpoint.old_body && view->checkpoint.logical_bytes)
        buf_append(selected, logical.ptr, view->checkpoint.logical_bytes);
    buf_free(&logical);
    return result;
}

static edit_document_view edit_document_scan_from(const edit_document *state,
        const char *text, size_t n, size_t committed, bool complete,
        edit_document_checkpoint restart, buf *selected) {
    edit_document_view view = {.valid = true, .phase = EDIT_DOCUMENT_STRUCTURE,
        .index = restart.index, .checkpoint = restart};
    view.checkpoint.offset = 0;
    view.checkpoint.logical_bytes = 0;
    edit_xml_cursor cursor = {.text = text, .length = n, .committed = committed};
    edit_xml_cursor *c = &cursor;
    buf path = {0};
    int result;
    bool all = restart.all;
    const edit_document_item *item = NULL;
#define EDIT_TAKE(s) do { result = edit_xml_take(c, (s)); if (result != 1) goto stop; } while (0)
    if (restart.new_body || restart.old_body) {
        item = state && view.index < state->count ? &state->item[view.index] : NULL;
        if (!item || !item->snapshot) { view.valid = false; goto done; }
        if (restart.old_body) goto selection;
        goto replacement;
    }
    for (;;) {
        edit_xml_skip_space(c);
        if (c->pos == n) {
            view.phase = view.index ? EDIT_DOCUMENT_DONE : EDIT_DOCUMENT_STRUCTURE;
            view.valid = !complete || (view.index && state && view.index == state->count);
            break;
        }
        if (view.index >= INTERACTIVE_MAX_RECORDS) { view.valid = false; break; }
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.native_escape = false;
        EDIT_TAKE("<edit path=\"");
        view.native_escape = true;
        path.len = 0;
        if (path.ptr) path.ptr[0] = 0;
        while (c->pos < n && text[c->pos] != '"') {
            if (text[c->pos] == '<' || text[c->pos] == '\r' || text[c->pos] == '\n' || text[c->pos] == '\t') {
                result = -1; goto stop;
            }
            result = edit_xml_text_byte(c, &path);
            if (result != 1) { view.native_escape = false; goto stop; }
            if (path.len > INTERACTIVE_MAX_VALUES) { result = -1; goto stop; }
        }
        if (c->pos == n) { result = 0; goto stop; }
        c->pos++;
        view.native_escape = false;
        if (!path.len || !edit_xml_unicode(path.ptr, true)) { result = -1; goto stop; }
        item = state && view.index < state->count ? &state->item[view.index] : NULL;
        if (!item) {
            view.phase = EDIT_DOCUMENT_AWAIT;
            view.path = xstrdup(path.ptr);
            view.valid = !complete && c->pos == n;
            break;
        }
        if (strcmp(path.ptr, item->path)) { result = -1; goto stop; }
        if (!item->snapshot) {
            view.phase = EDIT_DOCUMENT_AWAIT;
            view.valid = !complete && c->pos == n;
            break;
        }
        EDIT_TAKE(">");
        bool range = false;
        all = false;
        if (edit_xml_prefix(c, "<old_text>")) {
            EDIT_TAKE("<old_text>");
        } else {
            EDIT_TAKE("<old_text start_line=\"");
            result = edit_xml_number(c, 0, false); if (result != 1) goto stop;
            EDIT_TAKE("\" end_line=\"");
            result = edit_xml_number(c, 0, false); if (result != 1) goto stop;
            EDIT_TAKE("\">");
            range = true;
        }
        view.phase = EDIT_DOCUMENT_OLD;
        if (range) {
            EDIT_TAKE("</old_text>");
        } else {
            view.checkpoint = (edit_document_checkpoint){.offset = c->pos,
                .index = view.index, .old_body = true};
selection:
            view.phase = EDIT_DOCUMENT_OLD;
            result = edit_document_body(c, item, true, &all, &view,state->native_end,
                restart.old_body ? &restart : NULL, restart.old_body ? &state->selection : NULL, selected);
            if (result != 1) goto stop;
            restart = (edit_document_checkpoint){0};
        }
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.native_escape = false;
        if (edit_document_barrier(c)) { result = -1; goto stop; }
        EDIT_TAKE("<new_text>");
        view.checkpoint = (edit_document_checkpoint){.offset = c->pos,
            .index = view.index, .new_body = true, .all = all};
replacement:
        view.phase = EDIT_DOCUMENT_NEW;
        view.speculative = true;
        view.payload_start = !restart.started && c->pos == n;
        if (edit_document_barrier(c)) { result = -1; goto stop; }
        result = edit_document_body(c, item, false, &all, &view,state->native_end,
            restart.new_body ? &restart : NULL, NULL, NULL);
        restart = (edit_document_checkpoint){0};
        if (result != 1) goto stop;
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.native_escape = false;
        view.speculative = false;
        /* Both closing tags are structural. Keep their combined tokenizer
         * pieces viable, including lookahead later resolved as literal text. */
        edit_xml_skip_space(c);
        EDIT_TAKE("</edit>");
        view.index++;
        view.checkpoint = (edit_document_checkpoint){.offset = c->pos, .index = view.index};
    }
    goto done;
stop:
    if (!edit_xml_unicode(path.ptr ? path.ptr : "", false)) result = -1;
    if (!edit_xml_unicode(path.ptr ? path.ptr : "", true)) view.native_escape = false;
    view.valid = result == 0 && !complete;
    if (view.phase == EDIT_DOCUMENT_STRUCTURE) view.speculative = false;
done:
    buf_free(&path);
    return view;
#undef EDIT_TAKE
}

static edit_document_view edit_document_scan(const edit_document *state,
        const char *text, size_t n, size_t committed, bool complete) {
    return edit_document_scan_from(state, text, n, committed, complete,
        (edit_document_checkpoint){0}, NULL);
}

#endif

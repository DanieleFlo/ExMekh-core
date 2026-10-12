#ifndef DS4_EDIT_DOCUMENT_H
#define DS4_EDIT_DOCUMENT_H

/* The final new_text close precedes </edit>; earlier bare closes are text.
 * Probes only read state. Path authorizations are committed by the runtime,
 * never by a vocabulary/trie probe. */
typedef struct {
    char *path;
    edit_constraint *snapshot;
} edit_document_item;

typedef struct {
    edit_document_item item[INTERACTIVE_MAX_RECORDS];
    size_t count;
    edit_constraint *snapshots[INTERACTIVE_MAX_RECORDS];
    size_t snapshot_count;
    const char *native_end;
} edit_document;

static edit_document *edit_document_new(const char *native_end) {
    edit_document *document = xmalloc(sizeof(*document));
    memset(document, 0, sizeof(*document));
    document->native_end = native_end;
    return document;
}

static void edit_document_free(edit_document *document) {
    if (!document) return;
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

/* Preserve physical CR/LF bytes, unlike a general XML parser. */
static int edit_xml_cdata(edit_xml_cursor *c, buf *out) {
    int result = edit_xml_take(c, "<![CDATA[");
    if (result != 1) return result;
    while (c->pos < c->length) {
        if (edit_xml_prefix(c, "]]>") ) return edit_xml_take(c, "]]>");
        unsigned char ch = (unsigned char)c->text[c->pos];
        if (ch < 0x80 && !edit_xml_character(ch)) return -1;
        buf_putc(out, c->text[c->pos++]);
    }
    return 0;
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
                                   bool complete, bool *speculative) {
    size_t candidates = 0;
    if (!item || !item->snapshot) return false;
    bool valid = edit_selection_prefix(item->snapshot, text->ptr ? text->ptr : "",
                                        text->len, complete, &candidates);
    if (speculative) *speculative = candidates == 1 &&
        edit_selection_prefix(item->snapshot, text->ptr ? text->ptr : "", text->len, true, NULL);
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
        const char *entity, size_t n, const char *native_end) {
    size_t cursor = 0;
    const char *next;
    while ((next = edit_selection_candidate(item->snapshot,
            logical->ptr ? logical->ptr : "", logical->len, &cursor, NULL))) {
        if (!*next) continue;
        if (entity ? edit_document_entity_scalar(edit_document_next_scalar(next),entity,n) :
                     ds4_tool_text_needs_escape(next,native_end)) return true;
    }
    return false;
}

/* Body parser returns at its end delimiter. Text inside CDATA never has
 * command semantics, including a byte-exact closing delimiter. */
static int edit_document_body(edit_xml_cursor *c, const edit_document_item *item,
                               bool old, bool *all, edit_document_view *view, const char *native_end) {
    const char *closing = old ? "</old_text>" : "</new_text>";
    buf logical = {0};
    int result = 0;
    bool command = false, unfinished_markup = false;
    if (old && edit_xml_prefix(c, "<all/>")) {
        command = true; view->phase = EDIT_DOCUMENT_STRUCTURE;
        result = edit_xml_take(c, "<all/>");
        if (result != 1) goto done;
        *all = true;
    } else if (old && edit_xml_prefix(c, "<line number=\"")) {
        command = true; view->phase = EDIT_DOCUMENT_STRUCTURE;
        result = edit_xml_take(c, "<line number=\"");
        if (result != 1) goto done;
        result = edit_xml_number(c, edit_constraint_line_count(item->snapshot) + 1, true);
        if (result != 1) goto done;
        result = edit_xml_take(c, "\"/>");
        if (result != 1) goto done;
    } else if (!old && c->length - c->pos >= 2 && edit_xml_prefix(c, "<delete/>")) {
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
            if (old && !command && !edit_document_literal(item, &logical, true, NULL)) {
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
                size_t opening = c->pos;
                result = edit_xml_cdata(c, &logical);
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
            if (old && c->text[c->pos] == '&') {
                size_t entity_start=c->pos;
                result = edit_xml_entity(c,&logical);
                if (!result && !edit_document_escape_candidate(item,&logical,
                        c->text+entity_start,c->length-entity_start,NULL)) result = -1;
                unfinished_markup = !result;
            } else {
                bool entity = c->text[c->pos] == '&';
                result = edit_xml_text_byte(c, &logical);
                unfinished_markup = entity && !result;
            }
        }
        if (result < 0) goto done;
        if (!result) goto done;
    }
    result = 0;
    if (old && !command && !edit_document_literal(item, &logical, false, &view->speculative)) result = -1;
done:
    if (!edit_xml_unicode(logical.ptr ? logical.ptr : "", false)) result = -1;
    if (!result && old && !command && view->phase == EDIT_DOCUMENT_OLD &&
        !edit_document_literal(item, &logical, false, &view->speculative)) result = -1;
    if (!result && !command && view->phase != EDIT_DOCUMENT_STRUCTURE)
        /* A transport escape cannot finish an XML opener/entity or UTF-8
         * scalar. Accepting it there leaves no possible next token. */
        view->native_escape = !unfinished_markup &&
            edit_xml_unicode(logical.ptr ? logical.ptr : "", true) &&
            (!old || edit_document_escape_candidate(item,&logical,NULL,0,native_end));
    buf_free(&logical);
    return result;
}

static edit_document_view edit_document_scan(const edit_document *state,
        const char *text, size_t n, size_t committed, bool complete) {
    edit_document_view view = {.valid = true, .phase = EDIT_DOCUMENT_STRUCTURE};
    edit_xml_cursor cursor = {.text = text, .length = n, .committed = committed};
    edit_xml_cursor *c = &cursor;
    buf path = {0};
    int result;
#define EDIT_TAKE(s) do { result = edit_xml_take(c, (s)); if (result != 1) goto stop; } while (0)
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
        const edit_document_item *item = state && view.index < state->count ? &state->item[view.index] : NULL;
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
        bool range = false, all = false;
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
            result = edit_document_body(c, item, true, &all, &view,state->native_end);
            if (result != 1) goto stop;
        }
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.native_escape = false;
        if (edit_document_barrier(c)) { result = -1; goto stop; }
        EDIT_TAKE("<new_text>");
        view.phase = EDIT_DOCUMENT_NEW;
        view.speculative = true;
        view.payload_start = c->pos == n;
        if (edit_document_barrier(c)) { result = -1; goto stop; }
        result = edit_document_body(c, item, false, &all, &view,state->native_end);
        if (result != 1) goto stop;
        view.phase = EDIT_DOCUMENT_STRUCTURE;
        view.native_escape = false;
        view.speculative = false;
        /* Both closing tags are structural. Keep their combined tokenizer
         * pieces viable, including lookahead later resolved as literal text. */
        edit_xml_skip_space(c);
        EDIT_TAKE("</edit>");
        view.index++;
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

#endif

#ifndef DS4_EDIT_CONSTRAINT_H
#define DS4_EDIT_CONSTRAINT_H

/* File snapshots are control data. This recognizer knows neither paths nor
 * tokenizer IDs: it accepts prefixes of the native parameter representation. */
typedef struct {
    char *text;
    size_t length;
    buf wire;
    size_t *wire_offset;
    size_t *line_start;
    size_t *unique_after;
    size_t *literal_end;
    size_t lines;
    size_t *candidate_line;
    size_t bucket[257];
    bool create;
} edit_constraint;

static void edit_constraint_free(edit_constraint *c) {
    free(c->text);
    buf_free(&c->wire);
    free(c->wire_offset);
    free(c->line_start);
    free(c->unique_after);
    free(c->literal_end);
    free(c->candidate_line);
    memset(c, 0, sizeof(*c));
}

/* Adjacent suffix LCPs give the shortest globally unique prefix at each
 * physical line start. This is a text index, not an enumeration of possible
 * selections/token paths. Radix doubling and Kasai use O(n) memory. */
static void edit_index_unique(edit_constraint *c) {
    size_t n = c->length;
    if (!n) return;
    int *sa = xmalloc(n * sizeof(int)), *tmp = xmalloc(n * sizeof(int));
    int *rank = xmalloc(n * sizeof(int)), *next = xmalloc(n * sizeof(int));
    size_t capacity = n + 257;
    int *count = xmalloc(capacity * sizeof(int));
    for (size_t i = 0; i < n; i++) { sa[i] = (int)i; rank[i] = (unsigned char)c->text[i] + 1; }
    int classes = 256;
    for (size_t step = 1; step < n; step *= 2) {
        for (int pass = 0; pass < 2; pass++) {
            memset(count, 0, (size_t)(classes + 1) * sizeof(int));
            size_t shift = pass == 0 ? step : 0;
            int *input = pass == 0 ? sa : tmp;
            int *output = pass == 0 ? tmp : sa;
            for (size_t i = 0; i < n; i++) {
                size_t p = (size_t)input[i] + shift;
                count[p < n ? rank[p] : 0]++;
            }
            int offset = 0;
            for (int k = 0; k <= classes; k++) { int size = count[k]; count[k] = offset; offset += size; }
            for (size_t i = 0; i < n; i++) {
                size_t p = (size_t)input[i] + shift;
                output[count[p < n ? rank[p] : 0]++] = input[i];
            }
        }
        classes = 1; next[sa[0]] = 1;
        for (size_t i = 1; i < n; i++) {
            size_t a = (size_t)sa[i-1], b = (size_t)sa[i];
            if (rank[a] != rank[b] ||
                (a + step < n ? rank[a+step] : 0) !=
                (b + step < n ? rank[b+step] : 0)) classes++;
            next[b] = classes;
        }
        int *swap = rank; rank = next; next = swap;
        if ((size_t)classes == n) break;
    }
    /* n==1 has its initial character rank; inverse ranks must be contiguous. */
    for (size_t i = 0; i < n; i++) rank[sa[i]] = (int)i;
    size_t common = 0;
    for (size_t i = 0; i < n; i++) {
        int position = rank[i];
        if (!position) { next[0] = 0; common = 0; continue; }
        size_t previous = (size_t)sa[position-1];
        while (i + common < n && previous + common < n &&
               c->text[i+common] == c->text[previous+common]) common++;
        next[position] = (int)common;
        if (common) common--;
    }
    for (size_t line = 0; line < c->lines; line++) {
        size_t position = (size_t)rank[c->line_start[line]];
        int left = next[position], right = position + 1 < n ? next[position+1] : 0;
        c->unique_after[line] = (size_t)(left > right ? left : right) + 1;
    }
    size_t end = n, line = c->lines;
    for (size_t i = n; i > 0;) {
        i--;
        if (n - i >= strlen("@ex_meckh_") && !memcmp(c->text + i, "@ex_meckh_", strlen("@ex_meckh_"))) end = i;
        if (line && i == c->line_start[line-1]) c->literal_end[--line] = end;
    }
    free(sa); free(tmp); free(rank); free(next); free(count);
}

static void edit_constraint_init(edit_constraint *c, const char *text,
                                 bool create, const char *end) {
    memset(c, 0, sizeof(*c));
    c->text = xstrdup(text);
    c->length = strlen(text);
    c->create = create;
    c->wire_offset = xmalloc((c->length + 1) * sizeof(size_t));
    size_t line_capacity = 0;
    for (size_t i = 0; i < c->length; i++) {
        c->wire_offset[i] = c->wire.len;
        if (ds4_tool_text_needs_escape(text + i, end))
            buf_puts(&c->wire, text[i] == '<' ? "&lt;" : "&amp;");
        else buf_putc(&c->wire, text[i]);
        if (i == 0 || (text[i-1] == '\n') ||
            (text[i-1] == '\r' && text[i] != '\n')) {
            if (c->lines == line_capacity) {
                line_capacity = line_capacity ? line_capacity * 2 : 32;
                c->line_start = xrealloc(c->line_start, line_capacity * sizeof(size_t));
            }
            c->line_start[c->lines++] = i;
        }
    }
    c->wire_offset[c->length] = c->wire.len;
    c->unique_after = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    c->literal_end = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    edit_index_unique(c);
    /* First-byte buckets discard impossible token families without scanning
     * every physical line. Only branches capable of a unique closure enter. */
    c->candidate_line = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    for (size_t line = 0; line < c->lines; line++) {
        size_t start = c->line_start[line];
        if (c->unique_after[line] <= c->literal_end[line] - start)
            c->bucket[(unsigned char)c->wire.ptr[c->wire_offset[start]] + 1]++;
    }
    for (size_t i = 1; i < 257; i++) c->bucket[i] += c->bucket[i-1];
    size_t cursor[256]; memcpy(cursor, c->bucket, sizeof(cursor));
    for (size_t line = 0; line < c->lines; line++) {
        size_t start = c->line_start[line];
        if (c->unique_after[line] <= c->literal_end[line] - start)
            c->candidate_line[cursor[(unsigned char)c->wire.ptr[c->wire_offset[start]]]++] = line;
    }
}

static bool edit_marker_prefix(const char *value, size_t n,
                                const char *marker, bool complete) {
    size_t len = strlen(marker);
    return n <= len && (!complete || n == len) && !memcmp(value, marker, n);
}

static bool edit_line_marker_prefix(const char *value, size_t n,
                                     size_t max_line, bool complete) {
    const char *head = "@ex_meckh_line[";
    size_t h = strlen(head);
    if (n <= h) return !complete && !memcmp(value, head, n);
    if (memcmp(value, head, h) || value[h] < '1' || value[h] > '9') return false;
    size_t number = 0, i = h;
    while (i < n && value[i] >= '0' && value[i] <= '9') {
        unsigned digit = (unsigned)(value[i++] - '0');
        if (number > max_line / 10 ||
            (number == max_line / 10 && digit > max_line % 10))
            return false;
        number = number * 10 + digit;
    }
    if (i == n) return !complete;
    return i + 1 == n && value[i] == ']';
}

/* Closure is tested on logical bytes. A wire prefix ending in an escape or
 * UTF-8 fragment may remain viable, but cannot close the argument. */
static bool edit_selection_prefix(const edit_constraint *c, const char *value,
                                   size_t n, bool complete,
                                   size_t *candidate_count) {
    if (candidate_count) *candidate_count = 0;
    bool marker = edit_marker_prefix(value, n, "@ex_meckh_all", complete) ||
        edit_line_marker_prefix(value, n, c->lines + 1, complete);
    if (marker && !candidate_count) return true;
    bool literal = false;
    size_t first = n ? c->bucket[(unsigned char)value[0]] : 0;
    size_t last = n ? c->bucket[(unsigned char)value[0] + 1] : c->bucket[256];
    for (size_t candidate = first; candidate < last; candidate++) {
        size_t line = c->candidate_line[candidate];
        size_t start = c->line_start[line], wire = c->wire_offset[start];
        if (n > c->wire.len - wire || memcmp(value, c->wire.ptr + wire, n)) continue;
        size_t lo = start, hi = c->length + 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (c->wire_offset[mid] < wire + n) lo = mid + 1;
            else hi = mid;
        }
        size_t logical = lo - start;
        if (lo > c->literal_end[line]) continue;
        if (candidate_count) (*candidate_count)++;
        if (!complete) literal = true;
        else if (n && lo <= c->length && c->wire_offset[lo] == wire + n &&
                 (lo == c->length || ((unsigned char)c->text[lo] & 0xc0) != 0x80) &&
                 logical >= c->unique_after[line]) literal = true;
        if (literal && !candidate_count) return true;
    }
    return marker || literal;
}

static size_t edit_constraint_line_count(const edit_constraint *c) {
    return c->lines;
}

static bool edit_replacement_prefix(const edit_constraint *c, bool all,
                                     const char *value, size_t n, bool complete) {
    if (all && !c->create &&
        edit_marker_prefix(value, n, "@ex_meckh_delete", complete)) return true;
    /* Once the reserved family is complete it cannot become ordinary data. */
    const char *reserved = "@ex_meckh_";
    size_t r = strlen(reserved);
    for (size_t i = 0; i + r <= n; i++)
        if (!memcmp(value + i, reserved, r)) return false;
    return true;
}

#endif

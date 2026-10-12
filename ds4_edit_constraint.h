#ifndef DS4_EDIT_CONSTRAINT_H
#define DS4_EDIT_CONSTRAINT_H

/* Match logical UTF-8 bytes, independently of XML/native transport escapes. */
typedef struct {
    char *text;
    size_t length;
    size_t *line_start;
    size_t *unique_after;
    size_t lines;
    size_t *candidate_line;
    size_t *selection_line;
    size_t bucket[257];
    bool create;
} edit_constraint;

static size_t edit_constraint_count_lines(const char *text) {
    size_t lines = 0;
    for (size_t i = 0; text[i]; i++)
        if (!i || text[i-1] == '\n' || (text[i-1] == '\r' && text[i] != '\n')) lines++;
    return lines;
}

static size_t edit_constraint_line_count(const edit_constraint *c) {
    return c->lines;
}

static bool edit_constraint_is_creation(const edit_constraint *c) {
    return c->create;
}

static bool edit_constraint_matches(const edit_constraint *c, const char *text, bool create) {
    return c->create == create && !strcmp(c->text, text);
}

static size_t edit_constraint_memory_bound(size_t length) {
    /* Retained indexes plus radix-sort scratch, including fixed count bins
     * and the initial line allocation even for very small snapshots. */
    return 64 * (length + 1) + sizeof(edit_constraint) +
        257 * sizeof(int) + 32 * sizeof(size_t);
}

static void edit_constraint_free(edit_constraint *c) {
    free(c->text);
    free(c->line_start);
    free(c->unique_after);
    free(c->candidate_line);
    free(c->selection_line);
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
    /* Keep physical-line candidates in suffix order. Removing a common
     * accepted prefix preserves this order, so later probes use binary search. */
    size_t candidates = 0;
    for (size_t i = 0; i < n; i++) {
        size_t start = (size_t)sa[i];
        if (start && c->text[start - 1] != '\n' &&
            !(c->text[start - 1] == '\r' && c->text[start] != '\n')) continue;
        size_t low = 0, high = c->lines;
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (c->line_start[mid] < start) low = mid + 1;
            else high = mid;
        }
        if (c->unique_after[low] <= n - start) c->selection_line[candidates++] = low;
    }
    free(sa); free(tmp); free(rank); free(next); free(count);
}

static void edit_constraint_init(edit_constraint *c, const char *text, bool create) {
    memset(c, 0, sizeof(*c));
    c->text = xstrdup(text);
    c->length = strlen(text);
    c->create = create;
    size_t line_capacity = 0;
    for (size_t i = 0; i < c->length; i++) {
        if (i == 0 || (text[i-1] == '\n') ||
            (text[i-1] == '\r' && text[i] != '\n')) {
            if (c->lines == line_capacity) {
                line_capacity = line_capacity ? line_capacity * 2 : 32;
                c->line_start = xrealloc(c->line_start, line_capacity * sizeof(size_t));
            }
            c->line_start[c->lines++] = i;
        }
    }
    c->unique_after = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    c->candidate_line = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    c->selection_line = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
    edit_index_unique(c);
    /* First-byte buckets discard impossible token families without scanning
     * every physical line. Only branches capable of a unique closure enter. */
    for (size_t line = 0; line < c->lines; line++) {
        size_t start = c->line_start[line];
        if (c->unique_after[line] <= c->length - start)
            c->bucket[(unsigned char)c->text[start] + 1]++;
    }
    for (size_t i = 1; i < 257; i++) c->bucket[i] += c->bucket[i-1];
    size_t cursor[256]; memcpy(cursor, c->bucket, sizeof(cursor));
    for (size_t line = 0; line < c->lines; line++) {
        size_t start = c->line_start[line];
        if (c->unique_after[line] <= c->length - start)
            c->candidate_line[cursor[(unsigned char)c->text[start]]++] = line;
    }
}

/* Enumerate matching line starts without exposing the suffix index. Cursor
 * starts at zero; the returned suffix may be empty at snapshot EOF. */
static const char *edit_selection_candidate(const edit_constraint *c,
        const char *value, size_t n, size_t *cursor, bool *unique) {
    size_t first = n ? c->bucket[(unsigned char)value[0]] : 0;
    size_t last = n ? c->bucket[(unsigned char)value[0] + 1] : c->bucket[256];
    while (*cursor < last - first) {
        size_t line = c->candidate_line[first + (*cursor)++];
        size_t start = c->line_start[line];
        if (n > c->length - start || memcmp(value, c->text + start, n)) continue;
        if (unique) *unique = n && n >= c->unique_after[line];
        return c->text + start + n;
    }
    return NULL;
}

/* UTF-8 fragments may remain viable, but cannot close a selection. */
static bool edit_selection_prefix(const edit_constraint *c, const char *value,
                                   size_t n, bool complete,
                                   size_t *candidate_count) {
    if (candidate_count) *candidate_count = 0;
    bool literal = false;
    size_t cursor = 0;
    bool unique = false;
    const char *next;
    while ((next = edit_selection_candidate(c, value, n, &cursor, &unique))) {
        if (candidate_count) (*candidate_count)++;
        if (!complete) literal = true;
        else if (unique && (!*next || ((unsigned char)*next & 0xc0) != 0x80)) literal = true;
        if (literal && !candidate_count) return true;
    }
    return literal;
}

/* Only accepted logical bytes advance this frontier. Vocabulary and MTP
 * probes compare their suffix with the remaining snapshot candidates. */
typedef struct {
    const edit_constraint *snapshot;
    size_t first, last, length;
} edit_selection_frontier;

static void edit_selection_frontier_free(edit_selection_frontier *f) {
    memset(f, 0, sizeof(*f));
}

static void edit_selection_frontier_init(edit_selection_frontier *f,
                                          const edit_constraint *snapshot) {
    edit_selection_frontier_free(f);
    f->snapshot = snapshot;
    f->last = snapshot->bucket[256];
}

static size_t edit_selection_frontier_bound(const edit_selection_frontier *f,
        const char *value, size_t n, bool upper) {
    const edit_constraint *c = f->snapshot;
    size_t low = f->first, high = f->last;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        size_t start = c->line_start[c->selection_line[mid]] + f->length;
        size_t left = c->length - start;
        int comparison = memcmp(c->text + start, value, left < n ? left : n);
        if (!comparison && left < n) comparison = -1;
        if (comparison < 0 || (!comparison && upper)) low = mid + 1;
        else high = mid;
    }
    return low;
}

static const char *edit_selection_frontier_candidate(const edit_selection_frontier *f,
        const char *value, size_t n, size_t *cursor, bool *unique) {
    const edit_constraint *c = f->snapshot;
    size_t first = edit_selection_frontier_bound(f, value, n, false);
    size_t last = edit_selection_frontier_bound(f, value, n, true);
    if (*cursor < last - first) {
        size_t line = c->selection_line[first + (*cursor)++], start = c->line_start[line];
        if (unique) *unique = f->length + n >= c->unique_after[line];
        return c->text + start + f->length + n;
    }
    return NULL;
}

static bool edit_selection_frontier_prefix(const edit_selection_frontier *f,
        const char *value, size_t n, bool complete, size_t *candidate_count) {
    if (candidate_count) *candidate_count = 0;
    size_t first = edit_selection_frontier_bound(f, value, n, false);
    size_t last = edit_selection_frontier_bound(f, value, n, true);
    if (candidate_count) *candidate_count = last - first;
    if (first == last) return false;
    if (!complete) return true;
    if (last - first != 1) return false;
    const edit_constraint *c = f->snapshot;
    size_t line = c->selection_line[first], length = f->length + n;
    const char *next = c->text + c->line_start[line] + length;
    return length >= c->unique_after[line] && (!*next || ((unsigned char)*next & 0xc0) != 0x80);
}

static void edit_selection_frontier_advance(edit_selection_frontier *f,
                                            const char *value, size_t n) {
    size_t first = edit_selection_frontier_bound(f, value, n, false);
    f->last = edit_selection_frontier_bound(f, value, n, true);
    f->first = first;
    f->length += n;
}

#endif

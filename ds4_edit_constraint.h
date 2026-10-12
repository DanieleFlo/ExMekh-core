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
    /* Retained indexes plus radix-sort construction scratch. */
    return 64 * (length + 1) + sizeof(edit_constraint);
}

static void edit_constraint_free(edit_constraint *c) {
    free(c->text);
    free(c->line_start);
    free(c->unique_after);
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
    edit_index_unique(c);
    /* First-byte buckets discard impossible token families without scanning
     * every physical line. Only branches capable of a unique closure enter. */
    c->candidate_line = xmalloc((c->lines ? c->lines : 1) * sizeof(size_t));
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

#endif

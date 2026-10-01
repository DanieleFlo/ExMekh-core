#ifndef DS4_ENV_H
#define DS4_ENV_H

#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* Empty values use the default; other nonempty values retain legacy truthiness. */
static inline int ds4_env_flag_enabled(const char *name, int fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    return strcmp(value, "0") != 0 &&
           strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "off") != 0 &&
           strcasecmp(value, "no") != 0;
}

#endif

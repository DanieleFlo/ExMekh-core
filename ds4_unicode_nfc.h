#ifndef DS4_UNICODE_NFC_H
#define DS4_UNICODE_NFC_H
#include "ds4_runtime.h"
/* Unicode 9.0 NFC, matching the frozen image tokenizer (not the host locale).
 * Caller frees *out. Invalid UTF-8 and inputs over 4 MiB are rejected. */
ds4_status ds4_unicode_nfc(const char *, char **out);
#endif

// Android stand-in for <mach-o/getsect.h>: getsectiondata over this executable's ELF sections. The host's tables (hle.h DM_SECTION: swhle, swforce, swnvn) are
// ELF sections named after the Mach-O section, and the linker brackets each with __start_<name> / __stop_<name>, which getsectiondata looks up by the section
// name (the segment name is ignored). An unknown name, or a table with no entries, gives NULL like a missing Mach-O section.
#pragma once
#include <stdint.h>
struct mach_header_64 { uint32_t magic; };
extern const struct mach_header_64 _mh_execute_header;
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *segname, const char *sectname, unsigned long *size);

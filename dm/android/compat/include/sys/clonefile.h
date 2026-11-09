// Android stand-in for <sys/clonefile.h>: clonefile copies a file or a whole directory tree (Linux has no copy-on-write clone of a directory). The save
// transactions (hle_fs.c) only need a complete, independent copy at the new path; like clonefile, it fails with EEXIST when the destination exists and leaves
// nothing behind on failure (the partial copy is removed).
#pragma once
#define CLONE_NOFOLLOW 0x0001
#define CLONE_NOOWNERCOPY 0x0002
int clonefile(const char *src, const char *dst, unsigned int flags);

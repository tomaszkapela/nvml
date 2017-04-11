/*
 * Copyright 2017, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * os.h -- os abstaction layer
 */

#ifndef NVML_OS_H
#define NVML_OS_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/stat.h>
#include <stdio.h>

#ifndef _WIN32
typedef struct stat os_stat_t;
#define os_fstat	fstat
#define os_lseek	lseek
#else
typedef struct _stat64 os_stat_t;
#define os_fstat	_fstat64
#define os_lseek	_lseeki64
#endif

#define os_close close

int os_open(const char *pathname, int flags, ...);
int os_stat(const char *pathname, os_stat_t *buf);
int os_unlink(const char *pathname);
int os_access(const char *pathname, int mode);
FILE *os_fopen(const char *pathname, const char *mode);
FILE *os_fdopen(int fd, const char *mode);
int os_chmod(const char *pathname, mode_t mode);
int os_mkstemp(char *temp);

/*
 * XXX: missing APis (used in ut_file.c)
 *
 * ftruncate
 * rename
 * flock
 * read
 * write
 * writev
 * posix_fallocate
 * strsignal
 * rand_r
 * setenv
 * unsetenv
 * clock_gettime
 */

#ifdef __cplusplus
}
#endif

#endif /* os.h */

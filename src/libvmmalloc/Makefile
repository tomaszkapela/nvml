# Copyright 2014-2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#
#     * Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# src/libvmmalloc/Makefile -- Makefile for libvmmalloc
#

LIBRARY_NAME = vmmalloc
LIBRARY_SO_VERSION = 1
LIBRARY_VERSION = 0.0
SOURCE = libvmmalloc.c\
	$(COMMON)/file_linux.c\
	$(COMMON)/mmap.c\
	$(COMMON)/mmap_linux.c\
	$(COMMON)/out.c\
	$(COMMON)/os_linux.c\
	$(COMMON)/util.c\
	$(COMMON)/util_linux.c

default: all

include ../jemalloc/jemalloc.mk

INCS += -I$(JEMALLOC_DIR)/include/jemalloc
INCS += -I$(JEMALLOC_OBJDIR)/include/jemalloc
INCS += -I../libvmem
EXTRA_OBJS += $(JEMALLOC_LIB)
LIBS += -pthread

include ../Makefile.inc

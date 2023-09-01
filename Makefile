################################################################################
# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Edited by Marcos Luciano
# https://www.github.com/marcoslucianops
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CFLAGS:= -Wall -Wfatal-errors -fPIC -Wno-deprecated-declarations

APP:= deepstream

DS_SDK_ROOT:= /opt/nvidia/deepstream/deepstream

LIB_INSTALL_DIR?= $(DS_SDK_ROOT)/lib/

TARGET_DEVICE= $(shell gcc -dumpmachine | cut -f1 -d -)
ifeq ($(TARGET_DEVICE), aarch64)
	CFLAGS+= -DPLATFORM_TEGRA
endif

SRCS+= $(wildcard *.c)
SRCS+= $(wildcard modules/*.c)

INCS:= $(wildcard *.h)
INCS+= $(wildcard modules/*.h)

PKGS:= gstreamer-1.0

OBJS:= $(addsuffix .o, $(basename $(SRCS)))

CFLAGS+= -I$(DS_SDK_ROOT)/sources/apps/apps-common/includes -I$(DS_SDK_ROOT)/sources/includes \
 	     -I/usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= `pkg-config --cflags $(PKGS)`
LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)

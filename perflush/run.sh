#!/bin/bash -e

bench="./perflush"
tmpfs=/dev/shm/dram
devdax=/dev/dax5.0

tmpfs_out="tmpfs.log"
devdax_out="devdax.log"

rm -rf $tmpfs_out
rm -rf $devdax_out

touch $devdax_out
touch $tmpfs_out

$bench $tmpfs clflush_seq 64 >> $tmpfs_out
$bench $tmpfs clflush_rand 64 >> $tmpfs_out
$bench $tmpfs clflush_static 64 >> $tmpfs_out

$bench $devdax clflush_seq 64 >> $devdax_out
$bench $devdax clflush_rand 64 >> $devdax_out
$bench $devdax clflush_static 64 >> $devdax_out

$bench $tmpfs clflushopt_seq 64 >> $tmpfs_out
$bench $tmpfs clflushopt_rand 64 >> $tmpfs_out
$bench $tmpfs clflushopt_static 64 >> $tmpfs_out

$bench $devdax clflushopt_seq 64 >> $devdax_out
$bench $devdax clflushopt_rand 64 >> $devdax_out
$bench $devdax clflushopt_static 64 >> $devdax_out

$bench $tmpfs clflushopt_seq 256 >> $tmpfs_out
$bench $tmpfs clflushopt_rand 256 >> $tmpfs_out

$bench $devdax clflushopt_seq 256 >> $devdax_out
$bench $devdax clflushopt_rand 256 >> $devdax_out

$bench $tmpfs clflushopt_seq 1024 >> $tmpfs_out
$bench $tmpfs clflushopt_rand 1024 >> $tmpfs_out

$bench $devdax clflushopt_seq 1024 >> $devdax_out
$bench $devdax clflushopt_rand 1024 >> $devdax_out

$bench $tmpfs ntmemcpy_noflush 1024 >> $tmpfs_out
$bench $devdax ntmemcpy_noflush 1024 >> $devdax_out

$bench $tmpfs read_flushed 1024 >> $tmpfs_out
$bench $devdax read_flushed 1024 >> $devdax_out

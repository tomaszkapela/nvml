/*
 * Copyright 2016, Intel Corporation
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
 * memblock.c -- implementation of memory block
 *
 * Memory block is a representation of persistent object that resides in the
 * heap. A valid memory block must be either a huge (free or used) chunk or a
 * block inside a run.
 *
 * Huge blocks are 1:1 correlated with the chunk headers in the zone whereas
 * run blocks are represented by bits in corresponding chunk bitmap.
 *
 * This file contains implementations of abstract operations on memory blocks.
 * Instead of storing the mbops structure inside each memory block the correct
 * method implementation is chosen at runtime.
 */

#include <string.h>

#include "heap.h"
#include "memblock.h"
#include "out.h"
#include "valgrind_internal.h"

/*
 * memblock_autodetect_type -- looks for the corresponding chunk header and
 *	depending on the chunks type returns the right memory block type.
 */
enum memory_block_type
memblock_autodetect_type(struct memory_block *m, struct heap_layout *h)
{
	enum memory_block_type ret;

	switch (ZID_TO_ZONE(h, m->zone_id)->chunk_headers[m->chunk_id].type) {
		case CHUNK_TYPE_RUN:
			ret = MEMORY_BLOCK_RUN;
			break;
		case CHUNK_TYPE_FREE:
		case CHUNK_TYPE_USED:
		case CHUNK_TYPE_FOOTER:
			ret = MEMORY_BLOCK_HUGE;
			break;
		default:
			/* unreachable */
			FATAL("possible zone chunks metadata corruption");
	}
	return ret;
}

/*
 * huge_block_size -- returns the compile-time constant which defines the
 *	huge memory block size.
 */
static size_t
huge_block_size(struct memory_block *m, struct heap_layout *h)
{
	return CHUNKSIZE;
}

/*
 * run_block_size -- looks for the right chunk and returns the block size
 *	information that is attached to the run block metadata.
 */
static size_t
run_block_size(struct memory_block *m, struct heap_layout *h)
{
	struct zone *z = ZID_TO_ZONE(h, m->zone_id);
	struct chunk_run *run = (struct chunk_run *)&z->chunks[m->chunk_id];

	return run->block_size;
}

/*
 * run_get_data -- returns pointer to the data of a huge block
 */
static void *
huge_get_data(struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	void *data = &z->chunks[m->chunk_id].data;

	return data;
}

/*
 * run_get_data -- returns pointer to the data of a run block
 */
static void *
run_get_data(struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);

	struct chunk_run *run =
		(struct chunk_run *)&z->chunks[m->chunk_id].data;
	ASSERT(run->block_size != 0);

	return (char *)&run->data + (run->block_size * m->block_off);
}

/*
 * huge_block_offset -- huge chunks do not use the offset information of the
 *	memory blocks and must always be zeroed.
 */
static uint16_t
huge_block_offset(struct memory_block *m, struct palloc_heap *heap, void *ptr)
{
	return 0;
}

/*
 * run_block_offset -- calculates the block offset based on the number of bytes
 *	between the beginning of the chunk and the allocation data.
 *
 * Because the block offset is not represented in bytes but in 'unit size',
 * the number of bytes must also be divided by the chunks block size.
 * A non-zero remainder would mean that either the caller provided incorrect
 * pointer or the allocation algorithm created an invalid allocation block.
 */
static uint16_t
run_block_offset(struct memory_block *m, struct palloc_heap *heap, void *ptr)
{
	size_t block_size = MEMBLOCK_OPS(RUN, &m)->block_size(m, heap->layout);

	void *data = run_get_data(m, heap);
	uintptr_t diff = (uintptr_t)ptr - (uintptr_t)data;
	ASSERT(diff <= RUNSIZE);
	ASSERT((size_t)diff / block_size <= UINT16_MAX);
	ASSERT(diff % block_size == 0);
	uint16_t block_off = (uint16_t)((size_t)diff / block_size);

	return block_off;
}

/*
 * chunk_get_chunk_hdr_value -- (internal) get value of a header for redo log
 */
static uint64_t
chunk_get_chunk_hdr_value(uint16_t type, uint32_t size_idx)
{
	uint64_t val;
	COMPILE_ERROR_ON(sizeof(struct chunk_header) != sizeof(uint64_t));

	struct chunk_header hdr;
	hdr.type = type;
	hdr.size_idx = size_idx;
	hdr.flags = 0;
	memcpy(&val, &hdr, sizeof(val));

	return val;
}

/*
 * huge_prep_operation_hdr -- prepares the new value of a chunk header that will
 *	be set after the operation concludes.
 */
static void
huge_prep_operation_hdr(struct memory_block *m, struct palloc_heap *heap,
	enum memblock_state op, struct operation_context *ctx)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_header *hdr = &z->chunk_headers[m->chunk_id];

	/*
	 * Depending on the operation that needs to be performed a new chunk
	 * header needs to be prepared with the new chunk state.
	 */
	uint64_t val = chunk_get_chunk_hdr_value(
		op == MEMBLOCK_ALLOCATED ? CHUNK_TYPE_USED : CHUNK_TYPE_FREE,
		m->size_idx);

	operation_add_entry(ctx, hdr, val, OPERATION_SET);

	VALGRIND_DO_MAKE_MEM_NOACCESS(hdr + 1,
			(hdr->size_idx - 1) *
			sizeof(struct chunk_header));

	/*
	 * In the case of chunks larger than one unit the footer must be
	 * created immediately AFTER the persistent state is safely updated.
	 */
	if (m->size_idx == 1)
		return;

	struct chunk_header *footer = hdr + m->size_idx - 1;
	VALGRIND_DO_MAKE_MEM_UNDEFINED(footer, sizeof(*footer));

	val = chunk_get_chunk_hdr_value(CHUNK_TYPE_FOOTER, m->size_idx);

	/*
	 * It's only safe to write the footer AFTER the persistent part of
	 * the operation have been successfully processed because the footer
	 * pointer might point to a currently valid persistent state
	 * of a different chunk.
	 * The footer entry change is updated as transient because it will
	 * be recreated at heap boot regardless - it's just needed for runtime
	 * operations.
	 */
	operation_add_typed_entry(ctx,
		footer, val, OPERATION_SET, ENTRY_TRANSIENT);
}

/*
 * run_prep_operation_hdr -- prepares the new value for a select few bytes of
 *	a run bitmap that will be set after the operation concludes.
 *
 * It's VERY important to keep in mind that the particular value of the
 * bitmap this method is modifying must not be changed after this function
 * is called and before the operation is processed.
 */
static void
run_prep_operation_hdr(struct memory_block *m, struct palloc_heap *heap,
	enum memblock_state op, struct operation_context *ctx)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);

	struct chunk_run *r = (struct chunk_run *)&z->chunks[m->chunk_id];
	/*
	 * Free blocks are represented by clear bits and used blocks by set
	 * bits - which is the reverse of the commonly used scheme.
	 *
	 * Here a bit mask is prepared that flips the bits that represent the
	 * memory block provided by the caller - because both the size index and
	 * the block offset are tied 1:1 to the bitmap this operation is
	 * relatively simple.
	 */
	uint64_t bmask;
	if (m->size_idx == BITS_PER_VALUE) {
		ASSERTeq(m->block_off % BITS_PER_VALUE, 0);
		bmask = UINT64_MAX;
	} else {
		bmask = ((1ULL << m->size_idx) - 1ULL) <<
				(m->block_off % BITS_PER_VALUE);
	}

	/*
	 * The run bitmap is composed of several 8 byte values, so a proper
	 * element of the bitmap array must be selected.
	 */
	int bpos = m->block_off / BITS_PER_VALUE;

	/* the bit mask is applied immediately by the add entry operations */
	if (op == MEMBLOCK_ALLOCATED) {
		operation_add_entry(ctx, &r->bitmap[bpos],
			bmask, OPERATION_OR);
	} else if (op == MEMBLOCK_FREE) {
		operation_add_entry(ctx, &r->bitmap[bpos],
			~bmask, OPERATION_AND);
	} else {
		ASSERT(0);
	}
}

/*
 * huge_get_lock -- because huge memory blocks are always allocated from a
 *	single bucket there's no reason to lock them - the bucket itself is
 *	protected.
 */
static void *
huge_get_lock(struct memory_block *m, struct palloc_heap *heap)
{
	return NULL;
}

/*
 * run_get_lock -- gets the runtime mutex from the heap.
 */
static void *
run_get_lock(struct memory_block *m, struct palloc_heap *heap)
{
	return heap_get_run_lock(heap, m->chunk_id);
}

/*
 * huge_get_state -- returns whether a huge block is allocated or not
 */
static enum memblock_state
huge_get_state(struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_header *hdr = &z->chunk_headers[m->chunk_id];

	if (hdr->type == CHUNK_TYPE_USED)
		return MEMBLOCK_ALLOCATED;

	if (hdr->type == CHUNK_TYPE_FREE)
		return MEMBLOCK_FREE;

	return MEMBLOCK_STATE_UNKNOWN;
}

/*
 * huge_get_state -- returns whether a block from a run is allocated or not
 */
static enum memblock_state
run_get_state(struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_header *hdr = &z->chunk_headers[m->chunk_id];
	ASSERTeq(hdr->type, CHUNK_TYPE_RUN);

	struct chunk_run *r = (struct chunk_run *)&z->chunks[m->chunk_id];

	unsigned v = m->block_off / BITS_PER_VALUE;
	uint64_t bitmap = r->bitmap[v];
	unsigned b = m->block_off % BITS_PER_VALUE;

	unsigned b_last = b + m->size_idx;
	ASSERT(b_last <= BITS_PER_VALUE);

	for (unsigned i = b; i < b_last; ++i) {
		if (!BIT_IS_CLR(bitmap, i)) {
			return MEMBLOCK_ALLOCATED;
		}
	}

	return MEMBLOCK_FREE;
}

/*
 * run_claim -- marks the run as claimed by an owner in the current heap. This
 *	means that no-one but the actual owner use this memory block.
 */
static int
run_claim(const struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_run *r = (struct chunk_run *)&z->chunks[m->chunk_id];
	uint64_t claimant = r->incarnation_claim;
	if (claimant == heap->run_id)
		return -1; /* already claimed */

	VALGRIND_ADD_TO_TX(&r->incarnation_claim, sizeof(r->incarnation_claim));
	int ret = __sync_bool_compare_and_swap(&r->incarnation_claim,
		claimant, heap->run_id) ? 0 : -1;
	VALGRIND_SET_CLEAN(&r->incarnation_claim,
		sizeof(r->incarnation_claim));
	VALGRIND_REMOVE_FROM_TX(&r->incarnation_claim,
		sizeof(r->incarnation_claim));

	return ret;
}

/*
 * run_claim_revoke -- removes the claim of the current owner of the run
 */
static void
run_claim_revoke(const struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_run *r = (struct chunk_run *)&z->chunks[m->chunk_id];
	ASSERTeq(r->incarnation_claim, heap->run_id);

	VALGRIND_ADD_TO_TX(&r->incarnation_claim, sizeof(r->incarnation_claim));
	r->incarnation_claim = 0;
	VALGRIND_SET_CLEAN(&r->incarnation_claim,
		sizeof(r->incarnation_claim));
	VALGRIND_REMOVE_FROM_TX(&r->incarnation_claim,
		sizeof(r->incarnation_claim));
}

/*
 * run_is_claimed -- checks whether the run already has an owner in the current
 *	incarnation of the heap.
 */
static int
run_is_claimed(const struct memory_block *m, struct palloc_heap *heap)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
	struct chunk_run *r = (struct chunk_run *)&z->chunks[m->chunk_id];
	uint64_t claimant = r->incarnation_claim;
	if (claimant == heap->run_id)
		return 1;

	return 0;
}

const struct memory_block_ops mb_ops[MAX_MEMORY_BLOCK] = {
	[MEMORY_BLOCK_HUGE] = {
		.block_size = huge_block_size,
		.block_offset = huge_block_offset,
		.prep_hdr = huge_prep_operation_hdr,
		.get_lock = huge_get_lock,
		.get_state = huge_get_state,
		.get_data = huge_get_data,
		.claim = NULL,
		.claim_revoke = NULL,
		.is_claimed = NULL,
	},
	[MEMORY_BLOCK_RUN] = {
		.block_size = run_block_size,
		.block_offset = run_block_offset,
		.prep_hdr = run_prep_operation_hdr,
		.get_lock = run_get_lock,
		.get_state = run_get_state,
		.get_data = run_get_data,
		.claim = run_claim,
		.claim_revoke = run_claim_revoke,
		.is_claimed = run_is_claimed,
	}
};

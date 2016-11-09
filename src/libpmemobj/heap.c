/*
 * Copyright 2015-2016, Intel Corporation
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
 * heap.c -- heap implementation
 */

#include <errno.h>
#include <sys/queue.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <float.h>

#include "heap.h"
#include "out.h"
#include "util.h"
#include "sys_util.h"
#include "valgrind_internal.h"
#include "recycler.h"

#define MAX_RUN_LOCKS 1024

#define USE_PER_THREAD_BUCKETS

#define NCACHES_PER_CPU	2

/*
 * Value used to mark a reserved spot in the bucket array.
 */
#define BUCKET_RESERVED ((void *)0xFFFFFFFFULL)

/*
 * The last size that is handled by runs.
 */
#define MAX_RUN_SIZE (CHUNKSIZE / 2)

/*
 * Maximum number of bytes the allocation class generation algorithm can decide
 * to waste in a single run chunk.
 */
#define MAX_RUN_WASTED_BYTES 1024

/*
 * Converts size (in bytes) to bucket index.
 */
#define SIZE_TO_BID(_h, _s) ((_h)->bucket_map[SIZE_TO_ALLOC_BLOCKS(_s)])

/*
 * Allocation categories are used for allocation classes generation. Each one
 * defines the biggest handled size (in alloc blocks) and step of the generation
 * process. The bigger the step the bigger is the acceptable internal
 * fragmentation. For each category the internal fragmentation can be calculated
 * as: step/size. So for step == 1 the acceptable fragmentation is 0% and so on.
 */
#define MAX_ALLOC_CATEGORIES 5

/*
 * The first size (in alloc blocks) which is actually used in the allocation
 * class generation algorithm. All smaller sizes use the first predefined bucket
 * with the smallest run unit size.
 */
#define FIRST_GENERATED_CLASS_SIZE 2

static struct {
	size_t size;
	size_t step;
} categories[MAX_ALLOC_CATEGORIES] = {
	/* dummy category - the first allocation class is predefined */
	{FIRST_GENERATED_CLASS_SIZE, 0},

	{16, 1},
	{64, 2},
	{256, 4}
};

struct active_run {
	uint32_t chunk_id;
	uint32_t zone_id;
	SLIST_ENTRY(active_run) run;
};

struct bucket_cache {
	struct bucket *buckets[MAX_BUCKETS]; /* no default bucket */
};

struct heap_rt {
	struct bucket *default_bucket;
	struct bucket *buckets[MAX_BUCKETS];
	struct recycler *recyclers[MAX_BUCKETS];

	uint8_t *bucket_map;
	pthread_mutex_t run_locks[MAX_RUN_LOCKS];
	unsigned max_zone;
	unsigned zones_exhausted;
	size_t last_run_max_size;

	struct bucket_cache *caches;
	unsigned ncaches;
	uint32_t last_drained[MAX_BUCKETS];
};

static __thread unsigned Cache_idx = UINT32_MAX;
static unsigned Next_cache_idx;

/*
 * bucket_group_init -- (internal) creates new bucket group instance
 */
static void
bucket_group_init(struct bucket **buckets)
{
	for (int i = 0; i < MAX_BUCKETS; ++i)
		buckets[i] = NULL;
}

/*
 * bucket_group_destroy -- (internal) destroys bucket group instance
 */
static void
bucket_group_destroy(struct bucket **buckets)
{
	for (int i = 0; i < MAX_BUCKETS; ++i)
		if (buckets[i] != NULL)
			bucket_delete(buckets[i]);
}

/*
 * heap_max_zone -- (internal) calculates how many zones can the heap fit
 */
static unsigned
heap_max_zone(size_t size)
{
	unsigned max_zone = 0;
	size -= sizeof(struct heap_header);

	while (size >= ZONE_MIN_SIZE) {
		max_zone++;
		size -= size <= ZONE_MAX_SIZE ? size : ZONE_MAX_SIZE;
	}

	return max_zone;
}

/*
 * get_zone_size_idx -- (internal) calculates zone size index
 */
static uint32_t
get_zone_size_idx(uint32_t zone_id, unsigned max_zone, size_t heap_size)
{
	ASSERT(max_zone > 0);
	if (zone_id < max_zone - 1)
		return MAX_CHUNK;

	ASSERT(heap_size >= zone_id * ZONE_MAX_SIZE);
	size_t zone_raw_size = heap_size - zone_id * ZONE_MAX_SIZE;

	ASSERT(zone_raw_size >= (sizeof(struct zone_header) +
			sizeof(struct chunk_header) * MAX_CHUNK));
	zone_raw_size -= sizeof(struct zone_header) +
		sizeof(struct chunk_header) * MAX_CHUNK;

	size_t zone_size_idx = zone_raw_size / CHUNKSIZE;
	ASSERT(zone_size_idx <= UINT32_MAX);

	return (uint32_t)zone_size_idx;
}

/*
 * heap_chunk_write_footer -- writes a chunk footer
 */
static void
heap_chunk_write_footer(struct chunk_header *hdr, uint32_t size_idx)
{
	if (size_idx == 1) /* that would overwrite the header */
		return;

	VALGRIND_DO_MAKE_MEM_UNDEFINED(hdr + size_idx - 1, sizeof(*hdr));

	struct chunk_header f = *hdr;
	f.type = CHUNK_TYPE_FOOTER;
	f.size_idx = size_idx;
	*(hdr + size_idx - 1) = f;
	/* no need to persist, footers are recreated in heap_populate_buckets */
	VALGRIND_SET_CLEAN(hdr + size_idx - 1, sizeof(f));
}

/*
 * heap_chunk_init -- (internal) writes chunk header
 */
static void
heap_chunk_init(struct palloc_heap *heap, struct chunk_header *hdr,
	uint16_t type, uint32_t size_idx)
{
	struct chunk_header nhdr = {
		.type = type,
		.flags = 0,
		.size_idx = size_idx
	};
	VALGRIND_DO_MAKE_MEM_UNDEFINED(hdr, sizeof(*hdr));

	*hdr = nhdr; /* write the entire header (8 bytes) at once */
	pmemops_persist(&heap->p_ops, hdr, sizeof(*hdr));

	heap_chunk_write_footer(hdr, size_idx);
}

/*
 * heap_zone_init -- (internal) writes zone's first chunk and header
 */
static void
heap_zone_init(struct palloc_heap *heap, uint32_t zone_id)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);
	uint32_t size_idx = get_zone_size_idx(zone_id, heap->rt->max_zone,
			heap->size);

	heap_chunk_init(heap, &z->chunk_headers[0], CHUNK_TYPE_FREE, size_idx);

	struct zone_header nhdr = {
		.size_idx = size_idx,
		.magic = ZONE_HEADER_MAGIC,
	};
	z->header = nhdr;  /* write the entire header (8 bytes) at once */
	pmemops_persist(&heap->p_ops, &z->header, sizeof(z->header));
}

/*
 * heap_run_init -- (internal) creates a run based on a chunk
 */
static void
heap_run_init(struct palloc_heap *heap, struct bucket *b,
		struct chunk_header *hdr, struct chunk_run *run)
{
	ASSERTeq(b->type, BUCKET_RUN);
	struct bucket_run *r = (struct bucket_run *)b;

	/* add/remove chunk_run and chunk_header to valgrind transaction */
	VALGRIND_ADD_TO_TX(run, sizeof(*run));
	run->block_size = b->unit_size;
	pmemops_persist(&heap->p_ops, &run->block_size,
			sizeof(run->block_size));

	ASSERT(hdr->type == CHUNK_TYPE_FREE);

	/* set all the bits */
	memset(run->bitmap, 0xFF, sizeof(run->bitmap));

	unsigned nval = r->bitmap_nval;
	ASSERT(nval > 0);
	/* clear only the bits available for allocations from this bucket */
	memset(run->bitmap, 0, sizeof(uint64_t) * (nval - 1));
	run->bitmap[nval - 1] = r->bitmap_lastval;

	run->incarnation_claim = heap->run_id;
	VALGRIND_SET_CLEAN(&run->incarnation_claim,
		sizeof(run->incarnation_claim));

	VALGRIND_REMOVE_FROM_TX(run, sizeof(*run));

	pmemops_persist(&heap->p_ops, run->bitmap, sizeof(run->bitmap));

	VALGRIND_ADD_TO_TX(hdr, sizeof(*hdr));
	hdr->type = CHUNK_TYPE_RUN;
	VALGRIND_REMOVE_FROM_TX(hdr, sizeof(*hdr));

	pmemops_persist(&heap->p_ops, hdr, sizeof(*hdr));
}

/*
 * heap_run_insert -- (internal) inserts and splits a block of memory into a run
 */
static void
heap_run_insert(struct palloc_heap *heap, struct bucket *b, uint32_t chunk_id,
		uint32_t zone_id, uint32_t size_idx, uint16_t block_off)
{
	ASSERTeq(b->type, BUCKET_RUN);
	struct bucket_run *r = (struct bucket_run *)b;

	ASSERT(size_idx <= BITS_PER_VALUE);
	ASSERT(block_off + size_idx <= r->bitmap_nallocs);

	uint32_t unit_max = r->unit_max;
	struct memory_block m = {chunk_id, zone_id,
		unit_max - (block_off % unit_max), block_off};

	if (m.size_idx > size_idx)
		m.size_idx = size_idx;

	do {
		CNT_OP(b, insert, heap, m);
		ASSERT(m.size_idx <= UINT16_MAX);
		ASSERT(m.block_off + m.size_idx <= UINT16_MAX);
		m.block_off = (uint16_t)(m.block_off + (uint16_t)m.size_idx);
		size_idx -= m.size_idx;
		m.size_idx = size_idx > unit_max ? unit_max : size_idx;
	} while (size_idx != 0);
}

/*
 * heap_get_run_lock -- returns the lock associated with memory block
 */
pthread_mutex_t *
heap_get_run_lock(struct palloc_heap *heap, uint32_t chunk_id)
{
	return &heap->rt->run_locks[chunk_id % MAX_RUN_LOCKS];
}

/*
 * heap_process_run_metadata -- (internal) parses the run bitmap
 */
static uint32_t
heap_process_run_metadata(struct palloc_heap *heap, struct bucket *b,
	struct chunk_run *run, uint32_t chunk_id, uint32_t zone_id)
{
	ASSERTeq(b->type, BUCKET_RUN);
	struct bucket_run *r = (struct bucket_run *)b;

	ASSERT(RUN_NALLOCS(run->block_size) <= UINT16_MAX);

	uint16_t run_bits = (uint16_t)(RUNSIZE / run->block_size);
	ASSERT(run_bits < (MAX_BITMAP_VALUES * BITS_PER_VALUE));
	uint16_t block_off = 0;
	uint16_t block_size_idx = 0;
	uint32_t inserted_blocks = 0;

	for (unsigned i = 0; i < r->bitmap_nval; ++i) {
		uint64_t v = run->bitmap[i];
		ASSERT(BITS_PER_VALUE * i <= UINT16_MAX);
		block_off = (uint16_t)(BITS_PER_VALUE * i);
		if (v == 0) {
			heap_run_insert(heap, b, chunk_id, zone_id,
				BITS_PER_VALUE, block_off);
			inserted_blocks += BITS_PER_VALUE;
			continue;
		} else if (v == UINT64_MAX) {
			continue;
		}

		for (unsigned j = 0; j < BITS_PER_VALUE; ++j) {
			if (BIT_IS_CLR(v, j)) {
				block_size_idx++;
			} else if (block_size_idx != 0) {
				ASSERT(block_off >= block_size_idx);

				heap_run_insert(heap, b, chunk_id, zone_id,
					block_size_idx,
					(uint16_t)(block_off - block_size_idx));
				inserted_blocks += block_size_idx;
				block_size_idx = 0;
			}

			if ((block_off++) == run_bits) {
				i = MAX_BITMAP_VALUES;
				break;
			}
		}

		if (block_size_idx != 0) {
			ASSERT(block_off >= block_size_idx);

			heap_run_insert(heap, b, chunk_id, zone_id,
					block_size_idx,
					(uint16_t)(block_off - block_size_idx));
			inserted_blocks += block_size_idx;
			block_size_idx = 0;
		}
	}

	return inserted_blocks;
}

/*
 * heap_create_run -- (internal) initializes a new run on an existing free chunk
 */
static void
heap_create_run(struct palloc_heap *heap, struct bucket *b,
	uint32_t chunk_id, uint32_t zone_id)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);
	struct chunk_header *hdr = &z->chunk_headers[chunk_id];
	struct chunk_run *run = (struct chunk_run *)&z->chunks[chunk_id];

	ASSERT(hdr->size_idx == 1);
	ASSERT(hdr->type == CHUNK_TYPE_FREE);

	VALGRIND_DO_MAKE_MEM_UNDEFINED(run, sizeof(*run));
	heap_run_init(heap, b, hdr, run);
	heap_process_run_metadata(heap, b, run, chunk_id, zone_id);
}

/*
 * heap_reuse_run -- (internal) reuses existing run
 */
static uint32_t
heap_reuse_run(struct palloc_heap *heap, struct bucket *b,
	uint32_t chunk_id, uint32_t zone_id)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);
	struct chunk_header *hdr = &z->chunk_headers[chunk_id];
	struct chunk_run *run = (struct chunk_run *)&z->chunks[chunk_id];

	ASSERTeq(hdr->type, CHUNK_TYPE_RUN);
	ASSERTeq(hdr->size_idx, 1);
	ASSERTeq(b->unit_size, run->block_size);

	return heap_process_run_metadata(heap, b, run, chunk_id, zone_id);
}

/*
 * heap_find_first_free_bucket_slot -- (internal) searches for the first
 *	available bucket slot
 *
 * This function must be thread-safe because buckets can be created at runtime.
 */
static uint8_t
heap_find_first_free_bucket_slot(struct heap_rt *h)
{
	int n;
	for (n = 0; n < MAX_BUCKETS; ++n)
		if (util_bool_compare_and_swap64(&h->buckets[n],
				NULL, BUCKET_RESERVED))
			return (uint8_t)n;

	return MAX_BUCKETS;
}

/*
 * heap_create_alloc_class_buckets -- (internal) allocates all cache bucket
 * instances of the specified type
 */
static uint8_t
heap_create_alloc_class_buckets(struct heap_rt *h,
	size_t unit_size, unsigned unit_max, unsigned unit_max_alloc)
{
	uint8_t slot = heap_find_first_free_bucket_slot(h);
	if (slot == MAX_BUCKETS)
		goto out;

	h->buckets[slot] = &(bucket_run_new(slot, CONTAINER_CTREE,
			unit_size, unit_max, unit_max_alloc)->super);

	if (h->buckets[slot] == NULL)
		goto error_bucket_new;

	int i;
	for (i = 0; i < (int)h->ncaches; ++i) {
		h->caches[i].buckets[slot] =
			&(bucket_run_new(slot, CONTAINER_CTREE,
				unit_size, unit_max, unit_max_alloc)->super);
		if (h->caches[i].buckets[slot] == NULL)
			goto error_cache_bucket_new;
	}

out:
	return slot;

error_cache_bucket_new:
	bucket_delete(h->buckets[slot]);

	for (i -= 1; i >= 0; --i) {
		bucket_delete(h->caches[i].buckets[slot]);
	}

error_bucket_new:
	h->buckets[slot] = NULL; /* clear the reservation */

	return MAX_BUCKETS;
}

/*
 * heap_get_create_bucket_idx_by_unit_size -- (internal) retrieves or creates
 *	the memory bucket index that points to buckets that are responsible
 *	for allocations with the given unit size.
 */
static uint8_t
heap_get_create_bucket_idx_by_unit_size(struct heap_rt *h, uint64_t unit_size)
{
	uint8_t bucket_idx = SIZE_TO_BID(h, unit_size);
	if (h->buckets[bucket_idx]->unit_size != unit_size) {
		/*
		 * This code path is taken only if the allocation class
		 * generation algorithm have changed or the user created a
		 * custom allocation class in the previous incarnation of
		 * the pool. Normally all the buckets are created at
		 * initialization time.
		 */
		bucket_idx = heap_create_alloc_class_buckets(h, unit_size,
			RUN_UNIT_MAX, RUN_UNIT_MAX_ALLOC);

		if (bucket_idx == MAX_BUCKETS) {
			ERR("Failed to allocate new bucket class");
			return MAX_BUCKETS;
		}

		/*
		 * The created bucket has only one purpose - to handle the
		 * custom allocation class that was NOT generated by the
		 * algorithm. That's why it only ever handles one block size.
		 * If this is an unused bucket, then eventually it will
		 * be rendered redundant because all backing chunks
		 * will get freed.
		 */
		size_t supported_block = unit_size / ALLOC_BLOCK_SIZE;
		h->bucket_map[supported_block] = bucket_idx;
	}

	ASSERTne(bucket_idx, MAX_BUCKETS);
	return bucket_idx;
}

/*
 * heap_get_cache_bucket -- (internal) returns the bucket for given id from
 *	semi-per-thread cache
 */
static struct bucket *
heap_get_cache_bucket(struct heap_rt *heap, int bucket_id)
{
	/*
	 * Choose cache index only once in a threads lifetime.
	 * Sadly there are no thread exclusivity guarantees.
	 */
	while (Cache_idx == UINT32_MAX) {
		Cache_idx = __sync_fetch_and_add(&Next_cache_idx, 1);
	}

	return heap->caches[Cache_idx % heap->ncaches].buckets[bucket_id];
}

/*
 * heap_get_bucket_by_idx -- (internal) returns bucket with the given index
 */
static struct bucket *
heap_get_bucket_by_idx(struct heap_rt *heap, uint8_t idx)
{
#ifdef USE_PER_THREAD_BUCKETS
	return heap_get_cache_bucket(heap, idx);
#else
	return heap->buckets[idx];
#endif
}

/*
 * heap_get_best_bucket -- returns the bucket that best fits the requested size
 */
struct bucket *
heap_get_best_bucket(struct palloc_heap *heap, size_t size)
{
	struct heap_rt *rt = heap->rt;
	if (size <= rt->last_run_max_size) {
		return heap_get_bucket_by_idx(rt, SIZE_TO_BID(rt, size));
	} else {
		return rt->default_bucket;
	}
}

/*
 * heap_reclaim_run -- checks the run for available memory if unclaimed.
 *
 * Returns 1 if reclaimed chunk, 0 otherwise.
 */
static int
heap_reclaim_run(struct palloc_heap *heap, struct chunk_run *run,
	struct memory_block *m)
{
	if (MEMBLOCK_OPS(RUN, m)->claim(m, heap) != 0)
		return 0; /* this run already has an owner */

	pthread_mutex_t *lock = MEMBLOCK_OPS(RUN, m)->get_lock(m, heap);
	util_mutex_lock(lock);

	uint8_t bid = heap_get_create_bucket_idx_by_unit_size(heap->rt,
		run->block_size);
	struct bucket_run *r =
		(struct bucket_run *)heap_get_bucket_by_idx(heap->rt, bid);

	unsigned i;
	unsigned nval = r->bitmap_nval;
	for (i = 0; nval > 0 && i < nval - 1; ++i)
		if (run->bitmap[i] != 0)
			break;

	int empty = i == nval - 1 && run->bitmap[i] == r->bitmap_lastval;
	if (empty) {
		struct zone *z = ZID_TO_ZONE(heap->layout, m->zone_id);
		struct chunk_header *hdr = &z->chunk_headers[m->chunk_id];
		struct bucket *defb = heap_get_default_bucket(heap);

		/*
		 * The redo log ptr can be NULL if we are sure that there's only
		 * one persistent value modification in the entire operation
		 * context.
		 */
		struct operation_context ctx;
		operation_init(&ctx, heap->base, NULL, NULL);
		ctx.p_ops = &heap->p_ops;
		struct memory_block nb = *m;

		nb.block_off = 0;
		nb.size_idx = 1;
		heap_chunk_init(heap, hdr, CHUNK_TYPE_FREE, nb.size_idx);

		struct memory_block fm = heap_coalesce_huge(heap, nb);
		MEMBLOCK_OPS(HUGE, &fm)->prep_hdr(&fm, heap,
			MEMBLOCK_FREE, &ctx);

		operation_process(&ctx);

		CNT_OP(defb, insert, heap, fm);

		*m = fm;
	} else {
		recycler_put(heap->rt->recyclers[bid], m);
	}

	util_mutex_unlock(lock);

	return empty;
}

/*
 * heap_populate_buckets -- (internal) creates volatile state of memory blocks
 */
static int
heap_populate_buckets(struct palloc_heap *heap)
{
	struct heap_rt *h = heap->rt;

	if (h->zones_exhausted == h->max_zone)
		return ENOMEM;

	uint32_t zone_id = h->zones_exhausted++;
	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);

	/* ignore zone and chunk headers */
	VALGRIND_ADD_TO_GLOBAL_TX_IGNORE(z, sizeof(z->header) +
		sizeof(z->chunk_headers));

	if (z->header.magic != ZONE_HEADER_MAGIC)
		heap_zone_init(heap, zone_id);

	struct bucket *def_bucket = h->default_bucket;

	struct chunk_run *run = NULL;
	struct memory_block m = {0, zone_id, 0, 0};
	for (uint32_t i = 0; i < z->header.size_idx; ) {
		struct chunk_header *hdr = &z->chunk_headers[i];
		ASSERT(hdr->size_idx != 0);
		heap_chunk_write_footer(hdr, hdr->size_idx);

		m.chunk_id = i;
		m.size_idx = hdr->size_idx;

		switch (hdr->type) {
			case CHUNK_TYPE_RUN:
				run = (struct chunk_run *)&z->chunks[i];
				heap_reclaim_run(heap, run, &m);
				break;
			case CHUNK_TYPE_FREE:
				CNT_OP(def_bucket, insert, heap, m);
				break;
			case CHUNK_TYPE_USED:
				break;
			default:
				ASSERT(0);
		}

		i = m.chunk_id + m.size_idx; /* hdr might have changed */
	}

	return 0;
}

/*
 * heap_reclaim_zone_garbage -- (internal) creates volatile state of unused runs
 */
static int
heap_reclaim_zone_garbage(struct palloc_heap *heap, uint32_t zone_id)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);

	struct chunk_run *run = NULL;
	struct memory_block m = {0, zone_id, 0, 0};
	int rchunks = 0;

	for (uint32_t i = 0; i < z->header.size_idx; ) {
		struct chunk_header *hdr = &z->chunk_headers[i];
		ASSERT(hdr->size_idx != 0);

		m.chunk_id = i;
		m.size_idx = hdr->size_idx;

		switch (hdr->type) {
			case CHUNK_TYPE_RUN:
				run = (struct chunk_run *)&z->chunks[i];
				rchunks += heap_reclaim_run(heap, run, &m);
				break;
			case CHUNK_TYPE_FREE:
			case CHUNK_TYPE_USED:
				break;
			default:
				ASSERT(0);
		}

		i = m.chunk_id + m.size_idx; /* hdr might have changed */
	}

	return rchunks == 0 ? ENOMEM : 0;
}

/*
 * heap_reclaim_garbage -- (internal) creates volatile state of unused runs
 */
static int
heap_reclaim_garbage(struct palloc_heap *heap)
{
	struct memory_block m;
	for (size_t i = 0; i < MAX_BUCKETS; ++i) {
		while (recycler_get(heap->rt->recyclers[i], &m) == 0) {
			MEMBLOCK_OPS(RUN, &m)->claim_revoke(&m, heap);
		}
	}

	int ret = ENOMEM;
	for (unsigned i = 0; i < heap->rt->zones_exhausted; ++i) {
		if (heap_reclaim_zone_garbage(heap, i) == 0)
			ret = 0;
	}

	return ret;
}

/*
 * heap_get_default_bucket -- returns the bucket with CHUNKSIZE unit size
 */
struct bucket *
heap_get_default_bucket(struct palloc_heap *heap)
{
	return heap->rt->default_bucket;
}

/*
 * heap_ensure_bucket_filled -- (internal) refills the bucket if needed
 */
static int
heap_ensure_bucket_filled(struct palloc_heap *heap, struct bucket *b,
	uint32_t units)
{
	if (b->type == BUCKET_HUGE) {
		return (heap_reclaim_garbage(heap) == 0 ||
			heap_populate_buckets(heap) == 0) ? 0 : ENOMEM;
	}

	struct bucket_run *r = (struct bucket_run *)b;
	if (r->is_active) {
		CNT_OP(b, rm_all);
		MEMBLOCK_OPS(RUN, &r->active_memory_block)
			->claim_revoke(&r->active_memory_block, heap);

		r->is_active = 0;
	}

	struct heap_rt *h = heap->rt;
	struct memory_block m = {0, 0, 1, 0};

	if (recycler_get(h->recyclers[b->id], &m) == 0) {
		pthread_mutex_t *lock = MEMBLOCK_OPS(RUN, &m)
			->get_lock(&m, heap);

		util_mutex_lock(lock);
		heap_reuse_run(heap, b, m.chunk_id, m.zone_id);
		util_mutex_unlock(lock);

		r->active_memory_block = m;
		r->is_active = 1;

		return 0;
	}

	/* cannot reuse an existing run, create a new one */
	struct bucket *def_bucket = heap_get_default_bucket(heap);
	util_mutex_lock(&def_bucket->lock);
	if (heap_get_bestfit_block(heap, def_bucket, &m) == 0) {
		ASSERTeq(m.block_off, 0);

		heap_create_run(heap, b, m.chunk_id, m.zone_id);
		r->active_memory_block = m;
		r->is_active = 1;

		util_mutex_unlock(&def_bucket->lock);

		return 0;
	}
	util_mutex_unlock(&def_bucket->lock);

	/*
	 * Try the recycler again, the previous call to the bestfit_block for
	 * huge chunks might have reclaimed some unused runs.
	 */
	if (recycler_get(h->recyclers[b->id], &m) == 0) {
		uint32_t found = heap_reuse_run(heap, b, m.chunk_id, m.zone_id);
		if (found < units) {
			CNT_OP(b, rm_all);
			MEMBLOCK_OPS(RUN, &m)->claim_revoke(&m, heap);
			return ENOMEM;
		}

		r->active_memory_block = m;
		r->is_active = 1;

		return 0;
	}

	return ENOMEM;
}

/*
 * heap_find_or_create_alloc_class -- (internal) searches for the
 * biggest bucket allocation class for which unit_size is evenly divisible by n.
 * If no such class exists, create one.
 */
static uint8_t
heap_find_or_create_alloc_class(struct palloc_heap *heap, size_t n)
{
	struct heap_rt *h = heap->rt;
	COMPILE_ERROR_ON(MAX_BUCKETS > UINT8_MAX);

	for (int i = MAX_BUCKETS - 1; i >= 0; --i) {
		if (h->buckets[i] == NULL)
			continue;

		struct bucket_run *run = (struct bucket_run *)h->buckets[i];

		if (n % run->super.unit_size == 0 &&
			n / run->super.unit_size <= run->unit_max_alloc)
			return (uint8_t)i;
	}

	/*
	 * In order to minimize the wasted space at the end of the run the
	 * run data size must be divisible by the allocation class unit size
	 * with the smallest possible remainder, preferably 0.
	 */
	while ((RUNSIZE % n) > MAX_RUN_WASTED_BYTES) {
		n += ALLOC_BLOCK_SIZE;
	}

	/*
	 * Now that the desired unit size is found the existing classes need
	 * to be searched for possible duplicates. If a class with the
	 * calculated unit size already exists, simply return that.
	 */
	for (int i = MAX_BUCKETS - 1; i >= 0; --i) {
		if (h->buckets[i] == NULL)
			continue;
		if (h->buckets[i]->unit_size == n)
			return (uint8_t)i;
	}

	return heap_create_alloc_class_buckets(h, n,
		RUN_UNIT_MAX, RUN_UNIT_MAX_ALLOC);
}

/*
 * heap_find_min_frag_alloc_class -- searches for an existing allocation
 * class that will provide the smallest internal fragmentation for the given
 * size.
 */
static uint8_t
heap_find_min_frag_alloc_class(struct palloc_heap *h, size_t n)
{
	uint8_t best_bucket = MAX_BUCKETS;
	float best_frag = FLT_MAX;
	/*
	 * Start from the largest buckets in order to minimize unit size of
	 * allocated memory blocks.
	 */
	for (int i = MAX_BUCKETS - 1; i >= 0; --i) {
		if (h->rt->buckets[i] == NULL)
			continue;

		struct bucket_run *run = (struct bucket_run *)h->rt->buckets[i];

		size_t units = run->super.calc_units((struct bucket *)run, n);
		/* can't exceed the maximum allowed run unit max */
		if (units > run->unit_max_alloc)
			break;

		float frag = (float)(run->super.unit_size * units) / (float)n;
		if (frag == 1.f)
			return (uint8_t)i;

		ASSERT(frag >= 1.f);
		if (frag < best_frag) {
			best_bucket = (uint8_t)i;
			best_frag = frag;
		}
	}

	ASSERTne(best_bucket, MAX_BUCKETS);
	return best_bucket;
}

/*
 * heap_buckets_init -- (internal) initializes bucket instances
 */
static int
heap_buckets_init(struct palloc_heap *heap)
{
	struct heap_rt *h = heap->rt;

	h->last_run_max_size = MAX_RUN_SIZE;
	h->bucket_map = Malloc((MAX_RUN_SIZE / ALLOC_BLOCK_SIZE) + 1);
	if (h->bucket_map == NULL)
		goto error_bucket_map_malloc;

	h->default_bucket = &(bucket_huge_new(MAX_BUCKETS,
		CONTAINER_CTREE, CHUNKSIZE)->super);
	if (h->default_bucket == NULL)
		goto error_default_bucket_new;

	/*
	 * To make use of every single bit available in the run the unit size
	 * would have to be calculated using following expression:
	 * (RUNSIZE / (MAX_BITMAP_VALUES * BITS_PER_VALUE)), but to preserve
	 * cacheline alignment a little bit of memory at the end of the run
	 * is left unused.
	 */
	size_t size = 0;
	uint8_t slot = heap_create_alloc_class_buckets(h,
		MIN_RUN_SIZE, RUN_UNIT_MAX, RUN_UNIT_MAX_ALLOC);
	if (slot == MAX_BUCKETS)
		goto error_bucket_create;

	/*
	 * The first couple of bucket map slots are predefined and use the
	 * smallest bucket available.
	 */
	for (size_t i = 0; i < FIRST_GENERATED_CLASS_SIZE; ++i)
		h->bucket_map[i] = slot;

	/*
	 * Based on the defined categories, a set of allocation classes is
	 * created. The unit size of those classes is depended on the category
	 * initial size and step.
	 */
	for (int c = 1; c < MAX_ALLOC_CATEGORIES; ++c) {
		for (size_t i = categories[c - 1].size + 1;
			i <= categories[c].size; i += categories[c].step) {

			size = i + (categories[c].step - 1);
			if ((slot = heap_find_or_create_alloc_class(heap,
				size * ALLOC_BLOCK_SIZE)) == MAX_BUCKETS)
				goto error_bucket_create;
		}
	}

	/*
	 * Find the largest bucket and use it's unit size as run allocation
	 * threshold.
	 */
	for (slot = MAX_BUCKETS - 1;
			slot > 0 && h->buckets[slot] == NULL;
			--slot)
		;

	struct bucket_run *b = (struct bucket_run *)h->buckets[slot];

	/*
	 * The actual run might contain less unit blocks than the theoretical
	 * unit max variable. This may be the case for very large unit sizes.
	 */
	size_t real_unit_max = b->bitmap_nallocs < b->unit_max_alloc ?
		b->bitmap_nallocs : b->unit_max_alloc;

	size_t theoretical_run_max_size = b->super.unit_size * real_unit_max;

	h->last_run_max_size = MAX_RUN_SIZE > theoretical_run_max_size ?
		theoretical_run_max_size : MAX_RUN_SIZE;

	/*
	 * Now that the alloc classes are created, the bucket with the minimal
	 * internal fragmentation for that size is chosen.
	 */
	for (size_t i = FIRST_GENERATED_CLASS_SIZE;
		i <= h->last_run_max_size / ALLOC_BLOCK_SIZE; ++i) {
		uint8_t bucket = heap_find_min_frag_alloc_class(heap,
			i * ALLOC_BLOCK_SIZE);
		h->bucket_map[i] = bucket;
	}

#ifdef DEBUG
	/*
	 * Verify that each bucket's unit size points back to the bucket by the
	 * bucket map. This must be true for the default allocation classes,
	 * otherwise duplicate buckets will be created.
	 */
	for (size_t i = 0; i < MAX_BUCKETS; ++i) {
		if (h->buckets[i] != NULL) {
			struct bucket *b = h->buckets[i];
			size_t b_id = SIZE_TO_BID(h, b->unit_size);
			ASSERTeq(b_id, i);
		}
	}
#endif

	heap_populate_buckets(heap);

	return 0;

error_bucket_create:
	bucket_delete(h->default_bucket);
	bucket_group_destroy(h->buckets);
	for (unsigned i = 0; i < h->ncaches; ++i)
		bucket_group_destroy(h->caches[i].buckets);

error_default_bucket_new:
	Free(h->bucket_map);

error_bucket_map_malloc:
	return ENOMEM;
}

/*
 * heap_resize_chunk -- (internal) splits the chunk into two smaller ones
 */
static void
heap_resize_chunk(struct palloc_heap *heap,
	uint32_t chunk_id, uint32_t zone_id, uint32_t new_size_idx)
{
	uint32_t new_chunk_id = chunk_id + new_size_idx;

	struct zone *z = ZID_TO_ZONE(heap->layout, zone_id);
	struct chunk_header *old_hdr = &z->chunk_headers[chunk_id];
	struct chunk_header *new_hdr = &z->chunk_headers[new_chunk_id];

	uint32_t rem_size_idx = old_hdr->size_idx - new_size_idx;
	heap_chunk_init(heap, new_hdr, CHUNK_TYPE_FREE, rem_size_idx);
	heap_chunk_init(heap, old_hdr, CHUNK_TYPE_FREE, new_size_idx);

	struct bucket *def_bucket = heap->rt->default_bucket;
	struct memory_block m = {new_chunk_id, zone_id, rem_size_idx, 0};
	CNT_OP(def_bucket, insert, heap, m);
}

/*
 * heap_recycle_block -- (internal) recycles unused part of the memory block
 */
static void
heap_recycle_block(struct palloc_heap *heap, struct bucket *b,
		struct memory_block *m, uint32_t units)
{
	if (b->type == BUCKET_RUN) {
		ASSERT(units <= UINT16_MAX);
		ASSERT(m->block_off + units <= UINT16_MAX);
		struct memory_block r = {m->chunk_id, m->zone_id,
			m->size_idx - units, (uint16_t)(m->block_off + units)};
		CNT_OP(b, insert, heap, r);
	} else {
		heap_resize_chunk(heap, m->chunk_id, m->zone_id, units);
	}

	m->size_idx = units;
}

/*
 * heap_get_bestfit_block --
 *	extracts a memory block of equal size index
 */
int
heap_get_bestfit_block(struct palloc_heap *heap, struct bucket *b,
	struct memory_block *m)
{
	uint32_t units = m->size_idx;

	while (CNT_OP(b, get_rm_bestfit, m) != 0) {
		if (heap_ensure_bucket_filled(heap, b, units) != 0) {
			return ENOMEM;
		}
	}

	ASSERT(m->size_idx >= units);

	if (units != m->size_idx)
		heap_recycle_block(heap, b, m, units);

	return 0;
}

/*
 * heap_get_adjacent_free_block -- locates adjacent free memory block in heap
 */
static int
heap_get_adjacent_free_block(struct palloc_heap *heap,
	const struct memory_block *in, struct memory_block *out, int prev)
{
	struct zone *z = ZID_TO_ZONE(heap->layout, in->zone_id);
	struct chunk_header *hdr = &z->chunk_headers[in->chunk_id];
	out->zone_id = in->zone_id;

	if (prev) {
		if (in->chunk_id == 0)
			return ENOENT;

		struct chunk_header *prev_hdr =
			&z->chunk_headers[in->chunk_id - 1];
		out->chunk_id = in->chunk_id - prev_hdr->size_idx;

		if (z->chunk_headers[out->chunk_id].type != CHUNK_TYPE_FREE)
			return ENOENT;

		out->size_idx = z->chunk_headers[out->chunk_id].size_idx;
	} else { /* next */
		if (in->chunk_id + hdr->size_idx == z->header.size_idx)
			return ENOENT;

		out->chunk_id = in->chunk_id + hdr->size_idx;

		if (z->chunk_headers[out->chunk_id].type != CHUNK_TYPE_FREE)
			return ENOENT;

		out->size_idx = z->chunk_headers[out->chunk_id].size_idx;
	}

	return 0;
}

/*
 * heap_coalesce -- (internal) merges adjacent memory blocks
 */
static struct memory_block
heap_coalesce(struct palloc_heap *heap, struct memory_block *blocks[], int n)
{
	struct memory_block ret;
	struct memory_block *b = NULL;
	ret.size_idx = 0;
	for (int i = 0; i < n; ++i) {
		if (blocks[i] == NULL)
			continue;
		b = b ? b : blocks[i];
		ret.size_idx += blocks[i] ? blocks[i]->size_idx : 0;
	}

	ASSERTne(b, NULL);

	ret.chunk_id = b->chunk_id;
	ret.zone_id = b->zone_id;
	ret.block_off = b->block_off;

	return ret;
}

/*
 * heap_coalesce_huge -- finds neighbours of a huge block, removes them from the
 *	volatile state and returns the resulting block
 */
struct memory_block
heap_coalesce_huge(struct palloc_heap *heap, struct memory_block m)
{
	struct memory_block *blocks[3] = {NULL, &m, NULL};

	struct bucket *b = heap_get_default_bucket(heap);

	struct memory_block prev = {0, 0, 0, 0};
	if (heap_get_adjacent_free_block(heap, &m, &prev, 1) == 0 &&
		CNT_OP(b, get_rm_exact, prev) == 0) {
		blocks[0] = &prev;
	}

	struct memory_block next = {0, 0, 0, 0};
	if (heap_get_adjacent_free_block(heap, &m, &next, 0) == 0 &&
		CNT_OP(b, get_rm_exact, next) == 0) {
		blocks[2] = &next;
	}

	return heap_coalesce(heap, blocks, 3);
}

/*
 * heap_end -- returns first address after heap
 */
void *
heap_end(struct palloc_heap *h)
{
	ASSERT(h->rt->max_zone > 0);

	struct zone *last_zone = ZID_TO_ZONE(h->layout, h->rt->max_zone - 1);

	return &last_zone->chunks[last_zone->header.size_idx];
}

/*
 * heap_get_ncpus -- (internal) returns the number of available CPUs
 */
static unsigned
heap_get_ncpus(void)
{
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	if (cpus < 1)
		cpus = 1;
	return (unsigned)cpus;
}

/*
 * heap_get_ncaches -- (internal) returns the number of bucket caches according
 * to number of cpus and number of caches per cpu.
 */
static unsigned
heap_get_ncaches(void)
{
	return NCACHES_PER_CPU * heap_get_ncpus();
}

/*
 * heap_boot -- opens the heap region of the pmemobj pool
 *
 * If successful function returns zero. Otherwise an error number is returned.
 */
int
heap_boot(struct palloc_heap *heap, void *heap_start, uint64_t heap_size,
		uint64_t run_id, void *base, struct pmem_ops *p_ops)
{
	struct heap_rt *h = Malloc(sizeof(*h));
	int err;
	if (h == NULL) {
		err = ENOMEM;
		goto error_heap_malloc;
	}

	h->ncaches = heap_get_ncaches();
	h->caches = Malloc(sizeof(struct bucket_cache) * h->ncaches);
	if (h->caches == NULL) {
		err = ENOMEM;
		goto error_heap_cache_malloc;
	}

	h->max_zone = heap_max_zone(heap_size);
	h->zones_exhausted = 0;

	for (int i = 0; i < MAX_RUN_LOCKS; ++i)
		util_mutex_init(&h->run_locks[i], NULL);

	memset(h->last_drained, 0, sizeof(h->last_drained));

	heap->run_id = run_id;
	heap->p_ops = *p_ops;
	heap->layout = heap_start;
	heap->rt = h;
	heap->size = heap_size;
	heap->base = base;
	VALGRIND_DO_CREATE_MEMPOOL(heap->layout, 0, 0);

	bucket_group_init(h->buckets);

	for (unsigned i = 0; i < h->ncaches; ++i)
		bucket_group_init(h->caches[i].buckets);

	if ((err = heap_buckets_init(heap)) != 0)
		goto error_buckets_init;

	size_t rec_i;
	for (rec_i = 0; rec_i < MAX_BUCKETS; ++rec_i) {
		if ((h->recyclers[rec_i] = recycler_new(heap)) == NULL)
			goto error_recycler_new;
	}


	return 0;

error_recycler_new:
	for (size_t i = 0; i < rec_i; ++i)
		recycler_delete(h->recyclers[i]);
error_buckets_init:
	/* there's really no point in destroying the locks */
	Free(h->caches);
error_heap_cache_malloc:
	Free(h);
	heap->rt = NULL;
error_heap_malloc:
	return err;
}

/*
 * heap_write_header -- (internal) creates a clean header
 */
static void
heap_write_header(struct heap_header *hdr, size_t size)
{
	struct heap_header newhdr = {
		.signature = HEAP_SIGNATURE,
		.major = HEAP_MAJOR,
		.minor = HEAP_MINOR,
		.size = size,
		.chunksize = CHUNKSIZE,
		.chunks_per_zone = MAX_CHUNK,
		.reserved = {0},
		.checksum = 0
	};

	util_checksum(&newhdr, sizeof(newhdr), &newhdr.checksum, 1);
	*hdr = newhdr;
}

#ifdef USE_VG_MEMCHECK
/*
 * heap_vg_open -- notifies Valgrind about heap layout
 */
void
heap_vg_open(void *heap_start, uint64_t heap_size)
{
	VALGRIND_DO_MAKE_MEM_UNDEFINED(heap_start, heap_size);

	struct heap_layout *layout = heap_start;

	VALGRIND_DO_MAKE_MEM_DEFINED(&layout->header, sizeof(layout->header));

	unsigned zones = heap_max_zone(heap_size);

	for (unsigned i = 0; i < zones; ++i) {
		struct zone *z = ZID_TO_ZONE(layout, i);
		uint32_t chunks;

		VALGRIND_DO_MAKE_MEM_DEFINED(&z->header, sizeof(z->header));

		if (z->header.magic != ZONE_HEADER_MAGIC)
			continue;

		chunks = z->header.size_idx;

		for (uint32_t c = 0; c < chunks; ) {
			struct chunk_header *hdr = &z->chunk_headers[c];

			VALGRIND_DO_MAKE_MEM_DEFINED(hdr, sizeof(*hdr));

			if (hdr->type == CHUNK_TYPE_RUN) {
				struct chunk_run *run =
					(struct chunk_run *)&z->chunks[c];

				VALGRIND_DO_MAKE_MEM_DEFINED(run,
					sizeof(*run));
			}

			ASSERT(hdr->size_idx > 0);

			/* mark unused chunk headers as not accessible */
			VALGRIND_DO_MAKE_MEM_NOACCESS(&z->chunk_headers[c + 1],
					(hdr->size_idx - 1) *
					sizeof(struct chunk_header));

			c += hdr->size_idx;
		}

		/* mark all unused chunk headers after last as not accessible */
		VALGRIND_DO_MAKE_MEM_NOACCESS(&z->chunk_headers[chunks],
			(MAX_CHUNK - chunks) * sizeof(struct chunk_header));
	}
}
#endif

/*
 * heap_init -- initializes the heap
 *
 * If successful function returns zero. Otherwise an error number is returned.
 */
int
heap_init(void *heap_start, uint64_t heap_size, struct pmem_ops *p_ops)
{
	if (heap_size < HEAP_MIN_SIZE)
		return EINVAL;

	VALGRIND_DO_MAKE_MEM_UNDEFINED(heap_start, heap_size);

	struct heap_layout *layout = heap_start;
	heap_write_header(&layout->header, heap_size);
	pmemops_persist(p_ops, &layout->header, sizeof(struct heap_header));

	unsigned zones = heap_max_zone(heap_size);
	for (unsigned i = 0; i < zones; ++i) {
		pmemops_memset_persist(p_ops,
				&ZID_TO_ZONE(layout, i)->header,
				0, sizeof(struct zone_header));
		pmemops_memset_persist(p_ops,
				&ZID_TO_ZONE(layout, i)->chunk_headers,
				0, sizeof(struct chunk_header));

		/* only explicitly allocated chunks should be accessible */
		VALGRIND_DO_MAKE_MEM_NOACCESS(
			&ZID_TO_ZONE(layout, i)->chunk_headers,
			sizeof(struct chunk_header));
	}

	return 0;
}

/*
 * heap_cleanup -- cleanups the volatile heap state
 */
void
heap_cleanup(struct palloc_heap *heap)
{
	struct heap_rt *rt = heap->rt;

	bucket_delete(rt->default_bucket);

	bucket_group_destroy(rt->buckets);

	for (unsigned i = 0; i < rt->ncaches; ++i)
		bucket_group_destroy(rt->caches[i].buckets);

	for (int i = 0; i < MAX_RUN_LOCKS; ++i)
		util_mutex_destroy(&rt->run_locks[i]);

	Free(rt->bucket_map);

	Free(rt->caches);

	for (int i = 0; i < MAX_BUCKETS; ++i) {
		recycler_delete(rt->recyclers[i]);
	}

	VALGRIND_DO_DESTROY_MEMPOOL(heap->layout);

	Free(rt);
	heap->rt = NULL;
}

/*
 * heap_verify_header -- (internal) verifies if the heap header is consistent
 */
static int
heap_verify_header(struct heap_header *hdr)
{
	if (util_checksum(hdr, sizeof(*hdr), &hdr->checksum, 0) != 1) {
		ERR("heap: invalid header's checksum");
		return -1;
	}

	if (memcmp(hdr->signature, HEAP_SIGNATURE, HEAP_SIGNATURE_LEN) != 0) {
		ERR("heap: invalid signature");
		return -1;
	}

	return 0;
}

/*
 * heap_verify_zone_header --
 *	(internal) verifies if the zone header is consistent
 */
static int
heap_verify_zone_header(struct zone_header *hdr)
{
	if (hdr->size_idx == 0) {
		ERR("heap: invalid zone size");
		return -1;
	}

	return 0;
}

/*
 * heap_verify_chunk_header --
 *	(internal) verifies if the chunk header is consistent
 */
static int
heap_verify_chunk_header(struct chunk_header *hdr)
{
	if (hdr->type == CHUNK_TYPE_UNKNOWN) {
		ERR("heap: invalid chunk type");
		return -1;
	}

	if (hdr->type >= MAX_CHUNK_TYPE) {
		ERR("heap: unknown chunk type");
		return -1;
	}

	if (hdr->flags & CHUNK_FLAG_ZEROED) {
		ERR("heap: invalid chunk flags");
		return -1;
	}

	return 0;
}

/*
 * heap_verify_zone -- (internal) verifies if the zone is consistent
 */
static int
heap_verify_zone(struct zone *zone)
{
	if (zone->header.magic == 0)
		return 0; /* not initialized, and that is OK */

	if (zone->header.magic != ZONE_HEADER_MAGIC) {
		ERR("heap: invalid zone magic");
		return -1;
	}

	if (heap_verify_zone_header(&zone->header))
		return -1;

	uint32_t i;
	for (i = 0; i < zone->header.size_idx; ) {
		if (heap_verify_chunk_header(&zone->chunk_headers[i]))
			return -1;

		i += zone->chunk_headers[i].size_idx;
	}

	if (i != zone->header.size_idx) {
		ERR("heap: chunk sizes mismatch");
		return -1;
	}

	return 0;
}

/*
 * heap_check -- verifies if the heap is consistent and can be opened properly
 *
 * If successful function returns zero. Otherwise an error number is returned.
 */
int
heap_check(void *heap_start, uint64_t heap_size)
{
	if (heap_size < HEAP_MIN_SIZE) {
		ERR("heap: invalid heap size");
		return -1;
	}

	struct heap_layout *layout = heap_start;

	if (heap_size != layout->header.size) {
		ERR("heap: heap size missmatch");
		return -1;
	}

	if (heap_verify_header(&layout->header))
		return -1;

	for (unsigned i = 0; i < heap_max_zone(layout->header.size); ++i) {
		if (heap_verify_zone(ZID_TO_ZONE(layout, i)))
			return -1;
	}

	return 0;
}

#define PMALLOC_PTR_TO_OFF(heap, ptr)\
	((uintptr_t)(ptr) - (uintptr_t)(heap->base))

/*
 * heap_check_remote -- verifies if the heap of a remote pool is consistent
 *                      and can be opened properly
 *
 * If successful function returns zero. Otherwise an error number is returned.
 */
int
heap_check_remote(void *heap_start, uint64_t heap_size, struct remote_ops *ops)
{
	if (heap_size < HEAP_MIN_SIZE) {
		ERR("heap: invalid heap size");
		return -1;
	}

	struct heap_layout *layout = heap_start;

	struct heap_header header;
	if (ops->read(ops->ctx, ops->base, &header, &layout->header,
						sizeof(struct heap_header))) {
		ERR("heap: obj_read_remote error");
		return -1;
	}

	if (heap_size != header.size) {
		ERR("heap: heap size mismatch");
		return -1;
	}

	if (heap_verify_header(&header))
		return -1;

	for (unsigned i = 0; i < heap_max_zone(header.size); ++i) {
		struct zone zone_buff;
		if (ops->read(ops->ctx, ops->base, &zone_buff,
				ZID_TO_ZONE(layout, i), sizeof(struct zone))) {
			ERR("heap: obj_read_remote error");
			return -1;
		}

		if (heap_verify_zone(&zone_buff))
			return -1;
	}

	return 0;
}

/*
 * heap_run_foreach_object -- (internal) iterates through objects in a run
 */
static int
heap_run_foreach_object(struct palloc_heap *heap, object_callback cb,
		void *arg, struct chunk_run *run)
{
	uint64_t bs = run->block_size;
	uint64_t block_off;

	uint64_t bitmap_nallocs = RUN_NALLOCS(bs);
	uint64_t unused_bits = RUN_BITMAP_SIZE - bitmap_nallocs;
	uint64_t unused_values = unused_bits / BITS_PER_VALUE;
	uint64_t bitmap_nval = MAX_BITMAP_VALUES - unused_values;

	struct allocation_header *alloc;

	uint64_t i = 0;
	uint64_t block_start = 0;

	for (; i < bitmap_nval; ++i) {
		uint64_t v = run->bitmap[i];
		block_off = (BITS_PER_VALUE * (uint64_t)i);

		for (uint64_t j = block_start; j < BITS_PER_VALUE; ) {

			if (block_off + j >= bitmap_nallocs)
				break;

			if (!BIT_IS_CLR(v, j)) {
				alloc = (struct allocation_header *)
					(run->data + (block_off + j) * bs);
				j += (alloc->size / bs);
				if (cb(PMALLOC_PTR_TO_OFF(heap, alloc), arg)
						!= 0)
					return 1;
			} else {
				++j;
			}
		}
		block_start = 0;
	}

	return 0;
}

/*
 * heap_chunk_foreach_object -- (internal) iterates through objects in a chunk
 */
static int
heap_chunk_foreach_object(struct palloc_heap *heap, object_callback cb,
	void *arg, struct chunk_header *hdr, struct chunk *chunk)
{
	switch (hdr->type) {
		case CHUNK_TYPE_FREE:
			return 0;
		case CHUNK_TYPE_USED:
			return cb(PMALLOC_PTR_TO_OFF(heap, chunk), arg);
		case CHUNK_TYPE_RUN:
			return heap_run_foreach_object(heap, cb, arg,
				(struct chunk_run *)chunk);
		default:
			ASSERT(0);
	}

	return 0;
}

/*
 * heap_zone_foreach_object -- (internal) iterates through objects in a zone
 */
static int
heap_zone_foreach_object(struct palloc_heap *heap, object_callback cb,
	void *arg, struct zone *zone, struct memory_block start)
{
	if (zone->header.magic == 0)
		return 0;

	uint32_t i;
	for (i = start.chunk_id; i < zone->header.size_idx; ) {
		if (heap_chunk_foreach_object(heap, cb, arg,
			&zone->chunk_headers[i], &zone->chunks[i]) != 0)
			return 1;

		i += zone->chunk_headers[i].size_idx;
	}

	return 0;
}

/*
 * heap_foreach_object -- (internal) iterates through objects in the heap
 */
void
heap_foreach_object(struct palloc_heap *heap, object_callback cb, void *arg,
	struct memory_block start)
{
	struct heap_layout *layout = heap->layout;

	for (unsigned i = start.zone_id;
		i < heap_max_zone(layout->header.size); ++i)
		if (heap_zone_foreach_object(heap, cb, arg,
				ZID_TO_ZONE(layout, i), start) != 0)
			break;
}

"""Correctness tests for engine/prefix_cache.py's RadixTrie: chained
content hashing, full-block-only matching, ref-counting, and LRU eviction
of the leaves. Pure Python, no torch/CUDA -- see test_block_manager.py's
module docstring on why these run for real on CI, not just import-checked.
"""
from engine.prefix_cache import RadixTrie, hash_block


def blocks(*chunks):
    """chunks: any number of token-id lists -- concatenated into one flat
    list, the shape RadixTrie.match/insert take.
    """
    out = []
    for c in chunks:
        out.extend(c)
    return out


class TestMatch:
    def test_no_match_on_empty_trie(self):
        trie = RadixTrie(block_size=4)
        match = trie.match(list(range(8)))
        assert match.num_matched_tokens == 0
        assert match.physical_block_ids == []

    def test_exact_match_after_insert(self):
        trie = RadixTrie(block_size=4)
        tokens = list(range(8))  # 2 full blocks
        trie.insert(tokens, physical_block_ids=[10, 11], num_computed_tokens=8)

        match = trie.match(tokens)
        assert match.num_matched_tokens == 8
        assert match.physical_block_ids == [10, 11]

    def test_partial_match_stops_at_first_diverging_block(self):
        trie = RadixTrie(block_size=4)
        trie.insert(list(range(8)), physical_block_ids=[10, 11], num_computed_tokens=8)

        # Same first block, different second block.
        other = list(range(4)) + [99, 99, 99, 99]
        match = trie.match(other)
        assert match.num_matched_tokens == 4
        assert match.physical_block_ids == [10]

    def test_partial_block_at_the_end_never_matches(self):
        trie = RadixTrie(block_size=4)
        trie.insert(list(range(8)), physical_block_ids=[10, 11], num_computed_tokens=8)

        # 9 tokens: matches 2 full blocks, the 9th is a partial 3rd block --
        # not credited even though it agrees with what a longer cached
        # sequence would have held (nothing in this trie extends there).
        match = trie.match(list(range(9)))
        assert match.num_matched_tokens == 8

    def test_only_full_blocks_get_inserted(self):
        trie = RadixTrie(block_size=4)
        # 6 computed tokens -- only 1 full block's worth is cacheable.
        nodes = trie.insert(list(range(6)), physical_block_ids=[10, 11], num_computed_tokens=6)
        assert len(nodes) == 1
        match = trie.match(list(range(6)))
        assert match.num_matched_tokens == 4

    def test_hash_chaining_same_block_content_different_prefix_context(self):
        """Two sequences whose *second* block is byte-for-byte identical
        but whose first block differs must not cross-match on the second
        block -- content hashing is chained through the parent, not per
        block in isolation.
        """
        trie = RadixTrie(block_size=4)
        shared_tail = [7, 7, 7, 7]
        trie.insert(blocks([1, 1, 1, 1], shared_tail), [10, 11], num_computed_tokens=8)

        other = blocks([2, 2, 2, 2], shared_tail)
        match = trie.match(other)
        assert match.num_matched_tokens == 0  # first block differs -> no match at all

    def test_hash_block_is_stable_within_a_process(self):
        assert hash_block(None, (1, 2, 3)) == hash_block(None, (1, 2, 3))
        assert hash_block(None, (1, 2, 3)) != hash_block(None, (1, 2, 4))
        assert hash_block(None, (1, 2, 3)) != hash_block(123, (1, 2, 3))

    def test_match_is_side_effect_free(self):
        """Calling match() any number of times (e.g. can_allocate's peek
        ahead of allocate's own call) must never change ref_count or
        metrics -- only acquire() commits.
        """
        trie = RadixTrie(block_size=4)
        trie.insert(list(range(4)), [10], num_computed_tokens=4)
        for _ in range(5):
            trie.match(list(range(4)))
        node = next(iter(trie._by_hash.values()))
        assert node.ref_count == 1  # only insert()'s own hold, untouched by the 5 peeks
        stats = trie.stats()
        assert stats.num_lookups == 0
        assert stats.num_lookup_tokens == 0


class TestHashCollision:
    def test_colliding_hash_with_different_content_degrades_to_a_miss(self):
        # A fake hash function that always returns the same value --
        # forces every block to "collide" so match()/insert() must fall
        # back on comparing actual token content, not trust the hash alone.
        trie = RadixTrie(block_size=4, hash_fn=lambda parent, block: 0)
        trie.insert(list(range(4)), [10], num_computed_tokens=4)

        match = trie.match([99, 99, 99, 99])  # same (colliding) hash, different content
        assert match.num_matched_tokens == 0  # content check catches the collision

    def test_insert_with_colliding_hash_never_corrupts_the_existing_node(self):
        trie = RadixTrie(block_size=4, hash_fn=lambda parent, block: 0)
        trie.insert(list(range(4)), [10], num_computed_tokens=4)

        # Different content, same (forced) hash -- must not overwrite slot 10.
        registered = trie.insert([99, 99, 99, 99], [20], num_computed_tokens=4)
        assert registered == []  # nothing got registered -- collision stopped it immediately

        # The original content is still intact and correctly matchable.
        original_match = trie.match(list(range(4)))
        assert original_match.num_matched_tokens == 4
        assert original_match.physical_block_ids == [10]

        # The colliding content is still, correctly, never matchable.
        colliding_match = trie.match([99, 99, 99, 99])
        assert colliding_match.num_matched_tokens == 0


class TestRefCountingAndEviction:
    def test_two_requests_independently_inserting_the_same_content_both_get_a_hold(self):
        """Regression test: two requests admitted before either has
        registered anything (so neither matched the other at admission,
        e.g. concurrent admission in the same scheduler step) both
        computing and then insert()-ing the same prefix independently --
        not one matching()/acquire()-ing the other's already-cached
        result. Both must be able to release() without error: previously
        this raised (ref_count already 0) because only the first insert()
        actually bumped ref_count -- caught running measure_ttft.py for
        real on a GPU (see benchmarks/prefix_caching/README.md).
        """
        trie = RadixTrie(block_size=4)
        tokens = list(range(4))
        p1_nodes = trie.insert(tokens, [10], num_computed_tokens=4)
        p2_nodes = trie.insert(tokens, [10], num_computed_tokens=4)
        assert p1_nodes[0] is p2_nodes[0]
        assert p1_nodes[0].ref_count == 2
        trie.release(p1_nodes)  # must not raise
        trie.release(p2_nodes)  # must not raise
        assert p1_nodes[0].ref_count == 0

    def test_repeated_insert_by_the_same_request_does_not_inflate_ref_count(self):
        """Chunked prefill: the same request calls insert_computed_prefix
        again as more of its prompt gets computed. Passing back what it
        already owns (`known_prefix`, exactly what
        BlockManager.insert_computed_prefix does with
        request.cached_prefix_nodes) must leave already-held blocks'
        ref_count untouched -- only genuinely new blocks pick up a hold.
        """
        trie = RadixTrie(block_size=4)
        tokens = list(range(8))  # 2 blocks
        first = trie.insert(tokens, [10, 11], num_computed_tokens=4)  # only block 0 computed so far
        assert first[0].ref_count == 1
        second = trie.insert(tokens, [10, 11], num_computed_tokens=8, known_prefix=first)  # both blocks now
        assert second[0] is first[0]
        assert second[0].ref_count == 1  # unchanged -- this request already held it
        assert second[1].ref_count == 1  # brand new hold on the newly-computed block

    def test_insert_resumes_from_known_prefix_without_rewalking_it(self):
        """The performance half of known_prefix: a node it should skip
        entirely (not just leave ref_count alone on) must never even be
        looked at -- verified here by using a hash_fn that raises if
        called for a block index known_prefix already covers.
        """
        seen_indices = []

        def tracking_hash(parent, block):
            seen_indices.append(block[0] // 4)  # block_size=4, values 0,4,8,... per index
            return hash((parent, block))

        trie = RadixTrie(block_size=4, hash_fn=tracking_hash)
        tokens = list(range(0, 12, 1))  # 3 blocks: [0-3],[4-7],[8-11] -- values chosen so block[0]//4 == index
        first = trie.insert(tokens, [10, 11, 12], num_computed_tokens=4)  # walks index 0 only
        assert seen_indices == [0]
        seen_indices.clear()
        trie.insert(tokens, [10, 11, 12], num_computed_tokens=12, known_prefix=first)  # should walk 1,2 only
        assert seen_indices == [1, 2]  # index 0 never re-hashed

    def test_acquire_bumps_ref_count(self):
        trie = RadixTrie(block_size=4)
        inserted = trie.insert(list(range(4)), [10], num_computed_tokens=4)
        assert inserted[0].ref_count == 1  # the inserting request's own hold
        # A second request matches and acquires the same block.
        match = trie.match(list(range(4)))
        trie.acquire(match, num_lookup_tokens=4)
        node = trie._by_hash[match._nodes[0].block_hash]
        assert node.ref_count == 2

    def test_release_drops_ref_count(self):
        trie = RadixTrie(block_size=4)
        inserted = trie.insert(list(range(4)), [10], num_computed_tokens=4)
        match = trie.match(list(range(4)))
        trie.acquire(match, num_lookup_tokens=4)  # ref_count now 2 (inserter + this match)
        trie.release(match._nodes)
        assert match._nodes[0].ref_count == 1  # inserter's own hold remains
        trie.release(inserted)
        assert inserted[0].ref_count == 0

    def test_release_past_zero_raises(self):
        trie = RadixTrie(block_size=4)
        nodes = trie.insert(list(range(4)), [10], num_computed_tokens=4)  # ref_count starts at 1
        trie.release(nodes)  # -> 0
        try:
            trie.release(nodes)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_evict_one_lru_returns_none_when_nothing_evictable(self):
        trie = RadixTrie(block_size=4)
        assert trie.evict_one_lru() is None  # empty trie
        trie.insert(list(range(4)), [10], num_computed_tokens=4)  # ref_count 1, in use
        assert trie.evict_one_lru() is None

    def test_evict_one_lru_reclaims_unreferenced_leaf(self):
        trie = RadixTrie(block_size=4)
        nodes = trie.insert(list(range(4)), [10], num_computed_tokens=4)
        trie.release(nodes)
        assert trie.num_evictable_blocks == 1
        assert trie.evict_one_lru() == 10
        assert trie.num_evictable_blocks == 0
        # Gone for good -- a later identical insert starts fresh.
        assert trie.match(list(range(4))).num_matched_tokens == 0

    def test_eviction_is_oldest_used_first(self):
        trie = RadixTrie(block_size=4)
        a = trie.insert(list(range(0, 4)), [10], num_computed_tokens=4)
        b = trie.insert(list(range(100, 104)), [11], num_computed_tokens=4)  # unrelated content
        trie.release(a)
        trie.release(b)
        # `a` was inserted (and thus last-touched) before `b`.
        assert trie.evict_one_lru() == 10
        assert trie.evict_one_lru() == 11

    def test_matching_again_refreshes_lru_order(self):
        trie = RadixTrie(block_size=4)
        a = trie.insert(list(range(0, 4)), [10], num_computed_tokens=4)
        b = trie.insert(list(range(100, 104)), [11], num_computed_tokens=4)
        trie.release(a)
        trie.release(b)
        # Touch `a` again (a fresh match+acquire+release) after `b` was
        # already released -- `a` should now be the *more* recently used
        # one, so `b` gets evicted first.
        match_a = trie.match(list(range(0, 4)))
        trie.acquire(match_a, num_lookup_tokens=4)
        trie.release(match_a._nodes)
        assert trie.evict_one_lru() == 11
        assert trie.evict_one_lru() == 10

    def test_branching_ancestor_protected_by_a_different_live_child(self):
        """Two sequences share block 0 but diverge at block 1. Freeing one
        entirely must leave its own leaf evictable while the shared
        ancestor (block 0) stays protected -- not because *it* has a
        reference, but because its *other* child (the still-live branch)
        does. A plain straight chain doesn't exercise this: it never has a
        second branch to stay alive.
        """
        trie = RadixTrie(block_size=4)
        shared_head = [1, 1, 1, 1]
        p1_nodes = trie.insert(blocks(shared_head, [2, 2, 2, 2]), [10, 11], num_computed_tokens=8)
        p2_nodes = trie.insert(blocks(shared_head, [3, 3, 3, 3]), [10, 12], num_computed_tokens=8)
        # Both chains share node[0] (physical block 10) -- insert() found it
        # pre-existing for p2 (a different caller, no previously_owned
        # passed) and bumped its ref_count to 2 automatically, exactly as
        # if p2 had matched()/acquire()'d it instead of independently
        # computing the same content.
        assert p1_nodes[0] is p2_nodes[0]
        assert p1_nodes[0].ref_count == 2

        trie.release(p1_nodes)  # p1 fully freed: block 0 ref 1->0, block 1 (11) ref 1->0
        assert trie.num_evictable_blocks == 1  # only leaf 11 -- block 0 still has live child 12
        assert trie.evict_one_lru() == 11
        assert trie.num_evictable_blocks == 0  # block 0 still protected by p2's node (12)

        trie.release([p2_nodes[0]])  # drop p2's extra hold on block 0
        trie.release([p2_nodes[1]])  # p2 fully freed now
        # Block 12 is unreferenced now, but it's still *in the tree* (only
        # released, not evicted yet) -- so block 0 still has a child and
        # isn't a leaf. Only one block is evictable: 12 itself.
        assert trie.num_evictable_blocks == 1
        assert trie.evict_one_lru() == 12
        # With 12 actually gone, block 0 has no children left -- now it's
        # a leaf too, and it's unreferenced, so it becomes evictable.
        assert trie.num_evictable_blocks == 1
        assert trie.evict_one_lru() == 10


class TestStats:
    def test_hit_rate_zero_with_no_lookups(self):
        trie = RadixTrie(block_size=4)
        stats = trie.stats()
        assert stats.num_lookup_tokens == 0
        assert stats.hit_rate == 0.0
        assert stats.request_hit_rate == 0.0

    def test_hit_rate_reflects_hits_and_misses(self):
        trie = RadixTrie(block_size=4)
        trie.insert(list(range(8)), [10, 11], num_computed_tokens=8)

        hit = trie.match(list(range(8)))
        trie.acquire(hit, num_lookup_tokens=8)  # full hit: 8/8 tokens, 1/1 requests

        miss = trie.match([99, 99, 99, 99])
        trie.acquire(miss, num_lookup_tokens=4)  # total miss: 0/4 tokens

        stats = trie.stats()
        assert stats.num_lookup_tokens == 12
        assert stats.num_hit_tokens == 8
        assert stats.hit_rate == 8 / 12
        assert stats.num_lookups == 2
        assert stats.num_hits == 1
        assert stats.request_hit_rate == 0.5

    def test_match_alone_never_moves_the_metrics(self):
        trie = RadixTrie(block_size=4)
        trie.insert(list(range(4)), [10], num_computed_tokens=4)
        trie.match(list(range(4)))  # peek only, no acquire
        assert trie.stats().num_lookups == 0
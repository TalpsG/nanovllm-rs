use std::collections::{HashMap, HashSet};

use xxhash_rust::xxh64::Xxh64;

use crate::engine::sequence::Sequence;

pub fn compute_hash(token_ids: &[u32], prefix: Option<u64>) -> u64 {
    let u8_slice = unsafe {
        std::slice::from_raw_parts(
            token_ids.as_ptr() as *const u8,
            token_ids.len() * std::mem::size_of::<u32>(),
        )
    };
    let mut h = Xxh64::new(0);
    if let Some(p) = prefix {
        h.update(&p.to_le_bytes());
    }
    h.update(u8_slice);
    h.digest()
}
struct Block {
    pub block_id: usize,
    pub ref_count: usize,
    pub hash: Option<u64>,
    pub token_ids: Option<Vec<u32>>,
}
impl Block {
    pub fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: None,
            token_ids: None,
        }
    }
    pub fn update(&mut self, hash: u64, token_ids: &[u32]) {
        self.hash = Some(hash);
        self.token_ids = Some(token_ids.to_vec());
    }
    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids = None;
    }
}
struct BlockManager {
    pub block_size: usize,
    pub blocks: Vec<Block>,
    pub free_block_ids: Vec<usize>,
    pub used_block_ids: HashSet<usize>,
    hash_to_block_id: HashMap<u64, usize>,
}
impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(Block::new(i));
        }
        let free_block_ids = (0..num_blocks).collect();
        Self {
            block_size,
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            hash_to_block_id: HashMap::new(),
        }
    }
    fn _allocate_block(&mut self, block_id: usize) -> &mut Block {
        let block = &mut self.blocks[block_id];
        assert!(block.ref_count == 0);
        block.reset();
        // NOTE: this is O(n), consider using a different data structure if this becomes a bottleneck
        self.free_block_ids.retain(|&id| id != block_id);
        self.used_block_ids.insert(block_id);
        block
    }
    fn _deallocate_block(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        assert!(block.ref_count == 0);
        // Remove from used_block_ids and add to free_block_ids
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push(block_id);
    }
    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }
    pub fn allocate(&mut self, seq: &mut Sequence) {
        let mut h = None;
        let mut cache_miss = false;
        for i in 0..seq.num_blocks() {
            {
                let token_ids = seq.block(i);
                h = if token_ids.len() == self.block_size {
                    Some(compute_hash(token_ids, h))
                } else {
                    None
                };
            }
            // use None represents not found
            let block_id = h
                .map(|hash| self.hash_to_block_id.get(&hash).copied())
                .flatten();
            if block_id.is_none() {
                cache_miss = true;
            } else if let Some(id) = block_id
                && let Some(block_tokens) = self.blocks[id].token_ids.as_ref()
                && block_tokens.as_slice() != seq.block(i)
            {
                cache_miss = true;
            }
            let block;
            if cache_miss {
                let block_id = self.free_block_ids[0];
                block = self._allocate_block(block_id);
            } else {
                assert!(block_id.is_some());
                let id = block_id.unwrap();
                seq.num_cached_tokens += self.block_size;
                if self.used_block_ids.contains(&id) {
                    block = &mut self.blocks[id];
                    block.ref_count += 1;
                } else {
                    block = self._allocate_block(id);
                }
            }
            if let Some(block_id) = block_id {
                if let Some(hash) = h {
                    block.update(hash, seq.block(i));
                    self.hash_to_block_id.insert(hash, block_id);
                }

                seq.block_table.push(block_id);
            }
        }
    }
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[block_id];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self._deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }
    pub fn can_append(&self, seq: &Sequence) -> bool {
        let last_block_size = seq.len() % self.block_size;
        if last_block_size == 1 {
            self.free_block_ids.len() >= 1
        } else {
            true
        }
    }
    pub fn may_append(&mut self, seq: &mut Sequence) {
        let id = seq.block_table[seq.block_table.len() - 1];
        if seq.len() % self.block_size == 1 {
            let last_block = &self.blocks[id];
            assert!(last_block.hash.is_some());
            let block_id = self.free_block_ids[0];
            let _ = self._allocate_block(block_id);
            seq.block_table.push(block_id);
        } else if seq.len() % self.block_size == 0 {
            assert!(self.blocks[id].hash.is_none());
            let token_ids = seq.block(seq.num_blocks() - 1);
            let prefix = if seq.block_table.len() > 1 {
                self.blocks[seq.block_table[seq.block_table.len() - 2]].hash
            } else {
                None
            };
            let h = compute_hash(token_ids, prefix);
            let last_block = &mut self.blocks[id];
            last_block.update(h, token_ids);
            self.hash_to_block_id.insert(h, last_block.block_id);
        } else {
            assert!(self.blocks[id].hash.is_none());
        }
    }

    /*
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
     */
}

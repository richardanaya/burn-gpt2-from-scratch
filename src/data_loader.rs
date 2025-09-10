use anyhow::Result;
use burn::data::dataloader::Dataset as BurnDataset;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct TextDataset {
    pub data: Vec<i32>,
    pub context_length: usize,
}

impl TextDataset {
    pub fn from_file(path: &str, tokenizer: &Tokenizer, context_length: usize) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let encoding = tokenizer
            .encode(content.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let data = encoding.get_ids().iter().map(|&id| id as i32).collect();
        Ok(TextDataset {
            data,
            context_length,
        })
    }

    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.data.len() as f32 * train_ratio) as usize;
        let (train_data, val_data) = self.data.split_at(split_idx);

        (
            TextDataset {
                data: train_data.to_vec(),
                context_length: self.context_length,
            },
            TextDataset {
                data: val_data.to_vec(),
                context_length: self.context_length,
            },
        )
    }
}

// Dataset item is (input_tokens, target_tokens)
pub type DatasetItem = (Vec<i32>, Vec<i32>);

impl BurnDataset<DatasetItem> for TextDataset {
    fn get(&self, index: usize) -> Option<DatasetItem> {
        // Overlapping sliding window: each index is a potential starting position
        let start_idx = index;
        let end_idx = start_idx + self.context_length;

        if end_idx >= self.data.len() {
            return None;
        }

        let inputs = self.data[start_idx..end_idx].to_vec();
        let targets = self.data[start_idx + 1..end_idx + 1].to_vec();

        Some((inputs, targets))
    }

    fn len(&self) -> usize {
        if self.data.len() <= self.context_length {
            0
        } else {
            // Overlapping windows: every valid starting position
            self.data.len() - self.context_length
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,  // [batch_size, context_length]
    pub targets: Tensor<B, 2, Int>, // [batch_size, context_length]
}

pub struct TextBatcher {
    context_length: usize,
    current_position: AtomicUsize,
}

impl Clone for TextBatcher {
    fn clone(&self) -> Self {
        Self {
            context_length: self.context_length,
            current_position: AtomicUsize::new(self.current_position.load(Ordering::Relaxed)),
        }
    }
}

impl TextBatcher {
    pub fn new(context_length: usize) -> Self {
        Self {
            context_length,
            current_position: AtomicUsize::new(0),
        }
    }
}

impl<B: Backend> Batcher<B, DatasetItem, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<DatasetItem>, device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();
        let mut inputs_data = Vec::with_capacity(batch_size * self.context_length);
        let mut targets_data = Vec::with_capacity(batch_size * self.context_length);

        for (input_seq, target_seq) in items {
            inputs_data.extend(input_seq);
            targets_data.extend(target_seq);
        }

        let inputs = Tensor::<B, 1, Int>::from_data(TensorData::from(&inputs_data[..]), device)
            .reshape([batch_size, self.context_length]);

        let targets = Tensor::<B, 1, Int>::from_data(TensorData::from(&targets_data[..]), device)
            .reshape([batch_size, self.context_length]);

        TextBatch { inputs, targets }
    }
}

use burn::tensor::backend::AutodiffBackend;

pub fn create_data_loaders<B: AutodiffBackend>(
    data_path: &str,
    tokenizer: &Tokenizer,
    context_length: usize,
    train_batch_size: usize,
    eval_batch_size: usize,
    train_ratio: f32,
) -> Result<(
    impl Into<Arc<dyn DataLoader<B, TextBatch<B>>>>,
    impl Into<Arc<dyn DataLoader<B::InnerBackend, TextBatch<B::InnerBackend>>>>,
)> {
    // Create dataset using BPE tokenization
    let dataset = TextDataset::from_file(data_path, tokenizer, context_length)?;
    let (train_dataset, val_dataset) = dataset.split(train_ratio);

    println!("Dataset split:");
    println!("  Train: {} samples", train_dataset.len());
    println!("  Val: {} samples", val_dataset.len());

    let train_batcher = TextBatcher::new(context_length);
    let val_batcher = TextBatcher::new(context_length);

    let dataloader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(train_batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(val_batcher)
        .batch_size(eval_batch_size)
        .num_workers(1)
        .build(val_dataset);

    Ok((dataloader_train, dataloader_valid))
}

use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::{data::dataloader::Dataset as BurnDataset, record::CompactRecorder};
use burn::data::dataloader::batcher::Batcher;
use std::fs;
use anyhow::Result;
use burn::train::{LearnerBuilder, ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::nn::{
    Embedding, EmbeddingConfig, loss::CrossEntropyLoss,
};
use burn::tensor::{backend::Backend, Tensor, TensorData, Int};
use burn::config::Config;
use burn::module::Module;
use std::path::PathBuf;
use std::collections::HashMap;

type WgpuBackend = burn::backend::wgpu::Wgpu;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

// Character-level tokenizer similar to the Python tutorial
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let unique_chars: std::collections::HashSet<char> = chars.into_iter().collect();
        let mut chars: Vec<char> = unique_chars.into_iter().collect();
        chars.sort();
        
        let char_to_idx: HashMap<char, usize> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        
        let idx_to_char: HashMap<usize, char> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        
        let vocab_size = chars.len();
        
        Self {
            char_to_idx,
            idx_to_char,
            vocab_size,
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<i32> {
        text.chars()
            .map(|c| self.char_to_idx.get(&c).unwrap_or(&0).clone() as i32)
            .collect()
    }
    
    pub fn decode(&self, tokens: &[i32]) -> String {
        tokens
            .iter()
            .map(|&t| self.idx_to_char.get(&(t as usize)).unwrap_or(&' '))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct TextDataset {
    pub data: Vec<i32>,
    pub context_length: usize,
}

impl TextDataset {
    pub fn from_file(path: &str, tokenizer: &CharTokenizer, context_length: usize) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let data = tokenizer.encode(&content);
        
        Ok(TextDataset { data, context_length })
    }
    
    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.data.len() as f32 * train_ratio) as usize;
        let (train_data, val_data) = self.data.split_at(split_idx);
        
        (
            TextDataset { data: train_data.to_vec(), context_length: self.context_length },
            TextDataset { data: val_data.to_vec(), context_length: self.context_length }
        )
    }
}

// Dataset item is (input_tokens, target_tokens)
pub type DatasetItem = (Vec<i32>, Vec<i32>);

impl BurnDataset<DatasetItem> for TextDataset {
    fn get(&self, index: usize) -> Option<DatasetItem> {
        // Non-overlapping chunks: each index represents a chunk of context_length
        let start_idx = index * self.context_length;
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
            // Non-overlapping chunks like Python article
            self.data.len() / self.context_length
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,   // [batch_size, context_length]
    pub targets: Tensor<B, 2, Int>,  // [batch_size, context_length]
}

#[derive(Clone)]
pub struct TextBatcher {
    context_length: usize,
}

impl TextBatcher {
    pub fn new(context_length: usize) -> Self {
        Self { context_length }
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
        
        let inputs = Tensor::<B, 1, Int>::from_data(
            TensorData::from(&inputs_data[..]),
            device
        ).reshape([batch_size, self.context_length]);
        
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::from(&targets_data[..]),
            device
        ).reshape([batch_size, self.context_length]);
        
        TextBatch { inputs, targets }
    }
}

#[derive(Config, Debug)]
pub struct Gpt2Config {
    vocab_size: usize,
    d_model: usize,
}

impl Gpt2Config {
    /// Initialize the model from config
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2Model<B> {
        Gpt2Model::new(self.clone(), device)
    }
}

#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    wte: Embedding<B>, // word token embeddings
    vocab_size: usize,
}

impl<B: Backend> Gpt2Model<B> {
    /// Creates a new GPT-2 model with the given configuration  
    pub fn new(config: Gpt2Config, device: &B::Device) -> Self {
        Self {
            wte: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            vocab_size: config.vocab_size,
        }
    }
    
    pub fn forward(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Get embeddings: [batch_size, seq_len, d_model]
        let logits = self.wte.forward(inputs);
        logits
    }
    
    /// Generate text similar to the Python article's generate method
    pub fn generate(&self, inputs: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut current_inputs = inputs;
        
        for _ in 0..max_new_tokens {
            // Forward pass to get logits
            let logits = self.forward(current_inputs.clone()); // [batch_size, seq_len, vocab_size]
            
            // Get logits for the last token in sequence for each batch
            let dims = logits.dims();
            let last_logits: Tensor<B, 2> = logits.slice([0..dims[0], (dims[1] - 1)..dims[1], 0..dims[2]])
                .squeeze::<2>(1); // [batch_size, vocab_size]
            
            // Apply softmax to get probabilities
            let probs = burn::tensor::activation::softmax(last_logits, 1);
            
            // Sample from the probability distribution
            let next_token: Tensor<B, 2, Int> = probs.argmax(1).unsqueeze::<2>(); // Simple argmax sampling for now
            
            // Concatenate the new token to the sequence
            current_inputs = Tensor::cat(vec![current_inputs, next_token], 1);
        }
        
        current_inputs
    }
    
    /// Shared computation for both training and validation
    pub fn compute_loss(&self, batch: &TextBatch<B>) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
        // Forward pass
        let logits = self.forward(batch.inputs.clone()); // [batch_size, seq_len, vocab_size]
        
        // Reshape for loss calculation
        let [batch_size, seq_len, _] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, self.vocab_size]);
        let targets_flat = batch.targets.clone().reshape([batch_size * seq_len]);
        
        // Use cross entropy loss for classification
        let device = &logits_flat.device();
        let loss_tensor = CrossEntropyLoss::new(None, device)
            .forward(logits_flat.clone(), targets_flat.clone());
        
        (loss_tensor, logits_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let (loss_tensor, logits_flat, targets_flat) = self.compute_loss(&batch);
        
        let output = ClassificationOutput {
            loss: loss_tensor.clone(),
            output: logits_flat,
            targets: targets_flat,
        };
        
        let gradients = loss_tensor.backward();
        TrainOutput::new(self, gradients, output)
    }
}

impl<B: Backend> ValidStep<TextBatch<B>, ClassificationOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        let (loss_tensor, logits_flat, targets_flat) = self.compute_loss(&batch);
        
        ClassificationOutput {
            loss: loss_tensor,
            output: logits_flat,
            targets: targets_flat,
        }
    }
}

fn main() -> Result<()> {
    // Read text data and create character tokenizer
    let text = fs::read_to_string("data.txt")?;
    let tokenizer = CharTokenizer::from_text(&text);
    
    println!("Character tokenizer created:");
    println!("  Vocabulary size: {}", tokenizer.vocab_size);
    
    let context_length: usize = 256;
    
    // Create dataset with character tokenization
    let dataset = TextDataset::from_file("data.txt", &tokenizer, context_length)?;
    let (train_dataset, val_dataset) = dataset.split(0.8);
    
    println!("Dataset split:");
    println!("  Train: {} samples", train_dataset.len());
    println!("  Val: {} samples", val_dataset.len());
    
    let train_batcher = TextBatcher::new(context_length);
    let val_batcher = TextBatcher::new(context_length);
    
    let device = burn::backend::wgpu::WgpuDevice::default();

    let dataloader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(16)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(val_batcher)
        .batch_size(8)
        .num_workers(1)
        .build(val_dataset);

    WgpuAutodiffBackend::seed(42);

    let output_dir = PathBuf::from("training_output".to_string());

    let learner_builder =
        LearnerBuilder::new(output_dir.clone()).with_file_checkpointer(CompactRecorder::default());


    let config = Gpt2Config {
        vocab_size: tokenizer.vocab_size,
        d_model: tokenizer.vocab_size,
    };

    let learner = learner_builder
        .devices(vec![device.clone()])
        .num_epochs(5000)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .summary()
        .build(
            config.init::<WgpuAutodiffBackend>(&device),
            AdamConfig::new().init(),
            1e-3,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    println!("\n✅ Training Complete!");

    // Save model
    model_trained
        .clone()
        .save_file(
            format!("{}/model", output_dir.display()),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");
    println!("💾 Model saved to: {}/model", output_dir.display());
    
    // Test generation like the article
    println!("\n🎵 Generating text sample:");
    let test_input = "Hello";
    let test_tokens = tokenizer.encode(test_input);
    println!("Input: '{}'", test_input);
    
    // Create tensor for generation - need autodiff backend for trained model
    let input_tensor = Tensor::<WgpuAutodiffBackend, 1, Int>::from_data(
        TensorData::from(&test_tokens[..]),
        &device
    ).reshape([1, test_tokens.len()]); // batch_size=1, seq_len=input_len
    
    // Generate new tokens
    let max_new_tokens = 100;
    println!("Generating {} new tokens...\n", max_new_tokens);
    
    let generated_tensor = model_trained.generate(input_tensor, max_new_tokens);
    let generated_data = generated_tensor.flatten::<1>(0, 1).into_data().to_vec::<i32>().unwrap();
    let generated_text = tokenizer.decode(&generated_data);
    
    println!("Generated text:");
    println!("{}", generated_text);
    
    Ok(())
}

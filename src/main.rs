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
    Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig, loss::CrossEntropyLoss,
};
use burn::tensor::{backend::Backend, Tensor, TensorData, Int};
use burn::config::Config;
use burn::module::Module;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

type WgpuBackend = burn::backend::wgpu::Wgpu;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

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

#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    pe: Tensor<B, 3>, // [1, context_length, d_model]
    context_length: usize,
}

impl<B: Backend> PositionalEncoding<B> {
    pub fn new(context_length: usize, d_model: usize, device: &B::Device) -> Self {
        let ln_10000 = 10000f32.ln();
        let mut data = vec![0f32; context_length * d_model];
        for pos in 0..context_length {
            for i in (0..d_model).step_by(2) {
                let div = (-(ln_10000) / d_model as f32) * (i as f32);
                let weight = div.exp();
                let angle = (pos as f32) * weight;
                let base = pos * d_model + i;
                data[base] = angle.sin();
                if i + 1 < d_model {
                    data[base + 1] = angle.cos();
                }
            }
        }
        let pe = Tensor::<B, 1>::from_data(TensorData::from(&data[..]), device)
            .reshape([1, context_length, d_model]);
        Self { pe, context_length }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let seq_len = dims[1].min(self.context_length);
        let pe_slice = self.pe.clone().slice([0..1, 0..seq_len, 0..dims[2]]);
        x + pe_slice
    }
}

#[derive(Config, Debug)]
pub struct Gpt2Config {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    context_length: usize,
}

impl Gpt2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2Model<B> {
        Gpt2Model::new(self.clone(), device)
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            o_proj: LinearConfig::new(d_model, d_model).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            n_heads,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, d] = x.dims();
        let head_dim = d / self.n_heads;
        let xf = x.clone().reshape([b * l, d]);
        let q = self.q_proj.forward(xf.clone()).reshape([b, l, self.n_heads, head_dim]).swap_dims(1, 2); // [B, H, L, Dh]
        let k = self.k_proj.forward(xf.clone()).reshape([b, l, self.n_heads, head_dim]).swap_dims(1, 2); // [B, H, L, Dh]
        let v = self.v_proj.forward(xf).reshape([b, l, self.n_heads, head_dim]).swap_dims(1, 2); // [B, H, L, Dh]

        // Attention scores per head: [B, H, L, L]
        let k_t = k.swap_dims(2, 3); // [B, H, Dh, L]
        let mut scores = q.matmul(k_t); // [B, H, L, L]
        // Scale by sqrt(head_dim)
        let scale = (head_dim as f32).sqrt();
        scores = scores / scale;

        // Build causal mask (upper triangle) [L, L] and broadcast to [1,1,L,L]
        let mut mask = vec![0f32; l * l];
        for i in 0..l {
            for j in (i + 1)..l {
                mask[i * l + j] = 1.0;
            }
        }
        let device = &scores.device();
        let mask_tensor = Tensor::<B, 1>::from_data(TensorData::from(&mask[..]), device)
            .reshape([l, l])
            .unsqueeze::<3>() // [1, L, L]
            .unsqueeze::<4>(); // [1, 1, L, L]
        let scores = scores + mask_tensor * (-1e9f32);

        // Softmax over last dimension
        let attn = burn::tensor::activation::softmax(scores, 3);
        let attn = self.dropout.forward(attn);

        // Weighted sum: [B, H, L, Dh]
        let context = attn.matmul(v); // [B, H, L, Dh]
        // Merge heads: [B, L, D]
        let context = context.swap_dims(1, 2).reshape([b, l, d]);

        // Output projection
        let out = self.o_proj.forward(context.clone().reshape([b * l, d])).reshape([b, l, d]);
        out
    }
}

#[derive(Module, Debug)]
pub struct Gpt2Block<B: Backend> {
    att: SelfAttention<B>,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    dropout: Dropout,
    fcn1: Linear<B>,
    fcn2: Linear<B>,
}

impl<B: Backend> Gpt2Block<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        Self {
            att: SelfAttention::new(d_model, n_heads, device),
            ln1: LayerNormConfig::new(d_model).init(device),
            ln2: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            fcn1: LinearConfig::new(d_model, 4 * d_model).init(device),
            fcn2: LinearConfig::new(4 * d_model, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // att -> residual -> ln1
        let att_out = self.att.forward(x.clone());
        let adn_logits = self.ln1.forward(x + att_out);
        // dropout then MLP
        let x_drop = self.dropout.forward(adn_logits.clone());
        let [b, l, d] = x_drop.dims();
        let x2 = x_drop.clone().reshape([b * l, d]);
        let x2 = self.fcn1.forward(x2);
        let x2 = burn::tensor::activation::gelu(x2);
        let x2 = self.fcn2.forward(x2);
        let x2 = x2.reshape([b, l, d]);
        // second residual uses adn_logits like Python snippet
        self.ln2.forward(x2 + adn_logits)
    }
}

#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    wte: Embedding<B>,
    pos_enc: PositionalEncoding<B>,
    blocks: Vec<Gpt2Block<B>>, // n_layers blocks
    lm_head: Linear<B>,
    vocab_size: usize,
    context_length: usize,
}

impl<B: Backend> Gpt2Model<B> {
    pub fn new(config: Gpt2Config, device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(Gpt2Block::new(config.d_model, config.n_heads, device));
        }
        Self {
            wte: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            pos_enc: PositionalEncoding::new(config.context_length, config.d_model, device),
            blocks,
            lm_head: LinearConfig::new(config.d_model, config.vocab_size).init(device),
            vocab_size: config.vocab_size,
            context_length: config.context_length,
        }
    }
    
    pub fn forward(&self, inputs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.wte.forward(inputs);
        x = self.pos_enc.forward(x);
        for block in self.blocks.iter() {
            x = block.forward(x);
        }
        let [b, l, d] = x.dims();
        let logits = self
            .lm_head
            .forward(x.reshape([b * l, d]))
            .reshape([b, l, self.vocab_size]);
        logits
    }
    
    pub fn generate(&self, inputs: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut current_inputs = inputs.clone();
        let mut output = inputs;
        let mut rng = thread_rng();
        for _ in 0..max_new_tokens {
            let dims_in = current_inputs.dims();
            let seq_len = dims_in[1];
            let inp_for_model = if seq_len > self.context_length {
                current_inputs.clone().slice([
                    0..dims_in[0],
                    (seq_len - self.context_length)..seq_len,
                ])
            } else {
                current_inputs.clone()
            };

            let logits = self.forward(inp_for_model);
            let dims = logits.dims();
            let last_logits: Tensor<B, 2> = logits
                .slice([0..dims[0], (dims[1] - 1)..dims[1], 0..dims[2]])
                .squeeze::<2>(1);
            let probs = burn::tensor::activation::softmax(last_logits, 1);
            // Multinomial sampling (per-batch, single token)
            let probs_vec = probs.clone().into_data().to_vec::<f32>().unwrap();
            let dist = WeightedIndex::new(&probs_vec).unwrap();
            let idx = dist.sample(&mut rng) as i32; // single-batch assumption in demo
            let next_token: Tensor<B, 2, Int> = Tensor::<B, 1, Int>::from_data(
                TensorData::from(&[idx][..]),
                &probs.device(),
            ).reshape([1, 1]);

            output = Tensor::cat(vec![output, next_token.clone()], 1);
            current_inputs = Tensor::cat(vec![current_inputs, next_token], 1);
        }
        output
    }
    
    pub fn compute_loss(&self, batch: &TextBatch<B>) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
        let logits = self.forward(batch.inputs.clone());
        let [batch_size, seq_len, _] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, self.vocab_size]);
        let targets_flat = batch.targets.clone().reshape([batch_size * seq_len]);
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
    // Load GPT-2 BPE tokenizer from tokenizer.json (HuggingFace tokenizers)
    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    println!("Tokenizer loaded:");
    println!("  Vocabulary size: {}", tokenizer.get_vocab_size(true));
    
    let context_length: usize = 512; // increased context length for BPE tokens
    
    // Create dataset using BPE tokenization
    let dataset = TextDataset::from_file("data.txt", &tokenizer, context_length)?;
    let (train_dataset, val_dataset) = dataset.split(0.7);
    
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
        vocab_size: tokenizer.get_vocab_size(true),
        d_model: 768,
        n_heads: 4,
        n_layers: 8,
        context_length,
    };

    let learner = learner_builder
        .devices(vec![device.clone()])
        .num_epochs(10)
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
    let test_encoding = tokenizer
        .encode(test_input, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let test_tokens: Vec<i32> = test_encoding.get_ids().iter().map(|&id| id as i32).collect();
    println!("Input: '{}'", test_input);
    
    // Create tensor for generation - need autodiff backend for trained model
    let input_tensor = Tensor::<WgpuAutodiffBackend, 1, Int>::from_data(
        TensorData::from(&test_tokens[..]),
        &device
    ).reshape([1, test_tokens.len()]);
    
    let max_new_tokens = 100;
    println!("Generating {} new tokens...\n", max_new_tokens);
    
    let generated_tensor = model_trained.generate(input_tensor, max_new_tokens);
    let generated_data = generated_tensor.flatten::<1>(0, 1).into_data().to_vec::<i32>().unwrap();
    let generated_ids_u32: Vec<u32> = generated_data.iter().map(|&id| id as u32).collect();
    let generated_text = tokenizer
        .decode(&generated_ids_u32, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    
    println!("Generated text:");
    println!("{}", generated_text);
    
    Ok(())
}

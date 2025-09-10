mod data_loader;

use anyhow::Result;
use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig, loss::CrossEntropyLoss,
};
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use data_loader::{TextBatch, create_data_loaders};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::path::PathBuf;
use tokenizers::Tokenizer;

type WgpuBackend = burn::backend::wgpu::Wgpu;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

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
        let pe = Tensor::<B, 1>::from_data(TensorData::from(&data[..]), device).reshape([
            1,
            context_length,
            d_model,
        ]);
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
        assert!(
            d_model % n_heads == 0,
            "d_model must be divisible by n_heads"
        );
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
        let q = self
            .q_proj
            .forward(xf.clone())
            .reshape([b, l, self.n_heads, head_dim])
            .swap_dims(1, 2); // [B, H, L, Dh]
        let k = self
            .k_proj
            .forward(xf.clone())
            .reshape([b, l, self.n_heads, head_dim])
            .swap_dims(1, 2); // [B, H, L, Dh]
        let v = self
            .v_proj
            .forward(xf)
            .reshape([b, l, self.n_heads, head_dim])
            .swap_dims(1, 2); // [B, H, L, Dh]

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
        let out = self
            .o_proj
            .forward(context.clone().reshape([b * l, d]))
            .reshape([b, l, d]);
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
        // Pre-norm architecture: LayerNorm before attention
        let ln1_out = self.ln1.forward(x.clone());
        let att_out = self.att.forward(ln1_out);
        let after_att = x + att_out; // residual connection

        // Pre-norm architecture: LayerNorm before FFN
        let ln2_out = self.ln2.forward(after_att.clone());
        let x_drop = self.dropout.forward(ln2_out);
        let [b, l, d] = x_drop.dims();
        let x2 = x_drop.reshape([b * l, d]);
        let x2 = self.fcn1.forward(x2);
        let x2 = burn::tensor::activation::gelu(x2);
        let x2 = self.fcn2.forward(x2);
        let x2 = x2.reshape([b, l, d]);

        // Second residual connection
        after_att + x2
    }
}

#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    wte: Embedding<B>,
    pos_enc: PositionalEncoding<B>,
    blocks: Vec<Gpt2Block<B>>, // n_layers blocks
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
        // Parameter sharing: use embedding weights for output projection
        let wte_weight = self.wte.weight.val(); // [vocab_size, d_model]
        let x_flat = x.reshape([b * l, d]); // [b*l, d_model]
        let logits = x_flat.matmul(wte_weight.transpose()); // [b*l, vocab_size]
        logits.reshape([b, l, self.vocab_size])
    }

    pub fn generate(&self, inputs: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut current_inputs = inputs.clone();
        let mut output = inputs;
        let mut rng = thread_rng();
        for _ in 0..max_new_tokens {
            let dims_in = current_inputs.dims();
            let seq_len = dims_in[1];
            let inp_for_model = if seq_len > self.context_length {
                current_inputs
                    .clone()
                    .slice([0..dims_in[0], (seq_len - self.context_length)..seq_len])
            } else {
                current_inputs.clone()
            };

            let logits = self.forward(inp_for_model);
            let dims = logits.dims();
            let last_logits: Tensor<B, 2> = logits
                .slice([0..dims[0], (dims[1] - 1)..dims[1], 0..dims[2]])
                .squeeze::<2>(1);
            let probs = burn::tensor::activation::softmax(last_logits, 1);
            // Multinomial sampling matching Python's torch.multinomial
            let probs_vec = probs.clone().into_data().to_vec::<f32>().unwrap();
            let batch_size = probs.dims()[0];

            // Sample for each batch (though we assume single batch here)
            let mut next_tokens = Vec::new();
            for b in 0..batch_size {
                let batch_probs = &probs_vec[b * self.vocab_size..(b + 1) * self.vocab_size];
                let dist = WeightedIndex::new(batch_probs).unwrap();
                let idx = dist.sample(&mut rng) as i32;
                next_tokens.push(idx);
            }

            let next_token: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(TensorData::from(&next_tokens[..]), &probs.device())
                    .reshape([batch_size, 1]);

            output = Tensor::cat(vec![output, next_token.clone()], 1);
            current_inputs = Tensor::cat(vec![current_inputs, next_token], 1);
        }
        output
    }

    pub fn compute_loss(
        &self,
        batch: &TextBatch<B>,
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
        let logits = self.forward(batch.inputs.clone());
        let [batch_size, seq_len, _] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, self.vocab_size]);
        let targets_flat = batch.targets.clone().reshape([batch_size * seq_len]);
        let device = &logits_flat.device();
        let loss_tensor =
            CrossEntropyLoss::new(None, device).forward(logits_flat.clone(), targets_flat.clone());
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
    // This should be compatible with tiktoken GPT-2 encoding
    let tokenizer =
        Tokenizer::from_file("tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Tokenizer loaded:");
    println!("  Vocabulary size: {} (matches tiktoken gpt2)", vocab_size);

    let context_length: usize = 48; // increased context length for BPE tokens
    let train_batch_size = 16; // Match Python batch sizes exactly
    let eval_batch_size = 8;

    // Create data loaders
    let (dataloader_train, dataloader_valid) = create_data_loaders::<WgpuAutodiffBackend>(
        "data.txt",
        &tokenizer,
        context_length,
        train_batch_size,
        eval_batch_size,
        0.7,
    )?;
    let dataloader_train = dataloader_train.into();
    let dataloader_valid = dataloader_valid.into();

    let device = burn::backend::wgpu::WgpuDevice::default();

    WgpuAutodiffBackend::seed(42);

    let output_dir = PathBuf::from("training_output".to_string());

    let learner_builder =
        LearnerBuilder::new(output_dir.clone()).with_file_checkpointer(CompactRecorder::default());

    // Match Python hyperparameters exactly
    let config = Gpt2Config {
        vocab_size,     // Use the tokenizer vocab_size (should match tiktoken)
        d_model: 256,   // matches Python d_model
        n_heads: 1,     // matches Python n_heads
        n_layers: 1,    // matches Python n_layers
        context_length, // matches Python context_length (512)
    };

    // Match Python training parameters
    let lr = 1e-3;
    let epochs = 2000;

    let learner = learner_builder
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .summary()
        .build(
            config.init::<WgpuAutodiffBackend>(&device),
            AdamWConfig::new()
                .with_weight_decay(0.1)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .init(),
            lr,
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

    // Test generation like the Python article
    println!("\n🎵 Generating text sample:");
    let test_input = "Love "; // Match Python test input
    let test_encoding = tokenizer
        .encode(test_input, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let test_tokens: Vec<i32> = test_encoding
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    println!("Input: '{}'", test_input);

    // Create tensor for generation - need autodiff backend for trained model
    let input_tensor = Tensor::<WgpuAutodiffBackend, 1, Int>::from_data(
        TensorData::from(&test_tokens[..]),
        &device,
    )
    .reshape([1, test_tokens.len()]);

    let max_new_tokens = 500; // Match Python max_new_tokens
    println!("Generating {} new tokens...\n", max_new_tokens);

    let generated_tensor = model_trained.generate(input_tensor, max_new_tokens);
    let generated_data = generated_tensor
        .flatten::<1>(0, 1)
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let generated_ids_u32: Vec<u32> = generated_data.iter().map(|&id| id as u32).collect();
    let generated_text = tokenizer
        .decode(&generated_ids_u32, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    println!("Generated text:");
    println!("{}", generated_text);

    Ok(())
}

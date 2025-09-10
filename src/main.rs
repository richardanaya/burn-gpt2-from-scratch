use tokenizers::tokenizer::Tokenizer;
use burn::data::dataloader::Dataset as BurnDataset;
use std::fs;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct TextDataset {
    pub lines: Vec<String>,
}

impl TextDataset {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();
        
        println!("Loaded {} lines from {}", lines.len(), path);
        Ok(TextDataset { lines })
    }
}

impl BurnDataset<String> for TextDataset {
    fn get(&self, index: usize) -> Option<String> {
        self.lines.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.lines.len()
    }
}

fn main() -> Result<()> {
    // Load the tokenizer from tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Tokenize "hello world"
    let encoding = tokenizer.encode("hello world", false)
        .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
    println!("Tokens: {:?}", encoding.get_tokens());
    println!("Token IDs: {:?}", encoding.get_ids());
    
    // Load the text dataset
    let dataset = TextDataset::from_file("data.txt")?;
    println!("Dataset has {} lines", dataset.len());
    
    // Show first few lines tokenized
    for i in 0..3.min(dataset.len()) {
        if let Some(line) = dataset.get(i) {
            let encoding = tokenizer.encode(line.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Failed to encode line: {}", e))?;
            println!("Line {}: '{}' -> {:?}", i, line, encoding.get_ids());
        }
    }
    
    Ok(())
}

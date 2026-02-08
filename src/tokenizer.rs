use anyhow::Result;
use crate::constants;

#[derive(Debug, Clone)]
pub struct Tokenizer {}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer {}
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u8>> {
        let mut tokens = Vec::with_capacity(text.len());
        let mut skipped = 0;
        for c in text.chars() {
            // Only ASCII characters are supported in this simple tokenizer
            if !c.is_ascii() {
                skipped += 1;
                continue;
            }
            tokens.push(c as u8);
        }
        if skipped > 0 {
            eprintln!("warning: skipped {} tokens", skipped)
        }
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &[u8]) -> Result<String> {
        let mut text = String::with_capacity(tokens.len());
        for &token in tokens {
            match token {
                0..=127 => {
                    text.push(token as u8 as char);
                }
                constants::PAD_TOKEN => {}
                constants::EOS_TOKEN => {
                    break;
                }
                _ => return Err(anyhow::anyhow!("Invalid token found: {}", token)),
            }
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{EOS_TOKEN, PAD_TOKEN};
    
    #[test]
    fn test_tokenizer() -> Result<()> {
        let tokenizer = Tokenizer::new();
        let text = "Hello, World!";
        let tokens = tokenizer.encode(text)?;
        let decoded_text = tokenizer.decode(&tokens)?;
        assert_eq!(text, decoded_text);
        Ok(())
    }

    #[test]
    fn test_special_tokens() -> Result<()> {
        let tokenizer = Tokenizer::new();
        let text = "def foo():";
        let mut tokens = tokenizer.encode(text)?;
        tokens.push(EOS_TOKEN);
        tokens.push(PAD_TOKEN);
        tokens.push(PAD_TOKEN);
        let decoded_text = tokenizer.decode(&tokens)?;
        assert_eq!(text, decoded_text);
        Ok(())
    }
}

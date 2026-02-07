use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Tokenizer {}

pub const PAD_TOKEN: u8 = 128;
pub const EOS_TOKEN: u8 = 129;

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer {}
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u8>> {
        let mut tokens = Vec::with_capacity(text.len());
        for c in text.chars() {
            // Only ASCII characters are supported in this simple tokenizer
            if !c.is_ascii() {
                return Err(anyhow::anyhow!("Non-ASCII character found: {}", c));
            }
            tokens.push(c as u8);
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
                PAD_TOKEN => {}
                EOS_TOKEN => {
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

//! Cryptographic utilities for adapter signing and verification.
//!
//! - **ed25519**: Detached signatures for adapter weight files.
//! - **SHA-256**: Content-addressable checksums for reproducibility.

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use pyo3::prelude::*;
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Compute SHA-256 hex digest of a file.
#[pyfunction]
pub fn sha256_file(path: &str) -> PyResult<String> {
    let mut file = File::open(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8 MB chunks

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA-256 hex digest of raw bytes.
#[pyfunction]
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Generate a new ed25519 keypair and return (private_key_hex, public_key_hex)
#[pyfunction]
pub fn generate_ed25519_keypair() -> (String, String) {
    let mut csprng = OsRng;
    let signing_key: SigningKey = SigningKey::generate(&mut csprng);
    let verifying_key: VerifyingKey = signing_key.verifying_key();

    let priv_hex = hex::encode(signing_key.to_bytes());
    let pub_hex = hex::encode(verifying_key.to_bytes());

    (priv_hex, pub_hex)
}

/// Sign a message using an ed25519 private key (hex encoded)
#[pyfunction]
pub fn sign_message(private_key_hex: &str, message: &[u8]) -> PyResult<String> {
    let priv_bytes = hex::decode(private_key_hex)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid hex: {}", e)))?;

    let signing_key = SigningKey::from_bytes(
        priv_bytes
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid private key length"))?,
    );

    let signature = signing_key.sign(message);
    Ok(hex::encode(signature.to_bytes()))
}

/// Verify a message signature using an ed25519 public key (hex encoded)
#[pyfunction]
pub fn verify_message(public_key_hex: &str, message: &[u8], signature_hex: &str) -> PyResult<bool> {
    let pub_bytes = hex::decode(public_key_hex)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid hex: {}", e)))?;

    let verifying_key = VerifyingKey::from_bytes(
        pub_bytes
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid public key length"))?,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid public key: {}", e)))?;

    let sig_bytes = hex::decode(signature_hex)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid hex: {}", e)))?;

    let signature = ed25519_dalek::Signature::from_bytes(
        sig_bytes
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid signature length"))?,
    );

    Ok(verifying_key.verify_strict(message, &signature).is_ok())
}

/// Register this module's functions with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sha256_file, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ed25519_keypair, m)?)?;
    m.add_function(wrap_pyfunction!(sign_message, m)?)?;
    m.add_function(wrap_pyfunction!(verify_message, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_bytes() {
        let digest = sha256_bytes(b"hello world");
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}

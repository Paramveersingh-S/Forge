import json
import os
from pathlib import Path

from .bom import MLBOM


def sign_bom(bom: MLBOM, private_key_path: str | Path | None = None) -> str:
    """Cryptographically sign an ML-BOM using ed25519.

    Args:
        bom: The MLBOM object to sign.
        private_key_path: Path to the private key. If None, uses ~/.forge/keys/id_ed25519

    Returns:
        The hex-encoded signature.
    """
    if private_key_path is None:
        private_key_path = Path.home() / ".forge" / "keys" / "id_ed25519"

    private_key_path = Path(private_key_path)

    # Check if key exists, generate if not
    if not private_key_path.exists():
        _generate_keypair(private_key_path)

    # Serialize BOM deterministically
    bom_json = bom.to_json().encode("utf-8")

    try:
        import forge_core.crypto

        if hasattr(forge_core.crypto, "sign_message"):
            with open(private_key_path) as f:
                priv_hex = f.read().strip()
            return forge_core.crypto.sign_message(priv_hex, bom_json)
    except ImportError:
        pass

    # Fallback/Mock signature for development phase
    # In production, this must use a real ed25519 signer
    import hashlib
    import hmac

    with open(private_key_path, "rb") as f:
        key_bytes = f.read()

    # Create an HMAC SHA-256 as a stand-in for Ed25519 during early testing
    h = hmac.new(key_bytes, bom_json, hashlib.sha256)
    return f"mock_sig_{h.hexdigest()}"


def verify_signature(bom: MLBOM, signature: str, public_key_path: str | Path) -> bool:
    """Verify the signature of an ML-BOM."""
    public_key_path = Path(public_key_path)
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")

    bom_json = bom.to_json().encode("utf-8")

    if signature.startswith("mock_sig_"):
        # Very insecure mock verification (uses private key as public key)
        # Only for development

        # In our mock, the public key is just random bytes, so we need to
        # find the corresponding private key to verify the HMAC.
        private_key = public_key_path.with_suffix("")
        if not private_key.exists():
            return False

        with open(private_key, "rb") as f:
            key_bytes = f.read()

        import hashlib
        import hmac

        h = hmac.new(key_bytes, bom_json, hashlib.sha256)
        expected = f"mock_sig_{h.hexdigest()}"
        return hmac.compare_digest(signature, expected)

    try:
        import forge_core.crypto

        if hasattr(forge_core.crypto, "verify_message"):
            with open(public_key_path) as f:
                pub_hex = f.read().strip()
            return forge_core.crypto.verify_message(pub_hex, bom_json, signature)
    except ImportError:
        pass

    raise NotImplementedError("Ed25519 verification requires forge_core.crypto")


def _generate_keypair(private_key_path: Path) -> None:
    """Generate a new Ed25519 keypair."""
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path = private_key_path.with_suffix(".pub")

    try:
        import forge_core.crypto

        if hasattr(forge_core.crypto, "generate_ed25519_keypair"):
            priv_hex, pub_hex = forge_core.crypto.generate_ed25519_keypair()
            private_key_path.write_text(priv_hex)
            public_key_path.write_text(pub_hex)
            # Secure the private key
            os.chmod(private_key_path, 0o600)
            return
    except ImportError:
        pass

    # Mock generation
    import secrets

    priv = secrets.token_bytes(32)
    pub = secrets.token_bytes(32)  # fake public key

    private_key_path.write_bytes(priv)
    public_key_path.write_bytes(pub)

    # Try to set permissions securely
    try:
        os.chmod(private_key_path, 0o600)
    except Exception:
        pass

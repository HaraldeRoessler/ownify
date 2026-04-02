"""
ownify — MolTrust integration for decentralized agent identity and trust.

Built on the MolTrust Protocol (https://moltrust.ch/) by MoltyCel
(https://github.com/MoltyCel). All credit to MoltyCel for the protocol.

Provides:
- Agent identity via W3C DIDs (Ed25519 keypair)
- Trust verification of external agents before escalation
- Output provenance via Interaction Proof Records (IPR)
- Local key caching for offline verification
"""

import json
import os
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional

import requests


MOLTRUST_API = "https://api.moltrust.ch"


class MolTrustClient:
    """Client for the MolTrust Protocol API.

    Protocol: https://moltrust.ch/
    Source: https://github.com/MoltyCel/moltrust-api
    """

    def __init__(self, data_dir: str = "~/.ownify/trust", api_key: Optional[str] = None):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.environ.get("MOLTRUST_API_KEY", "")
        self.identity_file = self.data_dir / "identity.json"
        self.trust_cache_file = self.data_dir / "trust_cache.json"
        self.ipr_log_file = self.data_dir / "ipr_log.jsonl"

        self._identity = self._load_identity()
        self._trust_cache = self._load_trust_cache()

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    @property
    def did(self) -> Optional[str]:
        if self._identity:
            return self._identity.get("did")
        return None

    @property
    def is_registered(self) -> bool:
        return self._identity is not None and "did" in self._identity

    # ── Identity ─────────────────────────────────────────────

    def register(self, identifier: str, metadata: Optional[dict] = None) -> dict:
        """Register ownify as a DID-based agent on MolTrust.

        This creates your decentralized identity — an Ed25519 keypair
        anchored on Base L2. The keypair is stored locally.

        Args:
            identifier: Short name for this agent (max 40 chars)
            metadata: Optional metadata to attach to the DID

        Returns:
            Registration response with DID, credential, and on-chain anchor
        """
        if self.is_registered:
            print(f"Already registered as {self.did}")
            return self._identity

        payload = {
            "did": identifier,
            "identifier": identifier,
            "metadata": metadata or {"agent": "ownify", "adapter": "openclaw"},
        }

        response = requests.post(
            f"{MOLTRUST_API}/identity/register",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        # Store identity locally
        self._identity = {
            "did": result.get("did", f"did:web:moltrust.ch:agent:{identifier}"),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "identifier": identifier,
            "credential": result.get("verifiable_credential"),
            "on_chain": result.get("on_chain_anchor"),
        }
        self._save_identity()

        print(f"Registered as {self._identity['did']}")
        return result

    def _load_identity(self) -> Optional[dict]:
        if self.identity_file.exists():
            with open(self.identity_file) as f:
                return json.load(f)
        return None

    def _save_identity(self):
        with open(self.identity_file, "w") as f:
            json.dump(self._identity, f, indent=2)

    # ── Verification ─────────────────────────────────────────

    def verify_agent(self, did: str) -> dict:
        """Verify another agent's identity and trust status.

        Checks MolTrust for the agent's DID, credential validity,
        and on-chain anchor. Caches results locally for offline use.

        Args:
            did: The DID to verify (e.g., "did:web:moltrust.ch:agent:other-agent")

        Returns:
            Verification result with trust status
        """
        # Check cache first
        cached = self._get_cached_trust(did)
        if cached:
            return cached

        response = requests.get(
            f"{MOLTRUST_API}/identity/verify/{did}",
            headers=self.headers,
        )
        response.raise_for_status()
        result = response.json()

        # Cache the result
        self._cache_trust(did, result)

        return result

    def check_reputation(self, did: str) -> dict:
        """Query an agent's reputation score.

        Args:
            did: The DID to check

        Returns:
            Reputation data including trust score, ratings, risk level
        """
        cached = self._get_cached_trust(did)
        if cached and "trust_score" in cached:
            return cached

        response = requests.get(
            f"{MOLTRUST_API}/reputation/query/{did}",
            headers=self.headers,
        )
        response.raise_for_status()
        result = response.json()

        self._cache_trust(did, result)
        return result

    def is_trusted(self, did: str, min_score: int = 50) -> bool:
        """Quick check: is this agent trusted enough for escalation?

        Args:
            did: The DID to check
            min_score: Minimum trust score required (0-100)

        Returns:
            True if agent meets the trust threshold
        """
        try:
            rep = self.check_reputation(did)
            score = rep.get("trust_score", 0)
            return score >= min_score
        except Exception:
            return False

    # ── Output Provenance (IPR) ──────────────────────────────

    def sign_output(self, output: str, metadata: Optional[dict] = None) -> dict:
        """Create an Interaction Proof Record for an output.

        Signs the output with your DID, creating a cryptographic
        proof of what your agent produced and when.

        Args:
            output: The text output to sign
            metadata: Optional metadata (model, confidence, etc.)

        Returns:
            IPR record with signature and verification URL
        """
        if not self.is_registered:
            raise RuntimeError("Not registered. Call register() first.")

        output_hash = sha256(output.encode()).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "did": self.did,
            "agent_output": output,
            "output_hash": output_hash,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        response = requests.post(
            f"{MOLTRUST_API}/skill/interaction-proof",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        # Log locally
        self._log_ipr(output_hash, timestamp, result)

        return result

    def verify_ipr(self, ipr_id: str) -> dict:
        """Verify an Interaction Proof Record.

        Args:
            ipr_id: The IPR ID to verify

        Returns:
            Verification result
        """
        response = requests.get(
            f"{MOLTRUST_API}/ipr/verify/{ipr_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def _log_ipr(self, output_hash: str, timestamp: str, result: dict):
        entry = {
            "output_hash": output_hash,
            "timestamp": timestamp,
            "ipr_id": result.get("ipr_id"),
            "verification_url": result.get("verification_url"),
        }
        with open(self.ipr_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Trust Cache ──────────────────────────────────────────

    def _load_trust_cache(self) -> dict:
        if self.trust_cache_file.exists():
            with open(self.trust_cache_file) as f:
                return json.load(f)
        return {}

    def _save_trust_cache(self):
        with open(self.trust_cache_file, "w") as f:
            json.dump(self._trust_cache, f, indent=2)

    def _cache_trust(self, did: str, data: dict, ttl: int = 3600):
        self._trust_cache[did] = {
            "data": data,
            "cached_at": time.time(),
            "ttl": ttl,
        }
        self._save_trust_cache()

    def _get_cached_trust(self, did: str) -> Optional[dict]:
        entry = self._trust_cache.get(did)
        if not entry:
            return None
        if time.time() - entry["cached_at"] > entry["ttl"]:
            return None
        return entry["data"]

    def clear_cache(self):
        """Clear the local trust cache."""
        self._trust_cache = {}
        self._save_trust_cache()

    # ── Status ───────────────────────────────────────────────

    def status(self) -> dict:
        """Get current MolTrust integration status."""
        return {
            "registered": self.is_registered,
            "did": self.did,
            "identity_file": str(self.identity_file),
            "cached_agents": len(self._trust_cache),
            "api_key_set": bool(self.api_key),
            "api_url": MOLTRUST_API,
        }

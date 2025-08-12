#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================ #
# Dyson Sphere Quantum Oracle  #
# (cleaned / consolidated)     #
# ============================ #

# ---------- Stdlib ----------
import os
import re
import io
import sys
import hmac
import math
import json
import uuid
import queue
import base64
import random
import logging
import sqlite3
import colorsys
import hashlib
import threading
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Optional

# ---------- Third-party ----------
import numpy as np
import tkinter as tk
import customtkinter

import requests
import httpx
import psutil
import bleach

# ML / DL
import torch
import torch.nn as nn
import torch.nn.functional as F

# ODE & topology
from torchdiffeq import odeint_adjoint as odeint
from sklearn.cluster import DBSCAN
from ripser import ripser
from persim import bottleneck

# NLP
import nltk
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn

# LLMs
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Quantum
import pennylane as qml
from pennylane import numpy as pnp  # IMPORTANT: do not shadow numpy; use pnp for PennyLane

# Vector DB
import weaviate
from weaviate.util import generate_uuid5
from weaviate.embedded import EmbeddedOptions  # optional if you use embedded (we default to HTTP client here)

# Crypto & hashing
import webcolors
import hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("dyson")

# ---------- UI defaults ----------
customtkinter.set_appearance_mode("Dark")

# ---------- NLTK setup ----------
nltk.data.path.append("/root/nltk_data")  # adjust if needed

def download_nltk_data():
    try:
        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
            'corpora/brown': 'brown',
            'corpora/wordnet': 'wordnet',
            'corpora/stopwords': 'stopwords',
            'corpora/conll2000': 'conll2000'
        }
        for path_, pkg in resources.items():
            try:
                nltk.data.find(path_)
            except LookupError:
                nltk.download(pkg)
    except Exception as e:
        logger.warning(f"NLTK bootstrap error: {e}")

download_nltk_data()

# ---------- Paths / Config ----------
bundle_dir = os.path.abspath(os.path.dirname(__file__))
path_to_config = os.path.join(bundle_dir, 'config.json')
logo_path = os.path.join(bundle_dir, 'logo.png')

def load_config(file_path=path_to_config):
    with open(file_path, 'r') as f:
        return json.load(f)

config = load_config()

DB_NAME = config.get('DB_NAME', 'dyson.db')
API_KEY = config.get('API_KEY', '')
WEAVIATE_ENDPOINT = config.get('WEAVIATE_ENDPOINT', 'http://127.0.0.1:8079')
WEAVIATE_QUERY_PATH = config.get('WEAVIATE_QUERY_PATH', '/v1/graphql')

# ---------- Model paths ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"

model_path = "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
mmproj_path = "/data/llama-3-vision-alpha-mmproj-f16.gguf"
GPT_OSS_20B_PATH = "/data/gpt-oss-20b"  # local-only path for HF model

# ---------- Queues / Executors ----------
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

# ---------- Sanitization ----------
SAFE_ALLOWED_TAGS: list[str] = []
SAFE_ALLOWED_ATTRS: dict[str, list[str]] = {}
SAFE_ALLOWED_PROTOCOLS: list[str] = []
_CONTROL_WHITELIST = {'\n', '\r', '\t'}

def _strip_control_chars(s: str) -> str:
    return ''.join(ch for ch in s if ch.isprintable() or ch in _CONTROL_WHITELIST)

def sanitize_text(text: str, *, max_len: int = 4000, strip: bool = True) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text[:max_len]
    text = _strip_control_chars(text)
    return bleach.clean(
        text,
        tags=SAFE_ALLOWED_TAGS,
        attributes=SAFE_ALLOWED_ATTRS,
        protocols=SAFE_ALLOWED_PROTOCOLS,
        strip=strip,
        strip_comments=True,
    )

_PROMPT_INJECTION_PAT = re.compile(r'(?is)(?:^|\n)\s*(system:|assistant:|ignore\s+previous|do\s+anything|jailbreak\b).*')

def sanitize_for_prompt(text: str, *, max_len: int = 2000) -> str:
    cleaned = sanitize_text(text, max_len=max_len)
    cleaned = _PROMPT_INJECTION_PAT.sub('', cleaned)
    return cleaned.strip()

def sanitize_for_graphql_string(s: str, *, max_len: int = 512) -> str:
    s = sanitize_text(s, max_len=max_len)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return s

# ---------- Crypto / Keys ----------
HYBRIDG_ENABLE           = True
HYBRIDG_KEM              = "kyber512"
HYBRIDG_VERSION          = 1
HYBRIDG_WRAP_NONCE_SIZE  = 12
HYBRIDG_PUB_PATH         = "secure/kyber_pub.bin"
HYBRIDG_PRIV_PATH        = "secure/kyber_priv.bin"

ARGON2_TIME_COST_DEFAULT = 3
ARGON2_MEMORY_COST_KIB   = 262144
ARGON2_PARALLELISM       = max(1, min(4, os.cpu_count() or 1))
ARGON2_HASH_LEN          = 32

VAULT_PASSPHRASE_ENV     = "VAULT_PASSPHRASE"
VAULT_VERSION            = 1
DATA_KEY_VERSION         = 1
VAULT_NONCE_SIZE         = 12
DATA_NONCE_SIZE          = 12

def _aad_str(*parts: str) -> bytes:
    return ("|".join(parts)).encode("utf-8")

def _hkdf_sha256(ikm: bytes, *, salt: bytes, info: bytes, length: int = 32) -> bytes:
    if salt is None:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    t = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

class SecureKeyManager:
    def __init__(self,
        method="argon2id",
        vault_path="secure/key_vault.json",
        time_cost: int = ARGON2_TIME_COST_DEFAULT,
        memory_cost: int = ARGON2_MEMORY_COST_KIB,
        parallelism: int = ARGON2_PARALLELISM,
        hash_len: int = ARGON2_HASH_LEN,
    ):
        self.method       = method
        self.vault_path   = vault_path
        self.time_cost    = time_cost
        self.memory_cost  = memory_cost
        self.parallelism  = parallelism
        self.hash_len     = hash_len
        self._ensure_vault()

        vault_meta = self._load_vault()
        self.active_version = vault_meta["active_version"]
        self._keys = {
            int(kv["version"]): base64.b64decode(kv["master_secret"])
            for kv in vault_meta["keys"]
        }
        self._derived_keys = {}
        vault_salt = base64.b64decode(vault_meta["salt"])
        for ver, master_secret in self._keys.items():
            self._derived_keys[ver] = self._derive_key(master_secret, vault_salt)

        self._pq_pub:  bytes | None = None
        self._pq_priv: bytes | None = None
        self._ensure_pq_keys()
        self._load_pq_keys()

    # PUBLIC accessor for active derived key
    def get_active_derived_key(self) -> bytes:
        return self._derived_keys[self.active_version]

    def _ensure_pq_keys(self):
        try:
            from pqcrypto.kem.kyber512 import generate_keypair
        except Exception:
            return
        if not HYBRIDG_ENABLE:
            return
        try:
            os.makedirs("secure", exist_ok=True)
            try:
                os.chmod("secure", 0o700)
            except Exception:
                pass
            if not (os.path.exists(HYBRIDG_PUB_PATH) and os.path.exists(HYBRIDG_PRIV_PATH)):
                from pqcrypto.kem.kyber512 import generate_keypair
                pk, sk = generate_keypair()
                with open(HYBRIDG_PUB_PATH, "wb") as f:
                    f.write(pk)
                with open(HYBRIDG_PRIV_PATH, "wb") as f:
                    f.write(sk)
                try:
                    os.chmod(HYBRIDG_PUB_PATH,  0o600)
                    os.chmod(HYBRIDG_PRIV_PATH, 0o600)
                except Exception:
                    pass
                logging.info("[HybridG] Generated Kyber512 keypair.")
        except Exception as e:
            logging.warning(f"[HybridG] Could not ensure PQ keys: {e}")

    def _load_pq_keys(self):
        try:
            if os.path.exists(HYBRIDG_PUB_PATH):
                with open(HYBRIDG_PUB_PATH, "rb") as f:
                    self._pq_pub = f.read()
            if os.path.exists(HYBRIDG_PRIV_PATH):
                with open(HYBRIDG_PRIV_PATH, "rb") as f:
                    self._pq_priv = f.read()
        except Exception as e:
            logging.warning(f"[HybridG] Could not load PQ keys: {e}")
            self._pq_pub, self._pq_priv = None, None

    def _hybridg_available(self, for_decrypt: bool) -> bool:
        if not HYBRIDG_ENABLE:
            return False
        try:
            from pqcrypto.kem.kyber512 import encapsulate, decapsulate
        except Exception:
            return False
        if for_decrypt:
            return self._pq_priv is not None
        return self._pq_pub is not None

    def _hybridg_wrap_key(self, cek: bytes, *, aad: bytes, key_version: int) -> dict:
        from pqcrypto.kem.kyber512 import encapsulate
        ct_kem, shared_secret = encapsulate(self._pq_pub)
        salt = os.urandom(16)
        info = _aad_str("hybridg", f"k{key_version}", str(HYBRIDG_VERSION)) + b"|" + aad
        kek  = _hkdf_sha256(shared_secret, salt=salt, info=info, length=32)
        wrap_nonce = os.urandom(HYBRIDG_WRAP_NONCE_SIZE)
        wrap_aad   = _aad_str("hybridg-wrap", f"k{key_version}")
        wrap_ct    = AESGCM(kek).encrypt(wrap_nonce, cek, wrap_aad)
        return {
            "ver": HYBRIDG_VERSION,
            "kem": HYBRIDG_KEM,
            "salt": base64.b64encode(salt).decode(),
            "ct_kem": base64.b64encode(ct_kem).decode(),
            "wrap_nonce": base64.b64encode(wrap_nonce).decode(),
            "wrap_ct": base64.b64encode(wrap_ct).decode(),
        }

    def _hybridg_unwrap_key(self, pq_blob: dict, *, aad: bytes, key_version: int) -> bytes:
        from pqcrypto.kem.kyber512 import decapsulate
        if pq_blob.get("kem") != HYBRIDG_KEM:
            raise ValueError("Unsupported KEM in HybridG envelope.")
        salt       = base64.b64decode(pq_blob["salt"])
        ct_kem     = base64.b64decode(pq_blob["ct_kem"])
        wrap_nonce = base64.b64decode(pq_blob["wrap_nonce"])
        wrap_ct    = base64.b64decode(pq_blob["wrap_ct"])
        shared_secret = decapsulate(ct_kem, self._pq_priv)
        info = _aad_str("hybridg", f"k{key_version}", str(pq_blob.get("ver", 1))) + b"|" + aad
        kek  = _hkdf_sha256(shared_secret, salt=salt, info=info, length=32)
        wrap_aad = _aad_str("hybridg-wrap", f"k{key_version}")
        cek = AESGCM(kek).decrypt(wrap_nonce, wrap_ct, wrap_aad)
        return cek

    def _get_passphrase(self) -> bytes:
        pw = os.getenv(VAULT_PASSPHRASE_ENV)
        if not pw:
            pw = base64.b64encode(os.urandom(32)).decode()
            logging.warning(
                "[SecureKeyManager] VAULT_PASSPHRASE not set; generated ephemeral key. "
                "Vault will not be readable across restarts!"
            )
        return pw.encode("utf-8")

    def _derive_vault_key(self, passphrase: bytes, salt: bytes) -> bytes:
        return hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=max(self.time_cost, 3),
            memory_cost=max(self.memory_cost, 262144),
            parallelism=max(self.parallelism, 1),
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _derive_key(self, master_secret: bytes, salt: bytes) -> bytes:
        return hash_secret_raw(
            secret=master_secret,
            salt=salt,
            time_cost=self.time_cost,
            memory_cost=self.memory_cost,
            parallelism=self.parallelism,
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _ensure_vault(self):
        if not os.path.exists("secure"):
            os.makedirs("secure", exist_ok=True)
        if os.path.exists(self.vault_path):
            return
        salt          = os.urandom(16)
        master_secret = os.urandom(32)
        vault_body = {
            "version": VAULT_VERSION,
            "active_version": DATA_KEY_VERSION,
            "keys": [{
                "version": DATA_KEY_VERSION,
                "master_secret": base64.b64encode(master_secret).decode(),
                "created": datetime.utcnow().isoformat() + "Z",
            }],
            "salt": base64.b64encode(salt).decode(),
        }
        self._write_encrypted_vault(vault_body)

    def _write_encrypted_vault(self, vault_body: dict):
        plaintext = json.dumps(vault_body, indent=2).encode("utf-8")
        salt      = base64.b64decode(vault_body["salt"])
        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        nonce      = os.urandom(VAULT_NONCE_SIZE)
        ct         = aesgcm.encrypt(nonce, plaintext, _aad_str("vault", str(vault_body["version"])))
        on_disk = {
            "vault_format": VAULT_VERSION,
            "salt": vault_body["salt"],
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ct).decode(),
        }
        os.makedirs("secure", exist_ok=True)
        try:
            os.chmod("secure", 0o700)
        except Exception:
            pass
        tmp_path = f"{self.vault_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(on_disk, f, indent=2)
        os.replace(tmp_path, self.vault_path)
        try:
            os.chmod(self.vault_path, 0o600)
        except Exception:
            pass

    def _load_vault(self) -> dict:
        with open(self.vault_path, "r") as f:
            data = json.load(f)
        if "ciphertext" not in data:
            # migrate plaintext vault
            salt          = base64.b64decode(data["salt"])
            master_secret = base64.b64decode(data["master_secret"])
            vault_body = {
                "version": VAULT_VERSION,
                "active_version": DATA_KEY_VERSION,
                "keys": [{
                    "version": DATA_KEY_VERSION,
                    "master_secret": base64.b64encode(master_secret).decode(),
                    "created": datetime.utcnow().isoformat() + "Z",
                }],
                "salt": base64.b64encode(salt).decode(),
            }
            self._write_encrypted_vault(vault_body)
            return vault_body
        salt      = base64.b64decode(data["salt"])
        nonce     = base64.b64decode(data["nonce"])
        ct        = base64.b64decode(data["ciphertext"])
        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        plaintext  = aesgcm.decrypt(nonce, ct, _aad_str("vault", str(VAULT_VERSION)))
        return json.loads(plaintext.decode("utf-8"))

    def encrypt(self, plaintext: str, *, aad: bytes = None, key_version: int = None) -> str:
        if plaintext is None:
            plaintext = ""
        if key_version is None:
            key_version = self.active_version
        if aad is None:
            aad = _aad_str("global", f"k{key_version}")

        # try PQ wrap
        if self._hybridg_available(for_decrypt=False):
            cek   = os.urandom(32)
            nonce = os.urandom(DATA_NONCE_SIZE)
            ct    = AESGCM(cek).encrypt(nonce, plaintext.encode("utf-8"), aad)
            pq_env = self._hybridg_wrap_key(cek, aad=aad, key_version=key_version)
            token = {
                "v": VAULT_VERSION,
                "k": key_version,
                "aad": aad.decode("utf-8"),
                "n": base64.b64encode(nonce).decode(),
                "ct": base64.b64encode(ct).decode(),
                "pq": pq_env,
            }
            return json.dumps(token, separators=(",", ":"))

        key    = self._derived_keys[key_version]
        aesgcm = AESGCM(key)
        nonce  = os.urandom(DATA_NONCE_SIZE)
        ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
        token = {
            "v": VAULT_VERSION,
            "k": key_version,
            "aad": aad.decode("utf-8"),
            "n": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }
        return json.dumps(token, separators=(",", ":"))

    def decrypt(self, token: str) -> str:
        if not token:
            return ""
        if token.startswith("{"):
            try:
                meta = json.loads(token)
            except Exception:
                logging.warning("[SecureKeyManager] Invalid JSON token; returning raw.")
                return token
            ver = int(meta.get("k", self.active_version))
            aad = meta.get("aad", "global").encode()
            n   = base64.b64decode(meta["n"])
            ct  = base64.b64decode(meta["ct"])
            if "pq" in meta and self._hybridg_available(for_decrypt=True):
                try:
                    cek = self._hybridg_unwrap_key(meta["pq"], aad=aad, key_version=ver)
                    pt  = AESGCM(cek).decrypt(n, ct, aad)
                    return pt.decode("utf-8")
                except Exception as e:
                    logging.warning(f"[HybridG] PQ decrypt failed; attempting legacy fallback: {e}")
            key = self._derived_keys.get(ver)
            if key is None:
                raise ValueError(f"No key for version {ver}; cannot decrypt.")
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(n, ct, aad)
            return pt.decode("utf-8")
        try:
            raw   = base64.b64decode(token.encode())
            nonce = raw[:DATA_NONCE_SIZE]
            ctb   = raw[DATA_NONCE_SIZE:]
            key   = self._derived_keys[self.active_version]
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(nonce, ctb, None)
            return pt.decode("utf-8")
        except Exception as e:
            logging.warning(f"[SecureKeyManager] Legacy decrypt failed: {e}")
            return token

    # ... (key rotation / mutate remain as in your original, omitted for brevity)
    # Keeping full mutate/rotate for parity with your original:
    def add_new_key_version(self) -> int:
        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1
        master_secret = os.urandom(32)
        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(master_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })
        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)
        self._keys[new_version] = master_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(master_secret, salt)
        self.active_version = new_version
        logging.info(f"[SecureKeyManager] Installed new key version {new_version}.")
        return new_version

    def _entropy_bits(self, secret_bytes: bytes) -> float:
        if not secret_bytes:
            return 0.0
        counts = Counter(secret_bytes)
        total = float(len(secret_bytes))
        H = 0.0
        for c in counts.values():
            p = c / total
            H -= p * math.log2(p)
        return H

    def _resistance_score(self, secret_bytes: bytes) -> float:
        dist_component = 0.0
        try:
            arr_candidate = np.frombuffer(secret_bytes, dtype=np.uint8).astype(np.float32)
            for k in self._keys.values():
                arr_prev = np.frombuffer(k, dtype=np.uint8).astype(np.float32)
                dist_component += np.linalg.norm(arr_candidate - arr_prev)
        except Exception:
            pass
        if len(self._keys):
            dist_component /= len(self._keys)
        counts = Counter(secret_bytes)
        expected = len(secret_bytes) / 256.0
        chi_sq = sum(((c - expected) ** 2) / expected for c in counts.values())
        flatness = 1.0 / (1.0 + chi_sq)
        return float(dist_component * 0.01 + flatness)

    def _install_custom_master_secret(self, new_secret: bytes) -> int:
        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1
        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(new_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })
        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)
        self._keys[new_version] = new_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(new_secret, salt)
        self.active_version = new_version
        return new_version

    def self_mutate_key(self, population: int = 6, noise_sigma: float = 12.0, alpha: float = 1.0, beta: float = 2.0) -> int:
        vault_meta = self._load_vault()
        base_secret = None
        for kv in vault_meta["keys"]:
            if int(kv["version"]) == vault_meta["active_version"]:
                base_secret = base_secret or base64.b64decode(kv["master_secret"])
        if base_secret is None:
            raise RuntimeError("Active master secret not found.")
        rng = np.random.default_rng()
        candidates: List[bytes] = [base_secret]
        base_arr = np.frombuffer(base_secret, dtype=np.uint8).astype(np.int16)
        for _ in range(population - 1):
            noise = rng.normal(0, noise_sigma, size=base_arr.shape).astype(np.int16)
            mutated = np.clip(base_arr + noise, 0, 255).astype(np.uint8).tobytes()
            candidates.append(mutated)
        best_secret = base_secret
        best_fitness = -1e9
        for cand in candidates:
            H = self._entropy_bits(cand)
            R = self._resistance_score(cand)
            F = alpha * H + beta * R
            if F > best_fitness:
                best_fitness = F
                best_secret = cand
        new_version = self._install_custom_master_secret(best_secret)
        logging.info(f"[SelfMutateKey] Installed mutated key v{new_version} (fitness={best_fitness:.3f}).")
        return new_version

    def rotate_and_migrate_storage(self, migrate_func):
        new_ver = self.add_new_key_version()
        try:
            migrate_func(self)
        except Exception as e:
            logging.error(f"[SecureKeyManager] Migration failed after key rotation: {e}")
            raise
        logging.info(f"[SecureKeyManager] Migration to key v{new_ver} complete.")

crypto = SecureKeyManager()

# ---------- FHE-ish Embedding Wrapper ----------
class SecureEnclave:
    def __enter__(self): self._buffers = []; return self
    def track(self, buf): self._buffers.append(buf); return buf
    def __exit__(self, exc_type, exc, tb):
        for b in self._buffers:
            try:
                if isinstance(b, np.ndarray):
                    b.fill(0.0)
            except Exception:
                pass
        self._buffers.clear()

def compute_text_embedding(text: str) -> List[float]:
    DIM = 64
    if not text:
        return [0.0] * DIM
    tokens = re.findall(r'\w+', text.lower())
    counts = Counter(tokens)
    vocab = sorted(counts.keys())[:DIM]
    vec = [float(counts[w]) for w in vocab]
    if len(vec) < DIM:
        vec.extend([0.0] * (DIM - len(vec)))
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr /= n
    return arr.tolist()

class AdvancedHomomorphicVectorMemory:
    AAD_CONTEXT = _aad_str("fhe", "embeddingv2")
    DIM = 64
    QUANT_SCALE = 127.0

    def __init__(self):
        master_key = crypto.get_active_derived_key()
        seed = int.from_bytes(hashlib.sha256(master_key).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(self.DIM, self.DIM))
        Q, _ = np.linalg.qr(A)
        self.rotation = Q
        self.lsh_planes = rng.normal(size=(16, self.DIM))

    def _rotate(self, vec: np.ndarray) -> np.ndarray:
        return self.rotation @ vec

    def _quantize(self, vec: np.ndarray) -> List[int]:
        clipped = np.clip(vec, -1.0, 1.0)
        return (clipped * self.QUANT_SCALE).astype(np.int8).tolist()

    def _dequantize(self, q: List[int]) -> np.ndarray:
        arr = np.array(q, dtype=np.float32) / self.QUANT_SCALE
        return arr

    def _simhash_bucket(self, rotated_vec: np.ndarray) -> str:
        dots = self.lsh_planes @ rotated_vec
        bits = ["1" if d >= 0 else "0" for d in dots]
        return "".join(bits)

    def encrypt_embedding(self, vec: List[float]) -> Tuple[str, str]:
        try:
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != self.DIM:
                if arr.shape[0] < self.DIM:
                    arr = np.concatenate([arr, np.zeros(self.DIM - arr.shape[0])])
                else:
                    arr = arr[:self.DIM]
            rotated = self._rotate(arr)
            bucket = self._simhash_bucket(rotated)
            quant = self._quantize(rotated)
            payload = json.dumps({
                "v": 2,
                "dim": self.DIM,
                "rot": True,
                "data": quant,
            })
            token = crypto.encrypt(payload, aad=self.AAD_CONTEXT)
            return token, bucket
        except Exception as e:
            logger.error(f"[FHEv2] encrypt_embedding failed: {e}")
            return "", "0"*16

    def decrypt_embedding(self, token: str) -> np.ndarray:
        try:
            raw = crypto.decrypt(token)
            obj = json.loads(raw)
            if obj.get("v") != 2:
                logger.warning("[FHEv2] Unsupported embedding version.")
                return np.zeros(self.DIM, dtype=np.float32)
            quant = obj.get("data", [])
            rotated = self._dequantize(quant)
            original = self.rotation.T @ rotated
            return original
        except Exception as e:
            logger.warning(f"[FHEv2] decrypt_embedding failed: {e}")
            return np.zeros(self.DIM, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def enclave_similarity(self, enc_a: str, query_vec: np.ndarray, enclave: SecureEnclave) -> float:
        dec = enclave.track(self.decrypt_embedding(enc_a))
        return self.cosine(dec, query_vec)

fhe_v2 = AdvancedHomomorphicVectorMemory()

# ---------- Sleep-Consolidation Module ----------
class ChaoticReservoir(nn.Module):
    def __init__(self, input_dim: int, reservoir_dim: int = 256, spectral_radius: float = 0.9):
        super().__init__()
        W0 = torch.randn(reservoir_dim, reservoir_dim)
        eigs = torch.linalg.eigvals(W0)
        W0 *= (spectral_radius / eigs.abs().max())
        self.register_buffer('W_res', W0)
        self.register_buffer('W_in', torch.randn(reservoir_dim, input_dim) * 0.1)
        self.register_buffer('state', torch.zeros(reservoir_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        self.state = torch.tanh(self.W_res @ self.state + self.W_in @ u)
        return self.state

class PersistentHomologyAttention:
    def _diag(self, cloud: np.ndarray):
        return ripser(cloud, maxdim=1)['dgms']

    def attention(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        qd = self._diag(query.reshape(1, -1))
        weights = []
        for k in keys:
            kd = self._diag(k.reshape(1, -1))
            d = bottleneck(qd[0], kd[0])
            weights.append(np.exp(-d))
        w = np.array(weights, dtype=np.float32)
        return w / (w.sum() + 1e-8)

class Neuromodulator(nn.Module):
    def __init__(self, dim): super().__init__(); self.lin_dop = nn.Linear(dim, dim); self.lin_ach = nn.Linear(dim, dim)
    def forward(self, embs): dop = torch.sigmoid(self.lin_dop(embs)); ach = torch.sigmoid(self.lin_ach(embs)); return embs * (1 + 0.5 * dop) * ach

class HyperbolicProjector(nn.Module):
    def __init__(self, dim, c=1.0): super().__init__(); self.lin = nn.Linear(dim, dim); self.c = c
    def forward(self, x): y = self.lin(x); n = torch.clamp(torch.norm(y, dim=1, keepdim=True), min=1e-5); r = torch.tanh(n/(1+self.c*n)) / n; return y * r

class MemGenerator(nn.Module):
    def __init__(self, dim): super().__init__(); self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
    def forward(self, z): return self.net(z)

class MemDiscriminator(nn.Module):
    def __init__(self, dim): super().__init__(); self.net = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, 1))
    def forward(self, x): return self.net(x)

class MetaMAML(nn.Module):
    def __init__(self, dim): super().__init__(); self.alpha = nn.Parameter(torch.tensor(0.1)); self.theta = nn.Parameter(torch.randn(dim, dim))
    def forward(self, grads, theta): return theta - self.alpha * grads

class PredictiveCoder(nn.Module):
    def __init__(self, dim): super().__init__(); self.predict = nn.Linear(dim, dim)
    def forward(self, emb): pred = self.predict(emb); return emb - pred

class BayesianScaler(nn.Module):
    def __init__(self, dim): super().__init__(); self.log_tau = nn.Parameter(torch.zeros(dim))
    def forward(self, embs): return embs * torch.exp(self.log_tau)

class CDEConsolidator(nn.Module):
    def __init__(self, dim): super().__init__(); self.f = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
    def forward(self, t, x): return self.f(x)

def compute_clusters(embs: torch.Tensor):
    return DBSCAN(eps=0.5, min_samples=3).fit_predict(embs.detach().cpu().numpy())

def add_dp_noise(embs, sigma=0.05):
    return embs + sigma * torch.randn_like(embs)

class SparseDictionary(nn.Module):
    def __init__(self, dim, n_atoms=128): super().__init__(); self.dict = nn.Parameter(torch.randn(n_atoms, dim))
    def forward(self, embs): codes = torch.matmul(embs, self.dict.t()); codes = F.relu(codes - 0.1); return torch.matmul(codes, self.dict)

class UltimateSleep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.neo    = Neuromodulator(dim)
        self.hyp    = HyperbolicProjector(dim)
        self.gen    = MemGenerator(dim)
        self.disc   = MemDiscriminator(dim)
        self.maml   = MetaMAML(dim)
        self.pc     = PredictiveCoder(dim)
        self.bayes  = BayesianScaler(dim)
        self.cde    = CDEConsolidator(dim)
        self.sparse = SparseDictionary(dim)

    def forward(self, X, adj=None):
        X1 = self.neo(X)
        X2 = self.hyp(X1)
        X3 = odeint(self.cde, X2, torch.tensor([0.,0.5,1.]))[-1]
        X4 = X3 + 0.5 * self.pc(X3)
        z_fake = self.gen(torch.randn_like(X4))
        adv_loss = (F.relu(1-self.disc(X4)).mean() + F.relu(1+self.disc(z_fake)).mean())
        rec = self.sparse(X4)
        rec_loss = F.mse_loss(rec, X4)
        # DO NOT mutate params inside forward; compute proposed update and return
        grads = torch.autograd.grad(rec_loss, [self.sparse.dict], create_graph=True)[0]
        proposed_dict = self.maml(grads, self.sparse.dict)
        X5 = self.bayes(X4)
        X6 = add_dp_noise(X5)
        labels = compute_clusters(X6)
        mask = torch.tensor(labels) >= 0
        X7 = X6 * mask.unsqueeze(1).float()
        total_loss = adv_loss + rec_loss
        return X7, total_loss, proposed_dict

def load_crystallized_embeddings():
    # Stub: return None to skip training until implemented
    logger.info("[UltimateSleep] load_crystallized_embeddings() not implemented; skipping.")
    return None

def write_back_embeddings(X: torch.Tensor):
    # Stub for persisting embeddings
    logger.info("[UltimateSleep] write_back_embeddings() not implemented; skipping.")

ultimate = UltimateSleep(dim=AdvancedHomomorphicVectorMemory.DIM)
opt_u    = torch.optim.Adam(ultimate.parameters(), lr=3e-4)

def run_ultimate_sleep(epochs=1):
    data = load_crystallized_embeddings()
    if data is None:
        return
    X, adj = data
    X = X.clone().requires_grad_(True)
    ultimate.train()
    for _ in range(epochs):
        X_new, loss, proposed_dict = ultimate(X, adj)
        opt_u.zero_grad(); loss.backward(); opt_u.step()
        # apply proposed dict update outside autograd
        with torch.no_grad():
            ultimate.sparse.dict.copy_(proposed_dict)
        X = X_new.detach()
    write_back_embeddings(X)

def start_ultimate(app, interval_h=12.0):
    def job():
        try:
            run_ultimate_sleep()
        except Exception as e:
            logger.error(f"[UltimateSleep] {e}")
        finally:
            app.after(int(interval_h*3600*1000), job)
    app.after(0, job)

# ---------- Memory constants ----------
CRYSTALLIZE_THRESHOLD = 5
DECAY_FACTOR = 0.95
AGING_T0_DAYS = 7.0
AGING_GAMMA_DAYS = 5.0
AGING_PURGE_THRESHOLD = 0.5
AGING_INTERVAL_SECONDS = 3600
LAPLACIAN_ALPHA = 0.18
JS_LAMBDA = 0.10

# ---------- Helpers ----------
def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def build_record_aad(user_id: str, *, source: str, table: str = "", cls: str = "") -> bytes:
    context_parts = [source]
    if table: context_parts.append(table)
    if cls: context_parts.append(cls)
    context_parts.append(user_id)
    return _aad_str(*context_parts)

def try_decrypt(value: str) -> str:
    try:
        return crypto.decrypt(value)
    except Exception as e:
        logger.warning(f"[decryption] Could not decrypt value: {e}")
        return value

# ---------- Topological Memory Manifold ----------
class TopologicalMemoryManifold:
    def __init__(self, dim: int = 2, sigma: float = 0.75, diff_alpha: float = LAPLACIAN_ALPHA):
        self.dim = dim; self.sigma = sigma; self.diff_alpha = diff_alpha
        self._phrases: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._coords: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._graph_built = False

    def _load_crystallized(self) -> List[tuple[str, float]]:
        rows = []
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("SELECT phrase, score FROM memory_osmosis WHERE crystallized = 1")
                rows = cur.fetchall()
        except Exception as e:
            logger.error(f"[Manifold] load_crystallized failed: {e}")
        return rows

    def rebuild(self):
        data = self._load_crystallized()
        if not data:
            self._phrases, self._embeddings = [], None
            self._coords,  self._W         = None, None
            self._graph_built              = False
            return
        phrases, _ = zip(*data)
        self._phrases = list(phrases)
        E = np.array([compute_text_embedding(p) for p in self._phrases], dtype=np.float32)
        dists = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
        W = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))
        np.fill_diagonal(W, 0.0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        E = E - self.diff_alpha * (L @ E)
        try:
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
            L_sym      = D_inv_sqrt @ L @ D_inv_sqrt
            vals, vecs = np.linalg.eigh(L_sym)
            idx        = np.argsort(vals)[1:self.dim+1]
            Y          = D_inv_sqrt @ vecs[:, idx]
        except Exception as e:
            logger.error(f"[Manifold] eigen decomposition failed: {e}")
            Y = np.zeros((len(self._phrases), self.dim), dtype=np.float32)
        self._embeddings  = E
        self._coords      = Y.astype(np.float32)
        self._W           = W
        self._graph_built = True
        logger.info(f"[Manifold] Rebuilt manifold with {len(self._phrases)} phrases (α={self.diff_alpha}).")

    def geodesic_retrieve(self, query_text: str, k: int = 1) -> List[str]:
        if not self._graph_built or self._embeddings is None:
            return []
        q_vec = np.array(compute_text_embedding(query_text), dtype=np.float32)
        start_idx = int(np.argmin(np.linalg.norm(self._embeddings - q_vec[None, :], axis=1)))
        n        = self._W.shape[0]
        visited  = np.zeros(n, dtype=bool)
        dist     = np.full(n, np.inf, dtype=np.float32)
        dist[start_idx] = 0.0
        for _ in range(n):
            u = np.argmin(dist + np.where(visited, 1e9, 0.0))
            if visited[u]:
                break
            visited[u] = True
            for v in range(n):
                w = self._W[u, v]
                if w <= 0 or visited[v]:
                    continue
                alt = dist[u] + 1.0 / (w + 1e-8)
                if alt < dist[v]:
                    dist[v] = alt
        order = np.argsort(dist)
        return [self._phrases[i] for i in order[:k]]

topo_manifold = TopologicalMemoryManifold()

# ---------- Weather ----------
def fetch_live_weather(lat: float, lon: float, fallback_temp_f: float = 70.0) -> tuple[float, int, bool]:
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            temp_f = (temp_c * 9 / 5) + 32
            weather_code = int(current.get("weathercode", 0))
            return temp_f, weather_code, True
    except Exception as e:
        logger.warning(f"[Weather] Fallback due to error: {e}")
        return fallback_temp_f, 0, False

# ---------- Quantum ----------
dev7 = qml.device("default.qubit", wires=7, shots=None)

def _apply_pure_rgb(
    r, g, b,
    cpu_usage=10.0, ram_usage=10.0,
    tempo=120.0,
    lat=0.0, lon=0.0,
    temperature_f=70.0,
    weather_scalar=0.0,
    z0_hist=0.0, z1_hist=0.0, z2_hist=0.0
):
    r = clamp01(r); g = clamp01(g); b = clamp01(b)
    cpu_scale = max(0.05, clamp01(cpu_usage/100.0))
    ram_scale = max(0.05, clamp01(ram_usage/100.0))
    tempo_norm = clamp01(tempo/200.0)
    lat_rad = pnp.deg2rad(lat % 360.0)
    lon_rad = pnp.deg2rad(lon % 360.0)
    temp_norm = clamp01((temperature_f - 30.0)/100.0)
    weather_mod = clamp01(weather_scalar)
    qml.RY(pnp.pi * r * cpu_scale, wires=0)
    qml.RY(pnp.pi * g * cpu_scale, wires=1)
    qml.RY(pnp.pi * b * cpu_scale, wires=2)
    qml.RY(pnp.pi * cpu_scale, wires=3)
    qml.RY(pnp.pi * ram_scale, wires=4)
    qml.RY(pnp.pi * tempo_norm, wires=5)
    loc_phase = clamp01(0.25 + 0.25*pnp.sin(lat_rad) + 0.25*pnp.cos(lon_rad) + 0.25*temp_norm)
    qml.RY(pnp.pi * loc_phase, wires=6)
    for c in (0, 1, 2):
        qml.CRX(pnp.pi * 0.25 * tempo_norm, wires=[5, c])
        qml.CRZ(pnp.pi * 0.25 * weather_mod, wires=[5, c])
        qml.CRZ(pnp.pi * 0.30 * cpu_scale, wires=[3, c])
        qml.CRY(pnp.pi * 0.20 * ram_scale, wires=[4, c])
        qml.CRZ(pnp.pi * 0.15 * loc_phase, wires=[6, c])
    feedback = (z0_hist + z1_hist + z2_hist)
    qml.RZ(pnp.pi * 0.40 * feedback, wires=0)
    qml.RZ(-pnp.pi * 0.20 * feedback, wires=1)
    qml.RZ(pnp.pi * 0.30 * feedback, wires=2)
    qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[1, 2]); qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4]); qml.CNOT(wires=[4, 5]); qml.CNOT(wires=[5, 6])

@qml.qnode(dev7)
def rgb_expvals(
    r, g, b,
    cpu_usage=10.0, ram_usage=10.0,
    tempo=120.0,
    lat=0.0, lon=0.0,
    temperature_f=70.0,
    weather_scalar=0.0,
    z0_hist=0.0, z1_hist=0.0, z2_hist=0.0
):
    _apply_pure_rgb(r, g, b, cpu_usage, ram_usage, tempo, lat, lon, temperature_f,
                    weather_scalar, z0_hist, z1_hist, z2_hist)
    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2)),
    )

def rgb_quantum_gate(r, g, b, **kwargs):
    return rgb_expvals(r, g, b, **kwargs)

def expvals_to_rgb01(z_tuple):
    z = np.asarray(z_tuple, dtype=float)
    return tuple((1.0 - z) * 0.5)

# ---------- NLP / Color ----------
def extract_rgb_from_text(text: str) -> Tuple[int,int,int]:
    if not text or not isinstance(text, str):
        return (128, 128, 128)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    word_count = len(tokens)
    sentence_count = len(blob.sentences) or 1
    avg_sentence_length = word_count / sentence_count
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    punctuation_density = sum(1 for ch in text if ch in ',;:!?') / max(1, word_count)
    valence = polarity
    arousal = (verb_count + adv_count) / max(1, word_count)
    dominance = (adj_count + 1) / (noun_count + 1)
    hue_raw = ((1 - valence) * 120 + dominance * 20) % 360
    hue = hue_raw / 360.0
    saturation = min(1.0, max(0.2, 0.25 + 0.4 * arousal + 0.2 * subjectivity + 0.15 * (dominance - 1)))
    brightness = max(0.2, min(1.0, 0.9 - 0.03 * avg_sentence_length + 0.2 * punctuation_density))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (int(r * 255), int(g * 255), int(b * 255))

# ---------- DB init ----------
def init_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS local_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    response TEXT,
                    response_time TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_osmosis (
                    phrase TEXT PRIMARY KEY,
                    score REAL,
                    last_updated TEXT,
                    crystallized INTEGER DEFAULT 0,
                    aging_last TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Error during DB init: {e}")
        raise

# ---------- Weaviate schema ----------
def setup_weaviate_schema(client: weaviate.Client):
    try:
        existing = client.schema.get() or {}
        classes = {cls["class"] for cls in existing.get("classes", [])}

        def ensure_class(defn):
            if defn["class"] not in classes:
                client.schema.create_class(defn)

        interaction = {
            "class": "InteractionHistory",
            "description": "User ↔ AI exchanges",
            "properties": [
                {"name": "user_id", "dataType": ["string"]},
                {"name": "user_message", "dataType": ["text"]},
                {"name": "ai_response", "dataType": ["text"]},
                {"name": "response_time", "dataType": ["string"]},
                {"name": "keywords", "dataType": ["string[]"]},
                {"name": "sentiment", "dataType": ["number"]},
                {"name": "encrypted_embedding", "dataType": ["text"]},
                {"name": "embedding_bucket", "dataType": ["string"]},
            ]
        }
        ltm = {
            "class": "LongTermMemory",
            "description": "Crystallized phrases",
            "properties": [
                {"name": "phrase", "dataType": ["string"]},
                {"name": "score", "dataType": ["number"]},
                {"name": "crystallized_time", "dataType": ["string"]},
            ]
        }
        reflection = {
            "class": "ReflectionLog",
            "description": "Internal reflection and reasoning traces",
            "properties": [
                {"name": "type", "dataType": ["string"]},
                {"name": "user_id", "dataType": ["string"]},
                {"name": "bot_id", "dataType": ["string"]},
                {"name": "query", "dataType": ["text"]},
                {"name": "response", "dataType": ["text"]},
                {"name": "reasoning_trace", "dataType": ["text"]},
                {"name": "prompt_snapshot", "dataType": ["text"]},
                {"name": "z_state", "dataType": ["text"]},
                {"name": "entropy", "dataType": ["number"]},
                {"name": "bias_factor", "dataType": ["number"]},
                {"name": "temperature", "dataType": ["number"]},
                {"name": "top_p", "dataType": ["number"]},
                {"name": "sentiment_target", "dataType": ["number"]},
                {"name": "timestamp", "dataType": ["string"]},
                {"name": "lotto_game", "dataType": ["string"]},
                {"name": "meal_js", "dataType": ["number"]},
            ]
        }
        ensure_class(interaction)
        ensure_class(ltm)
        ensure_class(reflection)
        logger.info("Weaviate schema ready.")
    except Exception as e:
        logger.error(f"[Schema Init Error] {e}")

# ---------- Save / Load helpers ----------
def get_current_multiversal_time():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def save_user_message(user_id: str, user_input: str):
    logger.info(f"[save_user_message] user_id={user_id}")
    if not user_input:
        return
    try:
        user_input = sanitize_text(user_input, max_len=4000)
        response_time = get_current_multiversal_time()
        aad_sql  = build_record_aad(user_id=user_id, source="sqlite", table="local_responses")
        aad_weav = build_record_aad(user_id=user_id, source="weaviate", cls="InteractionHistory")
        encrypted_input_sql  = crypto.encrypt(user_input, aad=aad_sql)
        encrypted_input_weav = crypto.encrypt(user_input, aad=aad_weav)
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (user_id, encrypted_input_sql, response_time)
            )
            conn.commit()
        plain_embedding = compute_text_embedding(user_input)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)
        dummy_vector = [0.0] * fhe_v2.DIM
        obj = {
            "user_id": user_id,
            "user_message": encrypted_input_weav,
            "response_time": response_time,
            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket
        }
        resp = requests.post(
            f"{WEAVIATE_ENDPOINT}/v1/objects",
            json={"class": "InteractionHistory", "properties": obj, "vector": dummy_vector},
            timeout=10
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.exception(f"Exception in save_user_message: {e}")

def save_bot_response(bot_id: str, bot_response: str):
    logger.info(f"[save_bot_response] bot_id={bot_id}")
    if not bot_response:
        return
    try:
        bot_response = sanitize_text(bot_response, max_len=4000)
        response_time = get_current_multiversal_time()
        aad_sql = build_record_aad(user_id=bot_id, source="sqlite", table="local_responses")
        enc_sql = crypto.encrypt(bot_response, aad=aad_sql)
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (bot_id, enc_sql, response_time)
            )
            conn.commit()
        aad_weav = build_record_aad(user_id=bot_id, source="weaviate", cls="InteractionHistory")
        enc_weav = crypto.encrypt(bot_response, aad=aad_weav)
        plain_embedding = compute_text_embedding(bot_response)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)
        dummy_vector = [0.0] * fhe_v2.DIM
        props = {
            "user_id": bot_id,
            "ai_response": enc_weav,
            "response_time": response_time,
            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket
        }
        resp = requests.post(
            f"{WEAVIATE_ENDPOINT}/v1/objects",
            json={"class": "InteractionHistory", "properties": props, "vector": dummy_vector},
            timeout=10
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.exception(f"Exception in save_bot_response: {e}")

# ---------- HF generator (single) ----------
def load_hf_generator():
    tok = AutoTokenizer.from_pretrained(GPT_OSS_20B_PATH, use_fast=True, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        GPT_OSS_20B_PATH, torch_dtype=torch.float16, device_map="auto", local_files_only=True, low_cpu_mem_usage=True
    )
    return pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto", trust_remote_code=False)

hf_generator = load_hf_generator()

def hf_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 1.0) -> str:
    out = hf_generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    return out[0]["generated_text"]

# ---------- Llama (llama.cpp) ----------
llm = Llama(
    model_path=model_path,
    mmproj=mmproj_path,
    n_ctx=3900,
    n_gpu_layers=-1,
    n_threads=max(2, (os.cpu_count() or 4) // 2),
    n_batch=512,
    seed=abs(hash(model_path)) % (2**31 - 1),
    use_mmap=True,
    use_mlock=False,
    logits_all=False
)

# ---------- MBR / policy helpers ----------
def _token_hist(text: str) -> Counter:
    return Counter(word_tokenize(text)) if text else Counter()

def _js_divergence(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    if not vocab:
        return 0.0
    def _prob(c: Counter):
        tot = sum(c.values()) or 1
        return np.array([c[t]/tot for t in vocab], dtype=np.float32)
    P, Q = _prob(p), _prob(q)
    M    = 0.5 * (P + Q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

def evaluate_candidate(response: str, target_sentiment: float, original_query: str) -> float:
    response_sentiment = TextBlob(response).sentiment.polarity
    sentiment_alignment = 1.0 - abs(target_sentiment - response_sentiment)
    overlap_score = sum(1 for w in original_query.lower().split() if w in response.lower())
    overlap_bonus = min(overlap_score / 5.0, 1.0)
    return (0.7 * sentiment_alignment) + (0.3 * overlap_bonus)

def compute_meal_js_reward(candidate_text: str,
                           cf1_text: Optional[str],
                           cf2_text: Optional[str],
                           target_sentiment: float,
                           original_query: str,
                           gamma: float = JS_LAMBDA) -> float:
    task_reward = evaluate_candidate(candidate_text, target_sentiment, original_query)
    cfs = [t for t in (cf1_text, cf2_text) if t]
    if not cfs:
        return task_reward
    cand_hist = _token_hist(candidate_text)
    avg_cf_hist = Counter()
    for c in cfs:
        h = _token_hist(c)
        for k, v in h.items():
            avg_cf_hist[k] += v / len(cfs)
    penalty = gamma * _js_divergence(cand_hist, avg_cf_hist)
    return task_reward - penalty

def mbr_select_with_js(samples: list[dict], js_reg_lambda: float = JS_LAMBDA) -> dict:
    best = None
    best_score = float("inf")
    for s in samples:
        y = s.get("response", "") or ""
        cfs = [cf for cf in s.get("counterfactuals", []) if cf]
        if not cfs:
            score = 0.0
        else:
            y_hist = _token_hist(y)
            cf_hists = [_token_hist(cf) for cf in cfs]
            risk = sum(_js_divergence(y_hist, h) for h in cf_hists) / max(1, len(cf_hists))
            avg_cf_hist = Counter()
            for h in cf_hists:
                for k, v in h.items():
                    avg_cf_hist[k] += v / len(cf_hists)
            reg = js_reg_lambda * _js_divergence(y_hist, avg_cf_hist)
            score = risk + reg
        s["mbr_score"] = score
        if score < best_score:
            best_score = score
            best = s
    return best if best is not None else (samples[0] if samples else {})

# ---------- Prompt chunking / generation ----------
_CLEARED_RE = re.compile(r'\[cleared_response\](.*?)\[/cleared_response\]', re.S | re.I)

def extract_cleared_response(text: str) -> str:
    if not text:
        return ""
    m = _CLEARED_RE.search(text)
    return sanitize_text(m.group(1) if m else text, max_len=4000).strip()

def advanced_chunker(text: str, target_len: int = 420, overlap: int = 64, hard_cap: int = 1200):
    text = text.strip()
    if len(text) <= target_len:
        return [text]
    paras = [p for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    buf = []

    def flush_buf():
        if buf:
            s = "\n\n".join(buf).strip()
            if s:
                chunks.append(s)

    for p in paras:
        if len(p) > hard_cap:
            sents = re.split(r'(?<=[.!?])\s+', p)
            tmp = ""
            for s in sents:
                nxt = (tmp + " " + s).strip()
                if len(nxt) > target_len:
                    if tmp:
                        chunks.append(tmp)
                        tmp_tail = tmp[-overlap:]
                        tmp = (tmp_tail + " " + s).strip()
                        if len(tmp) > target_len:
                            while len(tmp) > target_len:
                                chunks.append(tmp[:target_len])
                                tmp = tmp[target_len - overlap:]
                    else:
                        start = 0
                        while start < len(s):
                            end = min(start + target_len, len(s))
                            chunk = s[start:end]
                            chunks.append(chunk if start == 0 else (chunks[-1][-overlap:] + chunk))
                            start = end
                        tmp = ""
                else:
                    tmp = nxt
            if tmp:
                chunks.append(tmp)
            continue

        nxt = ("\n\n".join(buf + [p])).strip()
        if len(nxt) > target_len:
            flush_buf()
            if chunks:
                tail = chunks[-1][-overlap:]
                buf = [tail + "\n\n" + p]
            else:
                buf = [p]
            if len(buf[0]) > target_len:
                s = buf[0]
                chunks.append(s[:target_len])
                buf = [s[target_len - overlap:]]
        else:
            buf.append(p)

    flush_buf()
    return chunks

def build_cognitive_tag(prompt: str) -> str:
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    N = 96
    psi = np.exp(1j * np.linspace(0, 2 * np.pi, N))
    align = np.tanh(rng.normal(size=N)).astype(np.float64)
    x = np.arange(N)
    kernel = np.exp(-(np.subtract.outer(x, x) ** 2) / (2 * (18.0 ** 2)))
    def _evolve_field(psi, align, kernel, dt: float = 1.0, lam: float = 0.008, decay: float = 0.001):
        dpsi = dt * (-kernel @ psi + align * psi) - decay * psi
        return psi + dpsi
    for _ in range(3):
        psi = _evolve_field(psi, align, kernel)
    B = np.eye(N, dtype=np.complex128) + 1j * np.triu(np.ones((N, N))) * 0.01
    B = (B + B.T.conj()) / 2.0
    E = float(np.real(np.einsum("i,ij,j->", np.conj(psi), B, psi)))
    tag = f"⟨ψ⟩={np.mean(psi).real:.3f}|E={E:.3f}"
    return f"[cog:{tag}]"

def _llama_call_safe(llm_inst, **p):
    import inspect
    if "mirostat_mode" in p and "mirostat" not in p:
        p["mirostat"] = p.pop("mirostat_mode")
    sig = inspect.signature(llm_inst.__call__)
    allowed = set(sig.parameters.keys())
    if "max_tokens" in p and "max_tokens" not in allowed and "n_predict" in allowed:
        p["n_predict"] = p.pop("max_tokens")
    p = {k: v for k, v in p.items() if k in allowed}
    return llm_inst(**p)

def tokenize_and_generate(
    chunk: str,
    token: str,
    max_tokens: int,
    chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    stop: Optional[list[str]] = None,
    images: Optional[list[bytes]] = None,
) -> Optional[str]:
    try:
        if stop is None:
            stop = ["[/cleared_response]"]
        logit_bias = {}
        try:
            for bad in ("system:", "assistant:", "user:"):
                toks = llm.tokenize(bad.encode("utf-8"), add_bos=False)
                if toks:
                    logit_bias[int(toks[0])] = -2.0
        except Exception:
            pass
        base_prompt = f"[{token}] {chunk}"
        params = {
            "prompt": base_prompt,
            "max_tokens": min(max_tokens, chunk_size),
            "temperature": float(max(0.2, min(1.5, temperature))),
            "top_p": float(max(0.2, min(1.0, top_p))),
            "repeat_penalty": 1.08,
            "mirostat_mode": 2,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "logit_bias": logit_bias,
            "stop": stop,
            "top_k": 40,
        }
        if images:
            params_vis = params.copy()
            params_vis["images"] = images
            out = _llama_call_safe(llm, **params_vis)
        else:
            out = _llama_call_safe(llm, **params)
        if isinstance(out, dict) and out.get("choices"):
            ch = out["choices"][0]
            return ch.get("text") or ch.get("message", {}).get("content", "")
        return ""
    except Exception as e:
        logger.error(f"Error in tokenize_and_generate: {e}")
        return None

def is_code_like(text: str) -> bool:
    if not text:
        return False
    if re.search(r'\b(def|class|import|from|return|if|else|elif|for|while|try|except|with|lambda)\b', text):
        return True
    if re.search(r'[{[()}\]]', text) and re.search(r'=\s*|::|->|=>', text):
        return True
    indented = sum(1 for ln in text.splitlines() if re.match(r'^\s{4,}|\t', ln))
    return indented >= 3

def determine_token(chunk: str, memory: str, max_words_to_check=500) -> str:
    combined_chunk = f"{memory} {chunk}"
    if not combined_chunk:
        return "[attention]"
    if is_code_like(combined_chunk):
        return "[code]"
    words = word_tokenize(combined_chunk)[:max_words_to_check]
    tagged_words = pos_tag(words)
    pos_counts = Counter(tag[:2] for _, tag in tagged_words)
    most_common_pos, _ = pos_counts.most_common(1)[0]
    if most_common_pos == 'VB':
        return "[action]"
    elif most_common_pos == 'NN':
        return "[subject]"
    elif most_common_pos in ['JJ', 'RB']:
        return "[description]"
    else:
        return "[general]"

def fetch_relevant_info(chunk: str, client: weaviate.Client, user_input: str) -> str:
    try:
        if not user_input:
            return ""
        query_vec = np.array(compute_text_embedding(user_input), dtype=np.float32)
        rotated = fhe_v2._rotate(query_vec)
        bucket = fhe_v2._simhash_bucket(rotated)
        gql = f"""
        {{
            Get {{
                InteractionHistory(
                    where: {{
                        path: ["embedding_bucket"],
                        operator: Equal,
                        valueString: "{bucket}"
                    }}
                    limit: 40
                    sort: {{path:"response_time", order: desc}}
                ) {{
                    user_message
                    ai_response
                    encrypted_embedding
                }}
            }}
        }}
        """
        response = client.query.raw(gql)
        results = response.get('data', {}).get('Get', {}).get('InteractionHistory', [])
        best = None
        best_score = -1.0
        with SecureEnclave() as enclave:
            for obj in results:
                enc_emb = obj.get("encrypted_embedding", "")
                if not enc_emb:
                    continue
                score = fhe_v2.enclave_similarity(enc_emb, query_vec, enclave)
                if score > best_score:
                    best_score = score
                    best = obj
        if not best or best_score <= 0:
            return ""
        user_msg_raw = try_decrypt(best.get("user_message", ""))
        ai_resp_raw  = try_decrypt(best.get("ai_response", ""))
        return f"{user_msg_raw} {ai_resp_raw}"
    except Exception as e:
        logger.error(f"[FHEv2 retrieval] failed: {e}")
        return ""

def llama_generate(
    prompt: str,
    *,
    weaviate_client: Optional[weaviate.Client] = None,
    user_input: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    images: Optional[list[bytes]] = None,
    policy_sample_fn = None,
    init_bias_factor: float = 1.0,
    reservoir: Optional[ChaoticReservoir] = None,
    ph_attention: Optional[PersistentHomologyAttention] = None
) -> Optional[str]:
    cfg = load_config()
    max_tokens = cfg.get('MAX_TOKENS', 2500)
    target_len = cfg.get('CHUNK_SIZE', 358)
    try:
        cog_tag = build_cognitive_tag(prompt)
        prompt = f"{cog_tag}\n{prompt}"
        prompt_chunks = advanced_chunker(prompt, target_len=min(480, max(240, target_len)), overlap=72)
        responses = []
        last_output = ""
        memory = ""
        bias_factor = float(init_bias_factor)

        for i, current_chunk in enumerate(prompt_chunks):
            emb = torch.tensor(compute_text_embedding(current_chunk), dtype=torch.float32)
            rv, ph = 0.0, 0.5
            if reservoir is not None:
                reservoir_state = reservoir(emb)
                rv = float(reservoir_state.mean().item())
            if ph_attention is not None:
                past_embs = []
                for prev in prompt_chunks[max(0, i-5):i]:
                    pv = torch.tensor(compute_text_embedding(prev), dtype=torch.float32)
                    past_embs.append(pv)
                past_mat = torch.stack(past_embs) if past_embs else emb.unsqueeze(0)
                ph = float(ph_attention.attention(emb.numpy(), past_mat.numpy()).mean())

            bias_factor = (1 + rv) * (1 + ph) * max(1e-6, bias_factor)
            if policy_sample_fn:
                sample = policy_sample_fn(bias_factor)
                temperature, top_p = sample["temperature"], sample["top_p"]

            retrieved = fetch_relevant_info(current_chunk, weaviate_client, user_input) if weaviate_client else ""
            geodesic_hint = ""
            try:
                if topo_manifold._graph_built:
                    hops = topo_manifold.geodesic_retrieve(user_input or current_chunk, k=1)
                    if hops:
                        geodesic_hint = f" [ltm_hint:{hops[0]}]"
            except Exception:
                geodesic_hint = ""

            combined_chunk = f"{retrieved} {geodesic_hint} {current_chunk}".strip()
            token = determine_token(combined_chunk, memory)
            output = tokenize_and_generate(
                combined_chunk,
                token,
                max_tokens,
                target_len,
                temperature,
                top_p,
                images=images,
            )
            if output is None:
                logger.error("Failed to generate output for chunk")
                continue
            if i > 0 and last_output:
                overlap = min(len(last_output), len(output), 240)
                # naive overlap trim
                output = output[overlap:]
            memory += output
            responses.append(output)
            last_output = output

        final_response = ''.join(responses)
        return extract_cleared_response(final_response) if final_response else None
    except Exception as e:
        logger.error(f"Error in llama_generate: {e}")
        return None

# ---------- App ----------
class App(customtkinter.CTk):
    @staticmethod
    def _encrypt_field(value: str) -> str:
        try:
            return crypto.encrypt(value if value is not None else "")
        except Exception as e:
            logger.error(f"[encrypt] Failed: {e}")
            return value if value is not None else ""

    @staticmethod
    def _decrypt_field(value: str) -> str:
        if value is None:
            return ""
        try:
            return crypto.decrypt(value)
        except Exception as e:
            logger.warning(f"[decrypt] Could not decrypt: {e}")
            return value

    def __init__(self, user_identifier: str):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.attached_images: list[bytes] = []
        self.response_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        # Weaviate client over HTTP endpoint
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        # UI
        self.setup_gui()
        # policy
        self._policy_lock = threading.RLock()
        self._policy_mtime = None
        self._load_policy_if_needed()
        # periodic tasks
        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)
        self.after(6 * 3600 * 1000, self._schedule_key_mutation)
        try:
            self.bind_all("<Control-v>", self.on_paste_image)
        except Exception as e:
            logger.warning(f"Bind paste failed: {e}")
        # Reservoir & PH-attention
        self.reservoir     = ChaoticReservoir(
            input_dim=AdvancedHomomorphicVectorMemory.DIM,
            reservoir_dim=256,
            spectral_radius=0.95
        )
        self.ph_attention = PersistentHomologyAttention()
        self.last_z = (0.0, 0.0, 0.0)

    # ----- Policy (clean) -----
    def _policy_params_path(self):
        return os.path.join(bundle_dir, "policy_params.json")

    def _load_policy_if_needed(self):
        defaults = {
            "temp_w": 0.0,
            "temp_b": 0.0,
            "temp_log_sigma": -0.7,
            "top_w": 0.0,
            "top_b": 0.0,
            "top_log_sigma": -0.7,
            "learning_rate": 0.05
        }
        path = self._policy_params_path()
        with self._policy_lock:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None
            reload_needed = (not hasattr(self, "pg_params") or (mtime is not None and mtime != self._policy_mtime))
            if reload_needed:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    for key, val in defaults.items():
                        data.setdefault(key, val)
                    self.pg_params = data
                    self._policy_mtime = mtime
                except Exception as e:
                    logger.warning(f"[Policy Load Error] could not load {path}: {e}")
                    self.pg_params = defaults.copy()
                    self._policy_mtime = mtime
            if not hasattr(self, "pg_learning_rate"):
                env_lr = os.getenv("PG_LEARNING_RATE")
                if env_lr is not None:
                    try:
                        lr = float(env_lr)
                    except ValueError:
                        logger.warning(f"[Policy] Invalid PG_LEARNING_RATE='{env_lr}', falling back")
                        lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                else:
                    lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                self.pg_learning_rate = lr

    def _save_policy(self):
        try:
            with open(self._policy_params_path(), "w") as f:
                json.dump(self.pg_params, f, indent=2)
        except Exception as e:
            logger.error(f"[PG] Failed saving policy params: {e}")

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _policy_forward(self, bias_factor: float):
        p = self.pg_params
        t_range = 1.5 - 0.2
        raw_t = p["temp_w"] * bias_factor + p["temp_b"]
        sig_t = self._sigmoid(raw_t)
        mu_t = 0.2 + sig_t * t_range
        p_range = 1.0 - 0.2
        raw_p = p["top_w"] * bias_factor + p["top_b"]
        sig_p = self._sigmoid(raw_p)
        mu_p = 0.2 + sig_p * p_range
        sigma_t = math.exp(p["temp_log_sigma"]) + 1e-4
        sigma_p = math.exp(p["top_log_sigma"]) + 1e-4
        cache = {"sig_t": sig_t, "t_range": t_range, "sig_p": sig_p, "p_range": p_range}
        return mu_t, sigma_t, mu_p, sigma_p, cache

    def _policy_sample(self, bias_factor: float):
        mu_t, sigma_t, mu_p, sigma_p, cache = self._policy_forward(bias_factor)
        t_sample = random.gauss(mu_t, sigma_t)
        p_sample = random.gauss(mu_p, sigma_p)
        t_clip = max(0.2, min(1.5, t_sample))
        p_clip = max(0.2, min(1.0, p_sample))
        return {
            "temperature": t_clip,
            "top_p": p_clip,
            "raw_temperature": t_sample,
            "raw_top_p": p_sample,
            "mu_t": mu_t, "sigma_t": sigma_t,
            "mu_p": mu_p, "sigma_p": sigma_p,
            "cache": cache
        }

    # ----- Memory aging -----
    def memory_aging_scheduler(self):
        self.run_long_term_memory_aging()
        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)

    def _weaviate_find_ltm(self, phrase: str):
        safe_phrase = sanitize_for_graphql_string(phrase, max_len=256)
        gql = f"""
        {{
          Get {{
            LongTermMemory(
              where: {{ path:["phrase"], operator:Equal, valueString:"{safe_phrase}" }}
              limit: 1
            ) {{
              phrase
              score
              crystallized_time
              _additional {{ id }}
            }}
          }}
        }}
        """
        try:
            resp = self.client.query.raw(gql)
            items = resp.get("data", {}).get("Get", {}).get("LongTermMemory", [])
            if not items:
                return None, None, None
            obj = items[0]
            return (obj["_additional"]["id"], float(obj.get("score", 0.0)), obj.get("crystallized_time", ""))
        except Exception as e:
            logger.error(f"[Aging] _weaviate_find_ltm failed: {e}")
            return None, None, None

    def _weaviate_update_ltm_score(self, uuid_str: str, new_score: float):
        try:
            self.client.data_object.update(
                class_name="LongTermMemory",
                uuid=uuid_str,
                data_object={"score": new_score}
            )
        except Exception as e:
            logger.error(f"[Aging] update score failed for {uuid_str}: {e}")

    def _weaviate_delete_ltm(self, uuid_str: str):
        try:
            self.client.data_object.delete(class_name="LongTermMemory", uuid=uuid_str)
        except Exception as e:
            logger.error(f"[Aging] delete failed for {uuid_str}: {e}")

    def run_long_term_memory_aging(self):
        try:
            now = datetime.utcnow()
            purged_any = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("""SELECT phrase, score, COALESCE(aging_last, last_updated) AS ts, crystallized
                               FROM memory_osmosis
                               WHERE crystallized=1""")
                rows = cur.fetchall()
                for phrase, score, ts, crystallized in rows:
                    if not ts:
                        continue
                    try:
                        base_dt = datetime.fromisoformat(ts.replace("Z", ""))
                    except Exception:
                        continue
                    delta_days = max(0.0, (now - base_dt).total_seconds() / 86400.0)
                    if delta_days <= 0:
                        continue
                    half_life = AGING_T0_DAYS + AGING_GAMMA_DAYS * math.log(1.0 + max(score, 0.0))
                    if half_life <= 0:
                        continue
                    decay_factor = 0.5 ** (delta_days / half_life)
                    new_score = score * decay_factor
                    uuid_str, _, _ = self._weaviate_find_ltm(phrase)
                    if new_score < AGING_PURGE_THRESHOLD:
                        purged_any = True
                        if uuid_str:
                            self._weaviate_delete_ltm(uuid_str)
                        cur.execute("""UPDATE memory_osmosis
                                       SET crystallized=0, score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        logger.info(f"[Aging] Purged crystallized phrase '{phrase}' (decayed to {new_score:.3f}).")
                    else:
                        cur.execute("""UPDATE memory_osmosis
                                       SET score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        if uuid_str:
                            self._weaviate_update_ltm_score(uuid_str, new_score)
                conn.commit()
            if purged_any:
                topo_manifold.rebuild()
        except Exception as e:
            logger.error(f"[Aging] run_long_term_memory_aging failed: {e}")

    # ----- UI / Events -----
    def setup_gui(self):
        self.title("Dyson Sphere Quantum Oracle")
        # center window
        window_width = 1920
        window_height = 1080
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        # layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

        # logo
        try:
            logo_photo = tk.PhotoImage(file=logo_path)
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo, text="")
            self.logo_label.image = logo_photo
        except Exception:
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Logo")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # image preview
        self.image_label = customtkinter.CTkLabel(self.sidebar_frame, text="(no image)")
        self.image_label.grid(row=1, column=0, padx=20, pady=10)

        # main chat
        self.text_box = customtkinter.CTkTextbox(
            self, bg_color="black", text_color="white", border_width=0, height=360, width=50,
            font=customtkinter.CTkFont(size=23)
        )
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # input frame
        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)

        self.input_textbox = tk.Text(
            self.input_textbox_frame,
            font=("Roboto Medium", 12),
            bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            relief="flat",
            height=1
        )
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")

        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)

        self.attach_button = customtkinter.CTkButton(self, text="Attach Image", command=self.on_attach_image)
        self.attach_button.grid(row=3, column=2, padx=(0, 10), pady=(20, 20), sticky="nsew")

        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)

        # settings
        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.username_label = customtkinter.CTkLabel(self.settings_frame, text="Username:")
        self.username_label.grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = customtkinter.CTkEntry(self.settings_frame, width=120, placeholder_text="Enter username")
        self.username_entry.insert(0, self.user_id)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.update_username_button = customtkinter.CTkButton(self.settings_frame, text="Update", command=self.update_username)
        self.update_username_button.grid(row=0, column=2, padx=5, pady=5)

        self.model_label = customtkinter.CTkLabel(self.settings_frame, text="Model:")
        self.model_label.grid(row=1, column=0, padx=5, pady=5)
        self.model_selector = customtkinter.CTkComboBox(self.settings_frame, values=["Llama", "HF GPT-OSS", "Both"])
        self.model_selector.set("Llama")
        self.model_selector.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # context
        self.context_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.context_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        fields = [
            ("Latitude:", "latitude_entry", 0, 0),
            ("Longitude:", "longitude_entry", 1, 0),
            ("Weather:", "weather_entry", 2, 0),
            ("Temperature (°F):", "temperature_entry", 3, 0),
            ("Last Song:", "last_song_entry", 4, 0),
        ]
        for label_text, attr_name, row, col in fields:
            customtkinter.CTkLabel(self.context_frame, text=label_text).grid(row=row, column=col, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200)
            setattr(self, attr_name, entry)
            span = 3 if col == 0 else 1
            entry.grid(row=row, column=col+1, columnspan=span, padx=5, pady=5)

        customtkinter.CTkLabel(self.context_frame, text="Event Type:").grid(row=4, column=0, padx=5, pady=5)
        self.event_type = customtkinter.CTkComboBox(self.context_frame, values=["Lottery", "Sports", "Politics", "Crypto", "Custom"])
        self.event_type.set("Sports")
        self.event_type.grid(row=4, column=1, columnspan=3, padx=5, pady=5)

        customtkinter.CTkLabel(self.context_frame, text="Chain Depth:").grid(row=5, column=0, padx=5, pady=5)
        self.chain_depth = customtkinter.CTkComboBox(self.context_frame, values=["1", "2", "3", "4", "5"])
        self.chain_depth.set("2")
        self.chain_depth.grid(row=5, column=1, columnspan=3, padx=5, pady=5)

        self.chaos_toggle = customtkinter.CTkSwitch(self.context_frame, text="Inject Entropy")
        self.chaos_toggle.select()
        self.chaos_toggle.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        self.emotion_toggle = customtkinter.CTkSwitch(self.context_frame, text="Emotional Alignment")
        self.emotion_toggle.select()
        self.emotion_toggle.grid(row=6, column=2, columnspan=2, padx=5, pady=5)

        game_fields = [
            ("Game Type:", "game_type_entry", "e.g. Football"),
            ("Team Name:", "team_name_entry", "e.g. Clemson Tigers"),
            ("Opponent:", "opponent_entry", "e.g. Notre Dame"),
            ("Game Date:", "game_date_entry", "YYYY-MM-DD"),
        ]
        for idx, (label, attr, placeholder) in enumerate(game_fields):
            customtkinter.CTkLabel(self.context_frame, text=label).grid(row=7 + idx, column=0, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200, placeholder_text=placeholder)
            setattr(self, attr, entry)
            entry.grid(row=7 + idx, column=1, columnspan=3, padx=5, pady=5)

    def update_username(self):
        new_username = self.username_entry.get()
        if new_username:
            self.user_id = new_username
            logger.info(f"Username updated to: {self.user_id}")
        else:
            logger.info("Please enter a valid username.")

    def _preview_image(self, img_bytes: bytes):
        try:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            ph = tk.PhotoImage(data=b64)  # works for PNG/GIF best
            self.image_label.configure(image=ph, text="")
            self.image_label.image = ph
        except Exception:
            self.image_label.configure(text="(image attached)", image=None)
            self.image_label.image = None

    def on_attach_image(self):
        try:
            import tkinter.filedialog as fd
            paths = fd.askopenfilenames(
                title="Select image(s)",
                filetypes=[("Images", "*.png *.gif *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")]
            )
            if not paths:
                return
            new_imgs = []
            for pth in paths:
                try:
                    with open(pth, "rb") as f:
                        ib = f.read()
                    if len(ib) > 15 * 1024 * 1024:
                        logger.warning(f"Skipping large image: {pth}")
                        continue
                    new_imgs.append(ib)
                except Exception as e:
                    logger.warning(f"Failed reading image {pth}: {e}")
            if not new_imgs:
                return
            self.attached_images = new_imgs
            self._preview_image(self.attached_images[0])
        except Exception as e:
            logger.error(f"on_attach_image error: {e}")

    def on_paste_image(self, event=None):
        try:
            txt = ""
            try:
                txt = self.clipboard_get()
            except Exception:
                txt = ""
            ib: bytes | None = None
            if txt and txt.strip().startswith("data:image"):
                try:
                    _, b64 = txt.split(",", 1)
                    ib = base64.b64decode(b64)
                except Exception:
                    ib = None
            elif txt and os.path.exists(txt.strip()):
                pth = txt.strip()
                try:
                    with open(pth, "rb") as f:
                        ib = f.read()
                except Exception:
                    ib = None
            else:
                for candidate in (x.strip() for x in txt.splitlines() if x.strip()):
                    if os.path.exists(candidate):
                        try:
                            with open(candidate, "rb") as f:
                                ib = f.read()
                            break
                        except Exception:
                            pass
            if ib:
                if len(ib) > 15 * 1024 * 1024:
                    logger.warning("Pasted image too large; ignored.")
                    return
                self.attached_images = [ib]
                self._preview_image(ib)
        except Exception as e:
            logger.error(f"on_paste_image error: {e}")

    def process_queue(self):
        try:
            while True:
                msg = self.response_queue.get_nowait()
                if msg['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {msg['data']}\n")
                    self.text_box.see(tk.END)
                elif msg['type'] == 'image':
                    self.image_label.configure(image=msg['data'])
                    self.image_label.image = msg['data']
                    self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)

    def on_submit(self, event=None):
        raw_input = self.input_textbox.get("1.0", tk.END)
        user_input = sanitize_text(raw_input, max_len=4000).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)
            self.executor.submit(self.generate_response, user_input)
            self.after(100, self.process_queue)
        return "break"

    # ----- Business logic -----
    def get_weather_sync(self, lat, lon) -> Tuple[Optional[float], Optional[int]]:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            weather_code = int(current.get("weathercode", 0))
            return temp_c, weather_code
        except Exception as e:
            logger.error(f"[Weather] Fetch failed: {e}")
            return None, None

    def generate_quantum_state(self, rgb=None) -> str:
        if rgb is None or not isinstance(rgb, tuple) or len(rgb) != 3:
            rgb = (128, 128, 128)
        try:
            cpu = psutil.cpu_percent(interval=None) / 100.0
            cpu = max(cpu, 0.05)
            r, g, b = [min(1.0, max(0.0, c / 255)) for c in rgb]
            try:
                lat = float(self.latitude_entry.get() or 0.0)
                lon = float(self.longitude_entry.get() or 0.0)
            except Exception:
                lat, lon = 0.0, 0.0
            try:
                user_temp_f = float(self.temperature_entry.get() or 70.0)
            except ValueError:
                user_temp_f = 70.0
            temp_f, weather_code, is_live = fetch_live_weather(lat, lon, user_temp_f)
            if weather_code in {1, 2, 3}:
                weather_scalar = 0.3
            elif weather_code >= 61:
                weather_scalar = 0.7
            else:
                weather_scalar = 0.0
            tempo = 120
            z0_hist, z1_hist, z2_hist = self.last_z
            z0, z1, z2 = rgb_quantum_gate(
                r, g, b,
                cpu_usage=cpu,
                tempo=tempo,
                lat=lat,
                lon=lon,
                temperature_f=temp_f,
                weather_scalar=weather_scalar,
                z0_hist=z0_hist,
                z1_hist=z1_hist,
                z2_hist=z2_hist
            )
            self.last_z = (z0, z1, z2)
            source = "Live" if is_live else "Manual"
            return (
                f"[QuantumGate+Coherence] RGB={rgb} │ CPU={cpu*100:.1f}% │ "
                f"Z=({z0:.3f}, {z1:.3f}, {z2:.3f}) │ "
                f"GPS=({lat:.3f},{lon:.3f}) │ Temp={temp_f:.1f}°F ({source}) │ "
                f"WeatherCode={weather_code}"
            )
        except Exception as e:
            logger.error(f"Error in generate_quantum_state: {e}")
            return "[QuantumGate] error"

    def fetch_relevant_info_internal(self, chunk: str) -> Tuple[str, str]:
        safe_chunk = sanitize_for_graphql_string(chunk, max_len=256)
        query = f"""
        {{
            Get {{
                InteractionHistory(
                    nearText: {{ concepts: ["{safe_chunk}"], certainty: 0.7 }}
                    limit: 1
                ) {{
                    user_message
                    ai_response
                    response_time
                }}
                LongTermMemory(
                    nearText: {{ concepts: ["{safe_chunk}"], certainty: 0.65 }}
                    limit: 1
                ) {{
                    phrase
                    score
                    crystallized_time
                }}
            }}
        }}
        """
        try:
            response = self.client.query.raw(query)
            data_root = response.get('data', {}).get('Get', {})
            hist_list = data_root.get('InteractionHistory', [])
            if hist_list:
                interaction = hist_list[0]
                user_msg_raw = self._decrypt_field(interaction.get('user_message', ''))
                ai_resp_raw = self._decrypt_field(interaction.get('ai_response', ''))
                user_msg = sanitize_text(user_msg_raw, max_len=4000)
                ai_resp = sanitize_text(ai_resp_raw, max_len=4000)
                return user_msg, ai_resp
            ltm_list = data_root.get('LongTermMemory', [])
            if ltm_list:
                phrase_obj = ltm_list[0]
                phrase = sanitize_text(phrase_obj.get('phrase', ''), max_len=400)
                return phrase, ""
            return "", ""
        except Exception as e:
            logger.error(f"Weaviate query failed: {e}")
            return "", ""

    def process_response_and_store_in_weaviate(self, user_message: str, ai_response: str):
        try:
            response_blob = TextBlob(ai_response)
            keywords = response_blob.noun_phrases
            sentiment = response_blob.sentiment.polarity
            enhanced_keywords = set()
            for phrase in keywords:
                enhanced_keywords.update(phrase.split())
            interaction_object = {
                "user_message": self._encrypt_field(user_message),
                "ai_response":  self._encrypt_field(ai_response),
                "keywords":     list(enhanced_keywords),
                "sentiment":    sentiment
            }
            interaction_uuid = str(uuid.uuid4())
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )
            logger.info(f"Interaction stored in Weaviate: {interaction_uuid}")
        except Exception as e:
            logger.error(f"Error storing interaction in Weaviate: {e}")

    def create_interaction_history_object(self, user_message: str, ai_response: str) -> str:
        interaction_object = {
            "user_message": self._encrypt_field(user_message),
            "ai_response":  self._encrypt_field(ai_response)
        }
        object_uuid = uuid.uuid4().hex
        try:
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            logger.info(f"Interaction history object created: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating object in Weaviate: {e}")
        return object_uuid

    def extract_keywords(self, message: str) -> List[str]:
        try:
            nouns = TextBlob(message).noun_phrases
            return list(nouns)
        except Exception as e:
            logger.warning(f"extract_keywords error: {e}")
            return []

    def retrieve_past_interactions(self, user_input: str, result_queue: queue.Queue):
        try:
            keywords = self.extract_keywords(user_input)
            concepts_query = ' '.join(keywords)
            user_message, ai_response = self.fetch_relevant_info_internal(concepts_query)
            if user_message and ai_response:
                combo = f"{user_message} {ai_response}"
                from summa import summarizer
                summarized = summarizer.summarize(combo) or combo
                sentiment = TextBlob(summarized).sentiment.polarity
                processed = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized,
                    "sentiment": sentiment
                }
                result_queue.put([processed])
            else:
                result_queue.put([])
        except Exception as e:
            logger.error(f"retrieve_past_interactions error: {e}")
            result_queue.put([])

    def quantum_memory_osmosis(self, user_message: str, ai_response: str):
        try:
            phrases_user = set(self.extract_keywords(user_message))
            phrases_ai = set(self.extract_keywords(ai_response))
            all_phrases = {p.strip().lower() for p in (phrases_user | phrases_ai) if len(p.strip()) >= 3}
            if not all_phrases:
                return
            now_iso = datetime.utcnow().isoformat() + "Z"
            newly_crystallized = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("UPDATE memory_osmosis SET score = score * ?, last_updated = ?", (DECAY_FACTOR, now_iso))
                for phrase in all_phrases:
                    cur.execute("SELECT score, crystallized FROM memory_osmosis WHERE phrase = ?", (phrase,))
                    row = cur.fetchone()
                    if row:
                        score, crystallized = row
                        new_score = score + 1.0
                        cur.execute("UPDATE memory_osmosis SET score=?, last_updated=? WHERE phrase=?", (new_score, now_iso, phrase))
                    else:
                        new_score = 1.0
                        crystallized = 0
                        cur.execute(
                            "INSERT INTO memory_osmosis (phrase, score, last_updated, crystallized) VALUES (?, ?, ?, 0)",
                            (phrase, new_score, now_iso)
                        )
                    if new_score >= CRYSTALLIZE_THRESHOLD and not crystallized:
                        try:
                            self.client.data_object.create(
                                data_object={"phrase": phrase, "score": new_score, "crystallized_time": now_iso},
                                class_name="LongTermMemory",
                            )
                            cur.execute("UPDATE memory_osmosis SET crystallized=1, aging_last=? WHERE phrase=?", (now_iso, phrase))
                            newly_crystallized = True
                            logger.info(f"[Osmosis] Crystallized '{phrase}' (score={new_score:.2f}).")
                        except Exception as we:
                            logger.error(f"[Osmosis] Weaviate store failed: {we}")
                conn.commit()
            if newly_crystallized:
                topo_manifold.rebuild()
        except Exception as e:
            logger.error(f"[Osmosis] Error: {e}")

    def generate_response(self, user_input: str) -> None:
        try:
            if not user_input:
                logger.error("User input is empty.")
                return
            self._load_policy_if_needed()
            save_user_message(self.user_id, user_input)

            use_context  = "[pastcontext]" in user_input.lower()
            show_reflect = "[reflect]"     in user_input.lower()
            cleaned_input = sanitize_text(
                re.sub(r"\[chain[_\-]?depth=\d+\]", "", user_input.replace("[pastcontext]", ""), flags=re.IGNORECASE),
                max_len=2048
            )
            try:
                chain_depth = int(self.chain_depth.get())
            except Exception:
                chain_depth = 1

            blob = TextBlob(cleaned_input)
            user_polarity     = blob.sentiment.polarity
            user_subjectivity = blob.sentiment.subjectivity

            past_context = ""
            if use_context:
                qres = queue.Queue()
                self.retrieve_past_interactions(cleaned_input, qres)
                interactions = qres.get()
                if interactions:
                    past_context = "\n".join(f"User: {i['user_message']}\nAI:   {i['ai_response']}" for i in interactions)[-1500:]

            try: lat = float(self.latitude_entry.get().strip() or "0")
            except: lat = 0.0
            try: lon = float(self.longitude_entry.get().strip() or "0")
            except: lon = 0.0
            try: temp_f = float(self.temperature_entry.get().strip() or "72")
            except: temp_f = 72.0

            weather     = self.weather_entry.get().strip() or "Clear"
            song        = self.last_song_entry.get().strip() or "None"
            game_type   = (self.event_type.get() or "Custom").strip().lower()

            rgb = extract_rgb_from_text(cleaned_input)
            r, g, b  = [c / 255.0 for c in rgb]
            cpu_load = psutil.cpu_percent(interval=None) / 100.0
            z0, z1, z2 = rgb_quantum_gate(r, g, b, cpu_usage=cpu_load)
            self.generate_quantum_state(rgb=rgb)
            self.last_z = (z0, z1, z2)

            bias_factor = (z0 + z1 + z2) / 3.0
            entropy     = float(np.std([r, g, b, cpu_load]))
            time_lock   = datetime.utcnow().isoformat()

            stage_memo = ""
            stage_evidence = ""
            mm_imgs = self.attached_images[:] if self.attached_images else None

            for stage in range(1, max(1, chain_depth) + 1):
                planner_prompt = f"""
[SYS] You are a careful planner. Do NOT reveal this section to the user.
Work in short bullets. Focus on correctness and practical next steps.

[CTX]
User Q: {cleaned_input}
lat={lat:.4f}, lon={lon:.4f}, weather="{weather}", temp_f={temp_f}, song="{song}", time="{time_lock}"
z=({z0:.4f},{z1:.4f},{z2:.4f}), sentiment=(pol:{user_polarity:.3f},subj:{user_subjectivity:.3f})
memory_active={'Yes' if past_context else 'No'}, bias={bias_factor:.4f}, entropy={entropy:.4f}
Prior Memo (stage {stage-1}): {stage_memo or '(none)'}
[/CTX]

[TASK]
Stage {stage}/{max(1,chain_depth)}.
- If images are provided, extract only details relevant to answering the question (objects, text, relations).
- Identify missing info and assumptions.
- Produce a brief, structured plan for THIS stage only (3–6 bullets).
Return your plan ONLY inside these tags:

[cleared_response]
- Key visual/ textual evidence:
- Assumptions:
- Stage plan:
[/cleared_response]
""".strip()

                candidate_rollouts = []
                for _ in range(4):
                    sample = self._policy_sample(bias_factor)
                    temp, topp = float(sample["temperature"]), float(sample["top_p"])
                    response = llama_generate(
                        planner_prompt,
                        weaviate_client=self.client,
                        user_input=cleaned_input,
                        temperature=temp,
                        top_p=topp,
                        images=mm_imgs,
                        policy_sample_fn=self._policy_sample,
                        init_bias_factor=1.0,
                        reservoir=self.reservoir,
                        ph_attention=self.ph_attention
                    )
                    if not response:
                        continue
                    cf1 = llama_generate(planner_prompt, weaviate_client=self.client, user_input=cleaned_input,
                                         temperature=max(0.2, 0.8 * temp), top_p=topp,
                                         images=mm_imgs, reservoir=self.reservoir, ph_attention=self.ph_attention)
                    cf2 = llama_generate(planner_prompt, weaviate_client=self.client, user_input=cleaned_input,
                                         temperature=min(1.5, 1.2 * temp), top_p=min(1.0, 1.1 * topp),
                                         images=mm_imgs, reservoir=self.reservoir, ph_attention=self.ph_attention)
                    task_reward  = evaluate_candidate(response, user_polarity, cleaned_input)
                    total_reward = compute_meal_js_reward(response, cf1, cf2, user_polarity, cleaned_input, gamma=JS_LAMBDA)
                    sample.update({
                        "response":        response,
                        "counterfactuals": [cf1 or "", cf2 or ""],
                        "reward":          total_reward,
                        "bias_factor":     bias_factor,
                        "prompt_used":     planner_prompt,
                        "meal_penalty":    task_reward - total_reward
                    })
                    candidate_rollouts.append(sample)

                if not candidate_rollouts:
                    self.response_queue.put({'type': 'text', 'data': '[Reasoner: No viable stage rollouts]'})
                    return

                best_stage = mbr_select_with_js(candidate_rollouts, js_reg_lambda=JS_LAMBDA)
                stage_text = best_stage["response"]
                try:
                    from summa import summarizer
                    stage_memo = (summarizer.summarize(stage_text) or stage_text)[:1200]
                except Exception:
                    stage_memo = stage_text[:1200]
                stage_evidence += f"\n[Stage {stage}] {stage_memo}"

            dyson_prompt = f"""
[SYS] Be concise, truthful, and safe. Do NOT repeat or restate this prompt.
Only write inside the tags below.

[CTX]
Q: {cleaned_input}
lat={lat:.4f}, lon={lon:.4f}, weather=\"{weather}\", temp_f={temp_f}, song=\"{song}\", time=\"{time_lock}\"
z=({z0:.4f},{z1:.4f},{z2:.4f}), sentiment=(pol:{user_polarity:.3f},subj:{user_subjectivity:.3f})
memory_active={'Yes' if past_context else 'No'}, bias={bias_factor:.4f}, entropy={entropy:.4f}
Reasoning Evidence (compressed):
{stage_evidence.strip()[:1800]}
[/CTX]

[TASK] Provide the most helpful final answer. Use short bullets/steps if useful.
If uncertain, say so briefly. No prefaces, no prompt quotes.

[cleared_response]
<<WRITE ANSWER HERE>>
[/cleared_response]
""".strip()

            candidate_rollouts = []
            for _ in range(4):
                sample = self._policy_sample(bias_factor)
                temp, topp = float(sample["temperature"]), float(sample["top_p"])
                response = llama_generate(
                    dyson_prompt,
                    weaviate_client=self.client,
                    user_input=cleaned_input,
                    temperature=temp,
                    top_p=topp,
                    images=mm_imgs,
                    policy_sample_fn=self._policy_sample,
                    init_bias_factor=1.0,
                    reservoir=self.reservoir,
                    ph_attention=self.ph_attention
                )
                if not response:
                    continue
                cf1 = llama_generate(dyson_prompt, weaviate_client=self.client, user_input=cleaned_input,
                                     temperature=max(0.2, 0.8 * temp), top_p=topp,
                                     images=mm_imgs, reservoir=self.reservoir, ph_attention=self.ph_attention)
                cf2 = llama_generate(dyson_prompt, weaviate_client=self.client, user_input=cleaned_input,
                                     temperature=min(1.5, 1.2 * temp), top_p=min(1.0, 1.1 * topp),
                                     images=mm_imgs, reservoir=self.reservoir, ph_attention=self.ph_attention)
                task_reward  = evaluate_candidate(response, user_polarity, cleaned_input)
                total_reward = compute_meal_js_reward(response, cf1, cf2, user_polarity, cleaned_input, gamma=JS_LAMBDA)
                sample.update({
                    "response":        response,
                    "counterfactuals": [cf1 or "", cf2 or ""],
                    "reward":          total_reward,
                    "bias_factor":     bias_factor,
                    "prompt_used":     dyson_prompt,
                    "meal_penalty":    task_reward - total_reward
                })
                candidate_rollouts.append(sample)

            if not candidate_rollouts:
                self.response_queue.put({'type': 'text', 'data': '[Dyson QPU: No viable rollouts]'})
                return

            best = mbr_select_with_js(candidate_rollouts, js_reg_lambda=JS_LAMBDA)
            answer_output = best["response"]
            reasoning_trace = f"""
[DYSON NODE SELF-REFLECTION TRACE]
Reward Score:       {best.get('reward',0):.3f}
MEAL-JS Penalty:    {best.get('meal_penalty',0):.4f}
Sampling Strategy:  T={best.get('temperature',1.0):.2f}, TopP={best.get('top_p',0.9):.2f}
Sentiment Target:   {user_polarity:.3f}
Z-Field Alignment:  μ={bias_factor:.4f}
Entropy:            {entropy:.4f}
Memory Context:     {'Yes' if past_context else 'No'}
Chain Depth:        {chain_depth}
""".strip()

            if show_reflect:
                answer_output += "\n\n" + reasoning_trace

            save_bot_response(self.bot_id, answer_output)
            self.response_queue.put({'type': 'text', 'data': answer_output})

            try:
                self.quantum_memory_osmosis(cleaned_input, answer_output)
            except Exception as e:
                logger.warning(f"[Memory Osmosis Error] {e}")

            try:
                self.client.data_object.create(
                    class_name="ReflectionLog",
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.user_id}-{time_lock}")),
                    data_object={
                        "type":             "prediction",
                        "user_id":          self.user_id,
                        "bot_id":           self.bot_id,
                        "query":            cleaned_input,
                        "response":         answer_output,
                        "reasoning_trace":  reasoning_trace,
                        "prompt_snapshot":  best.get("prompt_used", dyson_prompt),
                        "meal_js":          best.get("mbr_score",0.0),
                        "z_state":          json.dumps({"z0":float(z0),"z1":float(z1),"z2":float(z2)}),
                        "entropy":          float(entropy),
                        "bias_factor":      float(bias_factor),
                        "temperature":      float(best.get("temperature",1.0)),
                        "top_p":            float(best.get("top_p",0.9)),
                        "sentiment_target": float(user_polarity),
                        "timestamp":        time_lock,
                        "lotto_game":       game_type
                    }
                )
            except Exception as e:
                logger.warning(f"[Weaviate Log Error] {e}")

        except Exception as e:
            logger.error(f"[Gamma-13X Fatal Error] {e}")
            self.response_queue.put({'type': 'text', 'data': f"[Dyson QPU Error] {e}"})

    def _schedule_key_mutation(self):
        try:
            crypto.self_mutate_key(population=5, noise_sigma=18.0, alpha=1.0, beta=2.5)
        except Exception as e:
            logger.error(f"[SelfMutateKey] periodic failure: {e}")
        self.after(6 * 3600 * 1000, self._schedule_key_mutation)

# ---------- Main ----------
if __name__ == "__main__":
    try:
        user_id = "gray00"
        app_gui = App(user_id)
        init_db()
        setup_weaviate_schema(app_gui.client)
        start_ultimate(app_gui, interval_h=12.0)   # schedule before mainloop
        app_gui.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")

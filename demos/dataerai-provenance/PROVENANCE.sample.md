# Provenance — hls4ml-selftest

- **Mode:** dry-run (recorded locally)
- **Lineage run:** `0840976d-ec9f-5087-a319-8b009fc9366c`
- **Tools:** numpy 1.26.4, sklearn 1.7.2

## Lineage graph

```mermaid
graph LR
  n0["Jet-tagging dataset<br/><i>dataset</i>"]
  n1["Baseline MLP (part1)<br/><i>model</i>"]
  n2["HLS project — baseline (part1)<br/><i>hls_project</i>"]
  n3["Pruned MLP (part3)<br/><i>model</i>"]
  n4["HLS project — pruned (part3)<br/><i>hls_project</i>"]
  n5["Quantized QKeras MLP (part4)<br/><i>model</i>"]
  n6["HLS project — quantized (part4)<br/><i>hls_project</i>"]
  n7["HLS project — PYNQ-Z2 (part7)<br/><i>hls_project</i>"]
  n8["HLS inference predictions (part7)<br/><i>array</i>"]
  n9["Pruned CNN (part6)<br/><i>model</i>"]
  n10["Quantized pruned CNN (part6)<br/><i>model</i>"]
  n11["BDT — XGBoost (part5)<br/><i>model</i>"]
  n12["Conifer HLS project — BDT (part5)<br/><i>hls_project</i>"]
  n13["Symbolic regression (part8)<br/><i>artifact</i>"]
  n1 -->|trained_on| n0
  n2 -->|derived_from (synthesize)| n1
  n3 -->|derived_from (prune)| n1
  n3 -->|trained_on| n0
  n4 -->|derived_from (synthesize)| n3
  n5 -->|derived_from (quantize)| n3
  n5 -->|trained_on| n0
  n6 -->|derived_from (synthesize)| n5
  n7 -->|derived_from (synthesize)| n5
  n8 -->|evaluated_on (infer)| n7
  n10 -->|derived_from (quantize)| n9
  n11 -->|trained_on| n0
  n12 -->|derived_from (synthesize)| n11
```

## Preserved assets

| Artifact | Kind | Asset ID | DID |
|---|---|---|---|
| Jet-tagging dataset | dataset | `be8761ea-612c-53b2-a506-3fc551f77224` | `did:dataerai:asset:be8761ea-612c-53b2-a506-3fc551f77224` |
| Baseline MLP (part1) | model | `13732c2d-e12a-584a-8522-283467036fdc` | `did:dataerai:asset:13732c2d-e12a-584a-8522-283467036fdc` |
| HLS project — baseline (part1) | hls_project | `58b0aadd-2820-5e22-bb9d-88b17a4d77b3` | `did:dataerai:asset:58b0aadd-2820-5e22-bb9d-88b17a4d77b3` |
| Pruned MLP (part3) | model | `02f37a33-4e42-5a8c-bbca-95a6892f1915` | `did:dataerai:asset:02f37a33-4e42-5a8c-bbca-95a6892f1915` |
| HLS project — pruned (part3) | hls_project | `d5f29acb-4abe-5115-b33f-5236f1311799` | `did:dataerai:asset:d5f29acb-4abe-5115-b33f-5236f1311799` |
| Quantized QKeras MLP (part4) | model | `2fed29c6-8eec-5f1a-961e-10a96ff9799d` | `did:dataerai:asset:2fed29c6-8eec-5f1a-961e-10a96ff9799d` |
| HLS project — quantized (part4) | hls_project | `1ef5c921-a20e-5eef-a6a1-98b8f6e45392` | `did:dataerai:asset:1ef5c921-a20e-5eef-a6a1-98b8f6e45392` |
| HLS project — PYNQ-Z2 (part7) | hls_project | `8faf9455-2458-52d2-8376-0a4486632364` | `did:dataerai:asset:8faf9455-2458-52d2-8376-0a4486632364` |
| HLS inference predictions (part7) | array | `3fd0c093-3ed3-506a-85b2-ff2a16a8f14d` | `did:dataerai:asset:3fd0c093-3ed3-506a-85b2-ff2a16a8f14d` |
| Pruned CNN (part6) | model | `dc357b76-9747-50ee-ac3f-e64f6a58a7e2` | `did:dataerai:asset:dc357b76-9747-50ee-ac3f-e64f6a58a7e2` |
| Quantized pruned CNN (part6) | model | `0a088493-41ca-5f3c-8043-299e8f2e2239` | `did:dataerai:asset:0a088493-41ca-5f3c-8043-299e8f2e2239` |
| BDT — XGBoost (part5) | model | `4fd12668-49a1-5ed5-b5ed-f6c1a287e96f` | `did:dataerai:asset:4fd12668-49a1-5ed5-b5ed-f6c1a287e96f` |
| Conifer HLS project — BDT (part5) | hls_project | `bb5e8ef5-9a84-5759-a53e-1ef33771d7a4` | `did:dataerai:asset:bb5e8ef5-9a84-5759-a53e-1ef33771d7a4` |
| Symbolic regression (part8) | artifact | `04eb7494-daf4-5a0c-ab61-ea1d69be431f` | `did:dataerai:asset:04eb7494-daf4-5a0c-ab61-ea1d69be431f` |

## Citations (DIDs)

- **Jet-tagging dataset** — `did:dataerai:asset:be8761ea-612c-53b2-a506-3fc551f77224`
- **Baseline MLP (part1)** — `did:dataerai:asset:13732c2d-e12a-584a-8522-283467036fdc`
- **HLS project — baseline (part1)** — `did:dataerai:asset:58b0aadd-2820-5e22-bb9d-88b17a4d77b3`
- **Pruned MLP (part3)** — `did:dataerai:asset:02f37a33-4e42-5a8c-bbca-95a6892f1915`
- **HLS project — pruned (part3)** — `did:dataerai:asset:d5f29acb-4abe-5115-b33f-5236f1311799`
- **Quantized QKeras MLP (part4)** — `did:dataerai:asset:2fed29c6-8eec-5f1a-961e-10a96ff9799d`
- **HLS project — quantized (part4)** — `did:dataerai:asset:1ef5c921-a20e-5eef-a6a1-98b8f6e45392`
- **HLS project — PYNQ-Z2 (part7)** — `did:dataerai:asset:8faf9455-2458-52d2-8376-0a4486632364`
- **HLS inference predictions (part7)** — `did:dataerai:asset:3fd0c093-3ed3-506a-85b2-ff2a16a8f14d`
- **Pruned CNN (part6)** — `did:dataerai:asset:dc357b76-9747-50ee-ac3f-e64f6a58a7e2`
- **Quantized pruned CNN (part6)** — `did:dataerai:asset:0a088493-41ca-5f3c-8043-299e8f2e2239`
- **BDT — XGBoost (part5)** — `did:dataerai:asset:4fd12668-49a1-5ed5-b5ed-f6c1a287e96f`
- **Conifer HLS project — BDT (part5)** — `did:dataerai:asset:bb5e8ef5-9a84-5759-a53e-1ef33771d7a4`
- **Symbolic regression (part8)** — `did:dataerai:asset:04eb7494-daf4-5a0c-ab61-ea1d69be431f`

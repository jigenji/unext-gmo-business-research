# GPU Cloud Pricing Time Series Data / GPUクラウド価格時系列データ

調査日: 2026-03-10
目的: Exponential decay functionのフィッティングおよびconfidence interval推定に使用するための、GPUレンタル・ハードウェア価格の時系列データ収集

---

## 1. H100 SXM レンタル価格時系列 (2023-2026)

### 1.1 Silicon Data H100 Rental Index (SDH100RT) - 日次ベンチマーク

SDH100RTはBloomberg terminalで公開される世界初の日次GPUレンタル価格index。30以上のsourceから350万data pointを集約し、on-demand priceを正規化。

| 日付 | SDH100RT ($/GPU-hr) | 備考 |
|------|---------------------|------|
| 2023-Q1 | ~$8.00-$10.00 | 市場推定値。極端な供給不足、AI labsとhyperscalersの需要集中 |
| 2023-Q2 | ~$7.50-$10.00 | AWS P5 ~$7.50/GPU-hr, GCP A3 ~$11.00/GPU-hr |
| 2023-Q3 | ~$7.00-$9.00 | 供給不足継続、waitlist 8-12ヶ月 |
| 2023-Q4 | ~$6.00-$8.00 | NVIDIA出荷増加開始 |
| 2024-Q1 | ~$5.00-$8.00 | 大規模colocation施設でのH100 cluster稼働開始 |
| 2024-Q2 | ~$4.00-$6.00 | GPU marketplace出現、price discoveryの開始 |
| 2024-Q3 | ~$3.50-$5.00 | 新規providerの参入加速 |
| 2024-09-01 | $3.06 | SDH100RT index高値記録 (公式data point) |
| 2024-Q4 | ~$2.85-$3.50 | Peak比64%下落 |
| 2025-Q1 | ~$2.50-$3.00 | 市場正常化進行 |
| 2025-Q2 | ~$2.36-$2.50 | AWS 44%値下げ (2025-06-01), 市場全体のreset |
| 2025-05-27 | $2.37 | SDH100RT公式data point |
| 2025-06-19 | $2.36 | SDH100RT公式data point。2024-09-01の$3.06から23%下落 |
| 2025-Q3 (Aug) | $2.22-$2.26 | Volatility極低 (~1.7%) |
| 2025-Q3 (Sep) | $2.13-$2.16 | Index最安値圏 |
| 2025-Q4 | ~$2.00-$2.20 | |
| 2025-12-09 | $2.00 | Spike前の底値 |
| 2026-01-06 | $2.20 | 4週間で10%上昇 (最大の短期spike) |

Source: [Silicon Data H100 Rental Index](https://www.silicondata.com/products/silicon-index), [H100 Rental Price Over Time](https://www.silicondata.com/blog/h100-rental-price-over-time), [H100 Price Spike](https://www.silicondata.com/blog/h100-price-spike)

### 1.2 H100 Provider別 On-Demand価格 (2025年後半時点)

| Provider | $/GPU-hr | 備考 |
|----------|----------|------|
| Azure (ND H100 v5) | $6.98 | East US, 値下げなし |
| AWS EC2 (P5) | $3.93 | 2025-06値下げ後 (旧$7.57) |
| GCP (A3-High) | $3.00 | 旧~$11.00から大幅値下げ |
| Lambda Labs | $2.99 | |
| CoreWeave | ~$2.50-$3.00 | InfiniBand付き |
| GMI Cloud | $2.10 | |
| RunPod (on-demand) | $2.49 | |
| RunPod (spot) | $1.99 | |
| Vast.ai | ~$1.87 | Marketplace最安値圏 |
| Hyperbolic | $1.49 | 最安on-demand |

### 1.3 H100 Spot/Preemptible価格 (2025年後半)

| Provider | $/GPU-hr | 備考 |
|----------|----------|------|
| GCP Spot (A3-High) | $2.25 | |
| AWS Spot | ~$2.50 | |
| Azure Spot | ~$8.75-$9.38 | ND H100 v5 ~$70-75/hr ÷ 8 GPUs |
| RunPod Spot | $1.99 | |
| 市場最安spot | $0.73 | (ThunderCompute集計) |

### 1.4 H100 Reserved/Committed価格

| Provider | $/GPU-hr | Commitment期間 | Discount率 |
|----------|----------|---------------|-----------|
| AWS Savings Plan (1yr) | ~$2.10-$2.50 | 1年 | ~44% off on-demand |
| AWS Savings Plan (3yr) | ~$1.90-$2.10 | 3年 | ~45% off on-demand |
| GCP CUD | ~$2.40 | 1年 | ~20% sustained use |
| Azure Reserved (1yr) | ~$4.90 | 1年 | ~30% off on-demand |
| Azure Reserved (3yr) | ~$3.50-$4.20 | 3年 | ~40-50% off on-demand |
| Hyperstack Reserved | $1.90 | 契約ベース | ~21% off on-demand $2.40 |
| Lambda Reserved | $1.85-$1.89 | 大口契約 | ~38% off on-demand $2.99 |
| CoreWeave Reserved | ~$2.46 | 長期 | ~60% off on-demand ~$6.16 |

Source: [IntuitionLabs H100 Comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison), [Jarvislabs H100 Price Guide](https://docs.jarvislabs.ai/blog/h100-price)

---

## 2. A100 80GB SXM レンタル価格時系列 (2020-2025)

### 2.1 A100 Cloud Rental Price History

| 時期 | 価格帯 ($/GPU-hr) | 備考 |
|------|-------------------|------|
| 2020-Q3 (Launch) | $3.00-$5.00+ | GCP/AWS初期on-demand |
| 2021 | $3.00-$4.00 | Hyperscaler標準価格 |
| 2022 | $2.50-$3.50 | AI boom前の安定期 |
| 2023-Q1 | $2.00-$4.00 | H100需要spilloverでA100にも需要 |
| 2023-Q2 | $2.00-$3.50 | |
| 2024-Q1 | $1.50-$3.00 | H100供給増でA100需要減 |
| 2024-Q2 | $1.20-$2.50 | |
| 2024-Q3 | $1.00-$2.00 | |
| 2025-Q1 | $0.75-$1.50 | |
| 2025-Q2 | $0.75-$1.35 | AWS A100 33%値下げ (2025-06) |
| 2025-Q4 | $0.50-$1.00 | Sub-$1が主流化 |
| 2026予測 | <$1.00 | Nearly free (<$1) 予測 |

### 2.2 A100 Provider別価格 (2025年時点)

| Provider | $/GPU-hr | 備考 |
|----------|----------|------|
| AWS (P4d) | ~$2.20 | 2025-06値下げ後 (旧~$3.30, 33%cut) |
| GCP | ~$2.48 | |
| Hyperstack | $1.35 | On-demand |
| Lambda | $1.29 | |
| RunPod | ~$1.20 | |
| Vast.ai | ~$0.75 | Marketplace |

Source: [Jarvislabs A100 Price Guide](https://docs.jarvislabs.ai/blog/a100-price), [ThunderCompute Market Trends](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends)

---

## 3. V100 レンタル価格時系列 (2018-2025)

| 時期 | 価格帯 ($/GPU-hr) | 備考 |
|------|-------------------|------|
| 2018-2020 | $3.00-$3.50 | Hyperscaler標準 |
| 2021-2022 | $2.50-$3.06 | AWS/Azure $3.06、依然主要AI GPU |
| 2023-2024 | $1.00-$2.50 | A100/H100への移行進行 |
| 2025 (Hyperscaler) | $2.48-$3.06 | GCP/AWS/Azure (Legacy価格変更なし) |
| 2025 (Specialty) | $0.10-$0.50 | 専門provider、最安$0.10/hr |
| 2025-09 | V100 VMs引退開始 | Azure NCv3 series引退 (発売約7.5年後) |

Source: [GetDeploying V100](https://getdeploying.com/gpus/nvidia-v100)

---

## 4. H200 レンタル価格 (2024-2025)

H200は2024年後半から利用可能。141GB HBM3e VRAM搭載。

### 4.1 H200 Provider別 On-Demand価格 (2025年時点)

| Provider | $/GPU-hr | 備考 |
|----------|----------|------|
| Azure | $10.60 | 最高価格帯 |
| AWS (P5en) | $4.33 | 2025-06に25%値下げ後 |
| RunPod | $3.99 | |
| Lambda | $3.79 | |
| Jarvislabs | $3.80 | Single GPU対応 |
| GCP Spot (A3-High) | $3.72 | Preemptible |
| GMI Cloud | $2.50 | 最安on-demand圏 |
| Vast.ai | ~$2.43 | Marketplace |
| 市場最安 | ~$1.50 | 可用性による |

### 4.2 H200 価格推移

| 時期 | 価格帯 | 備考 |
|------|--------|------|
| 2024-Q4 (初期) | $5.00-$12.00+ | 初期リリース、limited availability |
| 2025-Q1 | $4.00-$8.00 | 供給拡大開始 |
| 2025-Q2 | $3.50-$5.50 | AWS 25%値下げ |
| 2025-Q3-Q4 | $2.50-$4.33 | 競争激化、H100比$1-2プレミアム |

Source: [Jarvislabs H200 Guide](https://docs.jarvislabs.ai/blog/h200-price), [ThunderCompute H200 Pricing](https://www.thundercompute.com/blog/nvidia-h200-pricing), [Cerebrium H200 Cost](https://www.cerebrium.ai/articles/how-much-does-a-h200-cost-2025-guide)

---

## 5. B200/GB300 Blackwell レンタル価格 (2025-2026)

### 5.1 B200 Cloud Pricing (Early Availability)

| Provider | $/GPU-hr | 備考 |
|----------|----------|------|
| RunPod | $4.99 | On-demand |
| Nebius | $5.50 | HGX B200 |
| Modal (Serverless) | $6.25 | Bursty workload向け |
| 市場平均 | $3.00-$5.00 | Independent providers |
| Spot最安 | ~$0.15 | Partial GPU/spot |

### 5.2 B200 vs H100 プレミアム

- B200 on-demand: ~$4-6/hr vs H100 on-demand: ~$2-3/hr → **約2倍のプレミアム**
- ただしB200は4-15x performance/cost improvement (per-token basis)
- B200本格展開でH100は10-20%追加値下げの予測 (2026)

### 5.3 GB300 NVL72

- CoreWeaveで利用可能 (価格は要問い合わせ)
- 72 Blackwell Ultra GPUs + 36 Grace CPUs のrack-scale architecture
- 具体的な$/GPU-hr データは未公開

Source: [Modal B200 Pricing](https://modal.com/blog/nvidia-b200-pricing), [RunPod B200](https://www.runpod.io/gpu-models/b200), [Northflank B200 Cost](https://northflank.com/blog/how-much-does-an-nvidia-b200-gpu-cost)

---

## 6. GPU Hardware購入価格時系列

### 6.1 新品購入価格

| GPU | 発売年 | 発売時価格 | 2024価格 | 2025-26価格 | 備考 |
|-----|--------|-----------|----------|------------|------|
| V100 32GB | 2017 | ~$8,000-$10,000 | ~$3,000-$5,000 | ~$2,500-$5,500 (中古) | 生産終了 |
| A100 40GB PCIe | 2020 | $10,000-$12,000 | ~$8,000-$10,000 | $7,000-$10,000 (新品) | |
| A100 80GB SXM | 2020 | $15,000-$20,000 | ~$12,000-$18,000 | $7,000-$15,000 (新品), $4,000-$9,000 (中古) | |
| H100 80GB PCIe | 2022 | ~$30,000-$33,000 | $25,000-$40,000 | $25,000-$30,000 | |
| H100 80GB SXM | 2022 | ~$33,000-$36,000 | $35,000-$50,000 (eBay高値) | $30,000-$40,000 | |
| H200 141GB | 2024 | $30,000-$40,000 | $30,000-$40,000 | $30,000-$40,000 | H100比15-20%プレミアム |
| B200 192GB SXM | 2025 | $30,000-$50,000 | - | $45,000-$50,000 (OEM), $30,000-$40,000 (bulk) | 製造コスト~$6,400 |

### 6.2 H100 中古/二次市場価格推移

| 時期 | 中古価格帯 | 新品比率 | 備考 |
|------|-----------|---------|------|
| 2023-04 | $39,995-$46,000 (eBay) | >100% (プレミアム) | 供給不足によるeBay高値。CNBC報道 |
| 2023-H2 | $35,000-$45,000 | ~100-110% | 継続的な供給不足 |
| 2024-H1 | $30,000-$40,000 | ~85% (refurbished) | 供給改善開始 |
| 2024-H2 | $18,000-$25,000 (使用1年未満), $12,000-$18,000 (1-2年) | ~61% (used中央値) | CoreWeave契約期限切れ分は95%でrebook |
| 2025 | $7,000-$12,000 (2年超), $12,000-$18,000 (1-2年) | ~69% (used中央値) | Blackwell移行加速 |
| 2025-H2/2026 | ~$6,000 (一部報告) | ~15-20% | 85%下落との報告 (GuruFocus) |

Source: [CNBC H100 eBay](https://www.cnbc.com/2023/04/14/nvidias-h100-ai-chips-selling-for-more-than-40000-on-ebay.html), [Silicon Data Market Value Trends](https://www.silicondata.com/use-cases/h100-gpu-market-value-trends/), [Oplexa Resale Analysis](https://oplexa.com/product/nvidia-h100-gpu-resale/)

### 6.3 DGX System価格

| System | GPU数 | 価格 | 備考 |
|--------|-------|------|------|
| DGX A100 | 8x A100 | ~$199,000 | 発売時 |
| DGX H100 | 8x H100 | $300,000-$450,000 | |
| DGX H200 | 8x H200 | $400,000-$500,000 | |
| DGX B200 | 8x B200 | ~$515,410 | Broadberry listing |

---

## 7. Hyperscaler GPU Instance 価格変更イベント

### 7.1 主要価格変更タイムライン

| 日付 | Provider | Instance | 変更内容 | 値下げ率 |
|------|----------|----------|---------|---------|
| 2023-Q2 | AWS | P5 (H100) | 初期価格設定 ~$60.54/hr (8GPU) | - |
| 2023-Q2 | GCP | A3 (H100) | 初期価格設定 ~$88/hr (8GPU, ~$11/GPU-hr) | - |
| 2024 | GCP | A3 | 段階的値下げ開始 | ~20-30% |
| 2025-03 | GCP | A3 | $11.06/GPU-hrから急速値下げ | - |
| 2025-06-01 | AWS | P5 (H100) | On-demand 44%値下げ ($7.57→~$4.10/GPU-hr) | **44%** |
| 2025-06-01 | AWS | P5en (H200) | On-demand 25%値下げ | **25%** |
| 2025-06-01 | AWS | P4d/P4de (A100) | On-demand 33%値下げ | **33%** |
| 2025-06-04 | AWS | P5 | Savings Plans 45%値下げ | **45%** |
| 2025-06-04 | AWS | P5en | Savings Plans 26%値下げ | **26%** |
| 2025-Q3 | GCP | A3-High | ~$3.00/GPU-hr到達 | ~73% (ピーク比) |
| 2025年通年 | Azure | ND H100 v5 | 大幅値下げなし、$6.98/GPU-hr維持 | ~0% |

Source: [AWS Pricing Announcement](https://aws.amazon.com/about-aws/whats-new/2025/06/pricing-usage-model-ec2-instances-nvidia-gpus/), [DCD AWS Price Cuts](https://www.datacenterdynamics.com/en/news/aws-cuts-costs-for-h100-h200-and-a100-instances-by-up-to-45/)

### 7.2 市場構造変化

- 2025年に300以上の新規GPU cloud providerが参入
- Specialist providers (RunPod, Vast.ai等)が$1.80-$1.87/hrを実現
- Hyperscalerは遅れて対応 (AWSは2025-06, Azureは未対応)
- GCPのspot H100は$2.25まで下落

---

## 8. 価格下落モデル・減価償却曲線

### 8.1 GPU FLOP/$ 改善率 (Epoch AI研究)

| 指標 | Doubling Time | 備考 |
|------|-------------|------|
| 全GPU平均 FLOP/s per $ | 2.46年 | 95% CI: 2.24-2.72年 (2006-2021, 470モデル) |
| Top GPU FLOP/s per $ | 2.95年 | 最先端GPUはやや遅い |
| ML Research GPU FLOP/s per $ | 2.07年 | ML用途GPUは改善が速い |
| Moore's Law比較 | ~2.0年 | GPU改善はMoore's Lawよりやや遅い (FP32ベース) |

Source: [Epoch AI GPU Price-Performance](https://epoch.ai/blog/trends-in-gpu-price-performance), [LessWrong GPU Trends](https://www.lesswrong.com/posts/c6KFvQcZggQKZzxr9/trends-in-gpu-price-performance)

### 8.2 H100 Cloud Rental Decay関数パラメータ

実測データからのdecay推定:

```
モデル: P(t) = P_peak * e^(-λt) + P_floor

推定パラメータ:
- P_peak: ~$8.00-$10.00/hr (2023-Q1)
- P_floor: ~$1.65/hr (provider profitability floor)
- 2024-09: $3.06 → 2025-09: $2.13 (12ヶ月で30%下落)
- 2023-Q1→2025-Q4: ~$8.00→$2.00 (約33ヶ月で75%下落)

Exponential decay rate λ推定:
- Peak ($8.00) → $3.06 (18ヶ月): λ ≈ ln(8.00/3.06)/18 ≈ 0.053/月
- $3.06 → $2.13 (12ヶ月): λ ≈ ln(3.06/2.13)/12 ≈ 0.030/月
- 全期間 ($8.00 → $2.00, 33ヶ月): λ ≈ ln(8.00/2.00)/33 ≈ 0.042/月

Floor付きモデル (P_floor = $1.65):
- ($8.00-$1.65) → ($2.00-$1.65) = $6.35 → $0.35, 33ヶ月
- λ_adjusted ≈ ln(6.35/0.35)/33 ≈ 0.087/月

年間下落率: 1 - e^(-0.042*12) ≈ 39.5% (floor無しモデル)
半減期: ln(2)/0.042 ≈ 16.5ヶ月
```

### 8.3 世代間価格下落パターン

| 世代遷移 | 旧世代下落率 | 期間 | 備考 |
|---------|------------|------|------|
| V100 → A100 (2020) | ~50-70% | 3-4年 | 緩やかな下落 |
| A100 → H100 (2022-23) | ~60-80% | 2-3年 | AI boomで一時的にA100再評価 |
| H100 → B200 (2025-26) | 予測10-20% (当初) | 12ヶ月 | B200本格展開後に加速見込み |

### 8.4 ハードウェア減価償却 (企業会計ベース)

| 企業 | 償却期間 | 備考 |
|------|---------|------|
| Microsoft | 5-6年 | GPU延長済み |
| Google | 5-6年 | GPU延長済み |
| Meta | 5-6年 | GPU延長済み |
| NVIDIA推奨 | 4-6年 | Private memo記載 |
| Epoch AI故障モデル | 平均5年 (σ=1.5年) | 正規分布仮定 |

### 8.5 Cloud Rental価格の年間下落率 (FP32 FLOP基準)

- 2019→2025: cloud GPU price per FP32 FLOP が約74%下落 (2025年は2019年の約26%水準)
- 年率換算: 約18-20%/年の下落
- ただしAI demand surge期 (2023) は一時的にspike

### 8.6 Provider Profitability分析

| 指標 | 値 | 備考 |
|------|-----|------|
| H100 provider profitability floor | ~$1.65/hr | これ以下では投資回収不可 |
| Stock market代替投資beat条件 | >$2.85/hr | 機会費用考慮 |
| Break-even at $2.50/hr | 10,000-12,000時間 | 稼働率100%で14-16ヶ月 |
| Commitment pricing | $1.90-$2.10/hr | On-demand比40%+安 |

---

## 9. 予測・アウトルック

### 9.1 短期予測 (2026)

| GPU | 現在価格 | 2026中盤予測 | 備考 |
|-----|---------|------------|------|
| H100 On-demand | $2.00-$3.00 | <$2.00 (universal) | Sub-$2時代到来 |
| A100 On-demand | $0.75-$1.35 | <$1.00 | Nearly free |
| H200 On-demand | $2.50-$4.33 | $2.00-$3.50 | 緩やかな下落 |
| B200 On-demand | $3.00-$5.00 | $2.50-$4.00 | Supply成熟で低下 |
| V100 On-demand | $0.10-$0.50 | <$0.10 | Retirement進行 |

### 9.2 Value Cascade Model

次世代GPU投入による「カスケード効果」:
1. **Tier 1 (Training)**: 新Blackwell GPUが大規模training jobを担当
2. **Tier 2 (Inference/Fine-tuning)**: H100がinference・fine-tuningへ移行
3. **Tier 3 (Standard Inference)**: A100がstandard inferenceへ
4. **Tier 4 (Budget)**: V100が教育・テスト用途へ

このcascadeにより各世代が経済的価値を維持しつつ段階的に価格低下。

---

## 10. Key Data Sources

1. [Silicon Data H100 Rental Index](https://www.silicondata.com/products/silicon-index) - Bloomberg Terminal日次index
2. [Silicon Data - H100 Rental Price Over Time](https://www.silicondata.com/blog/h100-rental-price-over-time)
3. [Silicon Data - H100 Price Spike](https://www.silicondata.com/blog/h100-price-spike)
4. [Silicon Data - GPU Pricing Trends 2026](https://www.silicondata.com/blog/gpu-pricing-trends-2026-what-to-expect-in-the-year-ahead)
5. [Silicon Data - H100 Market Value Trends](https://www.silicondata.com/use-cases/h100-gpu-market-value-trends/)
6. [Introl Blog - GPU Cloud Prices Collapse](https://introl.com/blog/gpu-cloud-price-collapse-h100-market-december-2025)
7. [Epoch AI - GPU Price-Performance Trends](https://epoch.ai/blog/trends-in-gpu-price-performance)
8. [Epoch AI - B200 Cost Breakdown](https://epoch.ai/data-insights/b200-cost-breakdown)
9. [Epoch AI - NVIDIA Chip Production Doubling](https://epoch.ai/data-insights/nvidia-chip-production)
10. [Carmen Li / Medium - Correction in Compute](https://medium.com/@cli_87015/a-correction-in-compute-tracing-the-decline-in-h100-rental-prices-af02da399f5a)
11. [Carmen Li / Medium - Cost per FP32 FLOP](https://medium.com/@cli_87015/the-evolution-of-gpu-pricing-a-deep-dive-into-cost-per-fp32-flop-for-hyperscalers-cbf072b85bb5)
12. [AWS Pricing Announcement (2025-06)](https://aws.amazon.com/about-aws/whats-new/2025/06/pricing-usage-model-ec2-instances-nvidia-gpus/)
13. [DCD - AWS Cuts H100/H200/A100 Prices](https://www.datacenterdynamics.com/en/news/aws-cuts-costs-for-h100-h200-and-a100-instances-by-up-to-45/)
14. [CNBC - H100 eBay $40K](https://www.cnbc.com/2023/04/14/nvidias-h100-ai-chips-selling-for-more-than-40000-on-ebay.html)
15. [SemiAnalysis - ClusterMAX Rating System](https://newsletter.semianalysis.com/p/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus)
16. [IntuitionLabs - H100 Rental Comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
17. [IntuitionLabs - NVIDIA AI GPU Pricing Guide](https://intuitionlabs.ai/articles/nvidia-ai-gpu-pricing-guide)
18. [Tom's Hardware - H100 vs MI300X Pricing](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-h100-ai-gpus-cost-up-to-four-times-more-than-amds-competing-mi300x-amds-chips-cost-dollar10-to-dollar15k-apiece-nvidias-h100-has-peaked-beyond-dollar40000)
19. [Applied Conjectures - GPU Depreciation](https://appliedconjectures.substack.com/p/how-long-do-gpus-last-anyway-a-look)
20. [Cast AI - 2025 GPU Price Report](https://cast.ai/reports/gpu-price/)
21. [LessWrong - GPU Price-Performance](https://www.lesswrong.com/posts/c6KFvQcZggQKZzxr9/trends-in-gpu-price-performance)
22. [Our World in Data - GPU Computational Performance per Dollar](https://ourworldindata.org/grapher/gpu-price-performance)
23. [The Chip Letter - GPU Compute Costs & Huang's Law](https://thechipletter.substack.com/p/gpu-compute-costs-trends-huangs-law)
24. [GMI Cloud - H100 Pricing 2025](https://www.gmicloud.ai/blog/nvidia-h100-gpu-pricing-2025-rent-vs-buy-cost-analysis)
25. [ThunderCompute - AI GPU Rental Market Trends](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends)

---

## 11. Exponential Decay Fitting用サマリーテーブル

### H100 On-Demand Market Average ($/GPU-hr)

decay functionフィッティング用のclean data points:

| 月 (2023-01=0) | 日付 | 価格 ($/hr) | Source/信頼度 |
|----------------|------|------------|-------------|
| 0 | 2023-01 | 9.00 | 市場推定 (中) |
| 3 | 2023-04 | 8.50 | 市場推定 (中) |
| 6 | 2023-07 | 7.50 | AWS/GCP初期価格ベース (中-高) |
| 9 | 2023-10 | 6.50 | 市場推定 (中) |
| 12 | 2024-01 | 5.50 | 市場推定 (中) |
| 15 | 2024-04 | 4.50 | 市場推定 (中) |
| 18 | 2024-07 | 3.80 | 市場推定 (中) |
| 20 | 2024-09 | 3.06 | SDH100RT公式 (高) |
| 24 | 2025-01 | 2.70 | 市場推定 (中-高) |
| 27 | 2025-04 | 2.50 | 市場推定 (中-高) |
| 28 | 2025-05 | 2.37 | SDH100RT公式 (高) |
| 29 | 2025-06 | 2.36 | SDH100RT公式 (高) |
| 31 | 2025-08 | 2.24 | SDH100RT (2.22-2.26) (高) |
| 32 | 2025-09 | 2.15 | SDH100RT (2.13-2.16) (高) |
| 35 | 2025-12 | 2.00 | SDH100RT (spike前) (高) |
| 36 | 2026-01 | 2.20 | SDH100RT (spike後) (高) |

### 推奨Fittingモデル

```python
# Exponential Decay with Floor
# P(t) = (P0 - P_floor) * exp(-lambda * t) + P_floor
#
# 初期推定:
#   P0 = 9.00  (2023-01推定ピーク)
#   P_floor = 1.65  (provider profitability floor)
#   lambda = 0.042/月 (全期間推定)
#
# Alternative: Two-phase decay
#   Phase 1 (2023-01 to 2024-09): lambda_1 ≈ 0.053/月 (急落期)
#   Phase 2 (2024-09 to 2025-12): lambda_2 ≈ 0.030/月 (安定化期)
#
# Confidence Interval:
#   SDH100RT公式data pointの信頼度は高い
#   2023年のdata pointは市場推定値のため不確実性が大きい
#   推奨: bootstrapまたはBayesian approachで信頼区間推定
```

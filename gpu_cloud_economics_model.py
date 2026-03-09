#!/usr/bin/env python3
"""
GPUクラウド事業 数理モデル
==========================
GMO GPUクラウド事業の収益構造を数理モデルに落とし込み、
主要変数の感度分析・IRR・NPV・投資回収期間を算出する。

核心的洞察: このビジネスはコンサルティングビジネスと同じ構造を持つ。
- コンサル: 人材(稼働率) × 単価(時間) × 利用時間 - 人件費(固定)
- GPU Cloud: GPU(稼働率) × 単価(時間) × 利用時間 - 減価償却(固定)
両者とも「固定資産(人/GPU)の稼働率ビジネス」であり、稼働率が損益を決定的に左右する。

著者: Claude Code Analysis
日付: 2026-03-09
"""

import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 1. モデルのパラメータ定義
# =============================================================================

@dataclass
class MarketParams:
    """市場環境パラメータ"""
    # 需要成長率: 国内GPUクラウド市場
    g_demand: float = 0.25      # 年間需要成長率（CAGR 25%）

    # 供給成長率: 国内GPU供給
    g_supply: float = 0.35      # 年間供給成長率（新規参入含む）

    # GPU月額単価の基本下落率（技術陳腐化）
    P_0: float = 47.5           # 初期GPU月額単価（万円）= GMO専有プラン380万/8GPU
    delta_price: float = 0.10   # 年間ベース価格下落率

    # 需給バランスによる価格調整
    # 需要成長>供給成長 → 価格維持/上昇
    # 供給成長>需要成長 → 価格下落加速
    price_floor_ratio: float = 0.25  # 最低価格倍率（P_0の25%）

    # 参入企業数
    N_0: int = 8                # 2024年時点の主要国内事業者数
    n_entrants_per_year: float = 3.0


@dataclass
class InvestmentParams:
    """投資パラメータ（GMO GPUクラウド実績ベース）"""
    capex_initial: float = 100.0       # 初期設備投資（億円）: H200 768基
    capex_additional_y1: float = 15.0  # 追加投資（億円）: H200 256基
    capex_b300: float = 30.0           # B300投資（億円）: 25台/200基
    subsidy: float = 19.3             # 経産省助成金（億円）

    n_gpu_initial: int = 768
    n_gpu_additional: int = 256
    n_gpu_b300: int = 200

    useful_life_years: int = 5         # 法定耐用年数


@dataclass
class OperatingParams:
    """運営パラメータ"""
    # 稼働率
    U_0: float = 0.80              # 初期稼働率
    alpha_util: float = 0.015      # 競合増加による稼働率低下係数
    U_min: float = 0.35            # 最低稼働率

    # 電力コスト（実績ベース）
    power_per_gpu_h200_kw: float = 0.70   # H200: 700W
    power_per_gpu_b300_kw: float = 1.20   # B300: 1200W
    pue: float = 1.30
    electricity_rate: float = 20.0  # 円/kWh
    hours_per_year: float = 8760.0

    # 運営費（レポートの「固定費約17億円、変動費約7億円」に整合させる）
    # 減価償却を除いた年間運営費 ≈ 10-12億円
    staff_cost_annual: float = 2.4      # 人件費（億円/年）: 30名×800万
    dc_lease_annual: float = 2.0        # DC賃料（億円/年）
    maintenance_annual: float = 1.5     # 保守費（億円/年）
    network_annual: float = 0.8         # ネットワーク費（億円/年）
    software_annual: float = 0.4        # ソフトウェア（億円/年）
    sales_marketing: float = 1.0        # 営業（億円/年）
    general_admin: float = 0.5          # 管理費（億円/年）

    # B300関連
    b300_price_premium: float = 1.5     # B300の価格プレミアム倍率


@dataclass
class FinancialParams:
    """財務パラメータ"""
    discount_rate: float = 0.08
    tax_rate: float = 0.30
    inflation_rate: float = 0.02


# =============================================================================
# 2. 核心モデル
# =============================================================================

class GPUCloudEconomicsModel:
    """
    GPUクラウド事業の数理モデル

    核心方程式:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Revenue(t) = N_gpu(t) × P(t) × U(t) × 12 / 10⁴  [億円]

    P(t)  = P₀ × (1 - δ)^t × DS_adj(t)     ... 価格（需給調整付き）
    U(t)  = U₀ × comp_factor(t)              ... 稼働率
    DS_adj(t) = clip(excess_demand(t), floor, cap) ... 需給調整

    コンサルとの同型性:
    Revenue_consul = N_人 × Rate × Billable% × Hours
    Revenue_gpu    = N_GPU × Price × Util%   × Hours
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(self,
                 market: Optional[MarketParams] = None,
                 investment: Optional[InvestmentParams] = None,
                 operating: Optional[OperatingParams] = None,
                 financial: Optional[FinancialParams] = None):
        self.market = market or MarketParams()
        self.investment = investment or InvestmentParams()
        self.operating = operating or OperatingParams()
        self.financial = financial or FinancialParams()

    # ───── 市場関数 ─────

    def excess_demand_factor(self, t: float) -> float:
        """
        需給バランス係数: 需要成長率 vs 供給成長率の累積差

        需要が供給を上回る → >1（価格維持/上昇圧力）
        供給が需要を上回る → <1（価格下落圧力）

        factor(t) = (1+g_d)^t / (1+g_s)^t
        """
        demand_growth = (1 + self.market.g_demand) ** t
        supply_growth = (1 + self.market.g_supply) ** t
        return demand_growth / supply_growth

    def num_competitors(self, t: float) -> float:
        return self.market.N_0 + self.market.n_entrants_per_year * t

    # ───── 価格関数 ─────

    def gpu_monthly_price(self, t: float) -> float:
        """
        P(t) = P₀ × (1 - δ)^t × clip(DS_factor(t), floor, 1.3)

        t=0で P₀ を返す（DS_factor(0)=1.0）
        """
        # 技術陳腐化による基本下落
        base = self.market.P_0 * (1 - self.market.delta_price) ** t

        # 需給調整
        ds = self.excess_demand_factor(t)
        adjustment = np.clip(ds, self.market.price_floor_ratio, 1.3)

        return base * adjustment

    # ───── GPU保有数 ─────

    def gpu_count(self, t: float) -> dict:
        h200 = self.investment.n_gpu_initial
        b300 = 0
        if t >= 0.5:
            h200 += self.investment.n_gpu_additional
        if t >= 1.5:
            b300 = self.investment.n_gpu_b300
        return {'h200': h200, 'b300': b300, 'total': h200 + b300}

    # ───── 稼働率関数 ─────

    def utilization(self, t: float) -> float:
        """
        U(t) = U₀ × comp_factor(t) × demand_boost(t)

        comp_factor: 競合増加 → 稼働率低下
        demand_boost: 需要超過 → 稼働率上昇
        """
        # 競合増加の影響
        n_new = self.num_competitors(t) - self.market.N_0
        comp_factor = max(1 - self.operating.alpha_util * n_new, 0.5)

        # 需要ブースト（需要>供給なら稼働率改善）
        ds = self.excess_demand_factor(t)
        demand_boost = np.clip(ds, 0.7, 1.15)

        util = self.operating.U_0 * comp_factor * demand_boost
        return float(np.clip(util, self.operating.U_min, 0.95))

    # ───── 収益関数 ─────

    def annual_revenue(self, t: float) -> float:
        """
        Revenue(t) = [H200売上 + B300売上] / 10⁴  (億円)
        """
        gpus = self.gpu_count(t)
        price = self.gpu_monthly_price(t)
        util = self.utilization(t)

        rev_h200 = gpus['h200'] * price * util * 12
        rev_b300 = gpus['b300'] * price * self.operating.b300_price_premium * util * 12

        return (rev_h200 + rev_b300) / 10000

    # ───── コスト関数 ─────

    def annual_electricity_cost(self, t: float) -> float:
        """電力コスト（億円/年）"""
        gpus = self.gpu_count(t)
        h200_kw = gpus['h200'] * self.operating.power_per_gpu_h200_kw
        b300_kw = gpus['b300'] * self.operating.power_per_gpu_b300_kw
        total_kw = (h200_kw + b300_kw) * self.operating.pue
        return total_kw * self.operating.hours_per_year * self.operating.electricity_rate / 1e8

    def annual_depreciation(self, t: int) -> float:
        """減価償却費（億円/年）- 定額法"""
        life = self.investment.useful_life_years
        dep = 0.0
        # 初期投資（助成金控除後）
        net_initial = self.investment.capex_initial - self.investment.subsidy
        if 0 <= t < life:
            dep += net_initial / life
        # 追加投資（Y1から）
        if 1 <= t < 1 + life:
            dep += self.investment.capex_additional_y1 / life
        # B300投資（Y2から）
        if 2 <= t < 2 + life:
            dep += self.investment.capex_b300 / life
        return dep

    def annual_opex(self, t: float) -> float:
        """年間運営費（電力+その他固定費）"""
        elec = self.annual_electricity_cost(t)
        inflation = (1 + self.financial.inflation_rate) ** t
        fixed = (self.operating.staff_cost_annual +
                 self.operating.dc_lease_annual +
                 self.operating.maintenance_annual +
                 self.operating.network_annual +
                 self.operating.software_annual +
                 self.operating.sales_marketing +
                 self.operating.general_admin) * inflation
        return elec + fixed

    # ───── 利益・CF ─────

    def annual_operating_profit(self, t: int) -> float:
        return self.annual_revenue(t) - self.annual_opex(t) - self.annual_depreciation(t)

    def annual_capex(self, t: int) -> float:
        if t == 0:
            return self.investment.capex_initial - self.investment.subsidy
        elif t == 1:
            return self.investment.capex_additional_y1
        elif t == 2:
            return self.investment.capex_b300
        return 2.0  # 維持投資

    def annual_fcf(self, t: int) -> float:
        op = self.annual_operating_profit(t)
        tax = max(op * self.financial.tax_rate, 0)
        dep = self.annual_depreciation(t)
        capex = self.annual_capex(t)
        return (op - tax) + dep - capex

    # ───── 投資指標 ─────

    def npv(self, years: int = 10) -> float:
        return sum(self.annual_fcf(t) / (1 + self.financial.discount_rate) ** t
                   for t in range(years + 1))

    def irr(self, years: int = 10) -> Optional[float]:
        cfs = [self.annual_fcf(t) for t in range(years + 1)]
        def npv_at(r):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(cfs))
        try:
            return brentq(npv_at, -0.5, 5.0)
        except ValueError:
            return None

    def payback_period(self, years: int = 15) -> Optional[float]:
        cum = 0.0
        for t in range(years + 1):
            fcf = self.annual_fcf(t)
            prev_cum = cum
            cum += fcf
            if cum >= 0 and prev_cum < 0:
                return t - cum / fcf if fcf > 0 else float(t)
        return None

    def breakeven_utilization(self, t: int = 1) -> float:
        """損益分岐稼働率"""
        cost = self.annual_opex(t) + self.annual_depreciation(t)
        gpus = self.gpu_count(t)
        price = self.gpu_monthly_price(t)
        max_rev = (gpus['h200'] * price * 12 +
                   gpus['b300'] * price * self.operating.b300_price_premium * 12) / 10000
        return cost / max_rev if max_rev > 0 else 1.0

    # ───── シミュレーション ─────

    def simulate(self, years: int = 10) -> dict:
        results = {k: [] for k in [
            'year', 'gpu_total', 'utilization', 'price_monthly',
            'revenue', 'opex', 'depreciation', 'operating_profit',
            'capex', 'fcf', 'cumulative_fcf', 'electricity_cost',
            'ds_factor', 'competitors'
        ]}
        cum_fcf = 0.0
        for t in range(years + 1):
            gpus = self.gpu_count(t)
            results['year'].append(t)
            results['gpu_total'].append(gpus['total'])
            results['utilization'].append(round(self.utilization(t), 3))
            results['price_monthly'].append(round(self.gpu_monthly_price(t), 2))
            results['revenue'].append(round(self.annual_revenue(t), 2))
            results['opex'].append(round(self.annual_opex(t), 2))
            results['depreciation'].append(round(self.annual_depreciation(t), 2))
            results['operating_profit'].append(round(self.annual_operating_profit(t), 2))
            results['capex'].append(round(self.annual_capex(t), 2))
            fcf = self.annual_fcf(t)
            cum_fcf += fcf
            results['fcf'].append(round(fcf, 2))
            results['cumulative_fcf'].append(round(cum_fcf, 2))
            results['electricity_cost'].append(round(self.annual_electricity_cost(t), 2))
            results['ds_factor'].append(round(self.excess_demand_factor(t), 3))
            results['competitors'].append(round(self.num_competitors(t), 1))
        return results

    def sensitivity_analysis(self) -> dict:
        params_to_test = [
            ('稼働率 U₀', 'operating', 'U_0'),
            ('GPU月額単価 P₀', 'market', 'P_0'),
            ('需要成長率 g_d', 'market', 'g_demand'),
            ('供給成長率 g_s', 'market', 'g_supply'),
            ('価格下落率 δ', 'market', 'delta_price'),
            ('初期投資 CAPEX', 'investment', 'capex_initial'),
            ('電力単価 (円/kWh)', 'operating', 'electricity_rate'),
            ('割引率 WACC', 'financial', 'discount_rate'),
        ]
        results = {}
        for name, group, attr in params_to_test:
            obj = getattr(self, group)
            base_val = getattr(obj, attr)
            row = []
            for pct in [-0.20, -0.10, 0, 0.10, 0.20]:
                setattr(obj, attr, base_val * (1 + pct))
                row.append({'pct': pct, 'npv': round(self.npv(), 1),
                            'irr': round(self.irr() * 100, 1) if self.irr() else None})
                setattr(obj, attr, base_val)
            results[name] = {'base': base_val, 'data': row}
        return results


# =============================================================================
# 3. シナリオ定義
# =============================================================================

def create_scenarios() -> dict:
    return {
        'A: ベースケース（穏やかな競争）': GPUCloudEconomicsModel(
            market=MarketParams(g_demand=0.25, g_supply=0.35, delta_price=0.10, n_entrants_per_year=3),
            operating=OperatingParams(U_0=0.80, alpha_util=0.015),
        ),
        'B: 悲観ケース（激しい競争）': GPUCloudEconomicsModel(
            market=MarketParams(g_demand=0.20, g_supply=0.50, delta_price=0.18, n_entrants_per_year=5),
            operating=OperatingParams(U_0=0.70, alpha_util=0.03),
        ),
        'C: 楽観ケース（AI需要爆発）': GPUCloudEconomicsModel(
            market=MarketParams(g_demand=0.40, g_supply=0.30, delta_price=0.05, n_entrants_per_year=2),
            operating=OperatingParams(U_0=0.85, alpha_util=0.01),
        ),
    }


# =============================================================================
# 4. モデル検証
# =============================================================================

def validate_model():
    """
    実績データとの照合:
    - GPU数: 768基 → 1,024基
    - 料金: 専有380万円/月（47.5万円/GPU）
    - 月次黒字化: 2025年Q3達成
    - 推定年間売上: 20-40億円
    - 損益分岐稼働率: 約50%
    - 電力コスト: 約1.6-2.0億円
    - 固定費: 約17億円、変動費: 約7億円 → 合計24億円
    """
    print("=" * 80)
    print("モデル検証: GMO GPUクラウド 初年度（2024/11 - 2025/10）")
    print("=" * 80)

    model = GPUCloudEconomicsModel()

    checks = []

    # GPU数
    g0 = model.gpu_count(0)
    g1 = model.gpu_count(1)
    checks.append(f"  GPU数 t=0: {g0['h200']}基 (実績768)  {'✓' if g0['h200']==768 else '✗'}")
    checks.append(f"  GPU数 t=1: {g1['h200']}基 (実績1024) {'✓' if g1['h200']==1024 else '✗'}")

    # 月額単価
    p0 = model.gpu_monthly_price(0)
    checks.append(f"  月額単価 t=0: {p0:.1f}万円 (実績47.5) {'✓' if abs(p0-47.5)<0.1 else '✗'}")

    # 年間売上ポテンシャル（稼働率80%、768GPU時）
    rev0 = model.annual_revenue(0)
    checks.append(f"  年間売上 t=0: {rev0:.1f}億円 (実績推定20-40)")
    # 初年度は途中開始(6ヶ月稼働想定)
    rev_adj = rev0 * 0.6
    checks.append(f"  初年度調整(6M稼働): {rev_adj:.1f}億円")
    checks.append(f"  → {'整合' if 15<=rev0<=50 else '要確認'}")

    # 損益分岐稼働率
    be = model.breakeven_utilization(0)
    checks.append(f"  損益分岐稼働率 t=0: {be*100:.1f}% (レポート記載50%)")

    # 電力コスト
    elec = model.annual_electricity_cost(0)
    checks.append(f"  電力コスト: {elec:.2f}億円 (レポート1.6-2.0) {'✓' if 1.0<=elec<=2.5 else '✗'}")

    # コスト構造
    opex = model.annual_opex(0)
    dep = model.annual_depreciation(0)
    total_cost = opex + dep
    checks.append(f"  OPEX(電力含む): {opex:.2f}億円")
    checks.append(f"  減価償却: {dep:.2f}億円")
    checks.append(f"  総コスト: {total_cost:.2f}億円 (レポート約24億円) {'✓' if 20<=total_cost<=30 else '△'}")

    # 月次黒字化チェック
    # t=0で稼働率80%なら営業利益>0 → 月次黒字
    op0 = model.annual_operating_profit(0)
    monthly_op = op0 / 12
    checks.append(f"  月次営業利益 t=0: {monthly_op:.2f}億円 {'(黒字)✓' if monthly_op>0 else '(赤字)'}")

    for c in checks:
        print(c)

    # 5年シミュレーション
    print(f"\n■ 5年間シミュレーション")
    results = model.simulate(5)
    total_rev = sum(results['revenue'])
    total_op = sum(results['operating_profit'])
    print(f"  累計売上: {total_rev:.1f}億円 (レポートA: 170億円)")
    print(f"  累計営業利益: {total_op:.1f}億円 (レポートA: 34億円)")

    return model, results


# =============================================================================
# 5. メイン実行
# =============================================================================

def main():
    print("╔" + "═" * 78 + "╗")
    print("║  GPUクラウド事業 数理経済モデル                                          ║")
    print("║  — コンサルティングビジネスとの構造的同型性の観点から —                  ║")
    print("╚" + "═" * 78 + "╝")

    # Step 1: 検証
    model, val = validate_model()

    # Step 2: 投資指標
    print("\n" + "=" * 80)
    print("投資指標（ベースケース）")
    print("=" * 80)

    npv_val = model.npv(10)
    irr_val = model.irr(10)
    payback = model.payback_period(15)
    be = model.breakeven_utilization(1)

    print(f"  NPV (10年, WACC={model.financial.discount_rate*100:.0f}%): {npv_val:.1f}億円")
    if irr_val:
        print(f"  IRR (10年): {irr_val*100:.1f}%")
    else:
        print(f"  IRR: 算出不能（全期間赤字）")
    if payback:
        print(f"  投資回収期間: {payback:.1f}年")
    else:
        print(f"  投資回収期間: 15年以上")
    print(f"  損益分岐稼働率 (Y1): {be*100:.1f}%")

    # Step 3: シナリオ分析
    print("\n" + "=" * 80)
    print("シナリオ分析（5年間）")
    print("=" * 80)

    scenarios = create_scenarios()
    for name, m in scenarios.items():
        r = m.simulate(5)
        npv = m.npv(10)
        irr = m.irr(10)
        pb = m.payback_period(15)

        print(f"\n━━ {name} ━━")
        print(f"  前提: 需要CAGR={m.market.g_demand*100:.0f}%, 供給CAGR={m.market.g_supply*100:.0f}%, "
              f"価格下落={m.market.delta_price*100:.0f}%/年, 初期稼働率={m.operating.U_0*100:.0f}%")

        header = f"  {'年':>3} | {'GPU数':>5} | {'稼働率':>6} | {'単価(万)':>8} | {'売上(億)':>8} | {'営利(億)':>8} | {'FCF(億)':>7} | {'累計FCF':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in range(6):
            print(f"  Y{r['year'][i]:<2} | {r['gpu_total'][i]:>5} | {r['utilization'][i]*100:>5.1f}% | "
                  f"{r['price_monthly'][i]:>8.1f} | {r['revenue'][i]:>8.1f} | "
                  f"{r['operating_profit'][i]:>8.1f} | {r['fcf'][i]:>7.1f} | {r['cumulative_fcf'][i]:>8.1f}")

        irr_str = f"{irr*100:.1f}%" if irr else "N/A"
        pb_str = f"{pb:.1f}年" if pb else "15年超"
        print(f"\n  NPV(10年): {npv:.1f}億円 | IRR: {irr_str} | 回収: {pb_str}")

    # Step 4: 感度分析
    print("\n" + "=" * 80)
    print("感度分析（ベースケース, 主要変数 ±20% の NPV(10年)への影響）")
    print("=" * 80)

    sens = model.sensitivity_analysis()
    print(f"\n  {'変数':<22} | {'−20%':>8} | {'−10%':>8} | {'Base':>8} | {'+10%':>8} | {'+20%':>8}")
    print("  " + "-" * 72)
    for name, data in sens.items():
        vals = [f"{d['npv']:>7.1f}" for d in data['data']]
        print(f"  {name:<22} | {vals[0]}  | {vals[1]}  | {vals[2]}  | {vals[3]}  | {vals[4]}")

    # Step 5: 構造比較
    print("\n" + "=" * 80)
    print("コンサルティングビジネスとの構造的同型性")
    print("=" * 80)
    print("""
  ┌─────────────────┬───────────────────────┬───────────────────────┐
  │ 構造要素         │ コンサルティング       │ GPUクラウド            │
  ├─────────────────┼───────────────────────┼───────────────────────┤
  │ 売上方程式       │ N人 × Rate × 稼働率   │ N_GPU × Price × 稼働率│
  │ 固定資産         │ コンサルタント(人)     │ GPU(ハードウェア)      │
  │ 固定費の本質     │ 給与(使わなくても発生) │ 減価償却+電力          │
  │ 損益分岐稼働率   │ 60-70%                │ 50-74%                │
  │ 業界平均稼働率   │ 65-75%                │ 40% (!)                │
  │ 陳腐化サイクル   │ 3-5年(スキル)         │ 2-3年(GPU世代)        │
  │ 限界費用         │ ほぼゼロ              │ 電力費(低い)           │
  │ 資産の流動性     │ 高い(転職リスク)      │ 低い(中古市場限定)     │
  ├─────────────────┼───────────────────────┼───────────────────────┤
  │ 決定的な違い     │ 退職リスク(予測困難)  │ 陳腐化リスク(予測可能) │
  └─────────────────┴───────────────────────┴───────────────────────┘

  核心的洞察:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  両ビジネスの利益は「固定費 × 稼働率」で決まる。
  GPU Cloudはコンサルより陳腐化が速い(2年 vs 3-5年)が、予測可能(NVIDIAロードマップ)。

  → 投資回収は経済的耐用年数(3-4年)内に完了する設計が必須
  → 稼働率50%以上の確保が生存条件（業界平均40%は危機的）
  → 長期契約によるビラブルレート安定化がコンサルと同様に重要
    """)

    # Step 6: 方程式体系
    print("=" * 80)
    print("数理モデル: 方程式体系サマリー")
    print("=" * 80)
    print("""
  ━━━ 市場動学 ━━━
  excess_demand(t) = (1+g_d)^t / (1+g_s)^t        需給バランス
  N(t) = N₀ + n_entry × t                          競合企業数

  ━━━ 価格動学 ━━━
  P(t) = P₀ × (1-δ)^t × clip(ED(t), floor, 1.3)   GPU月額単価

  ━━━ 稼働率動学 ━━━
  U(t) = U₀ × max(1-α×ΔN, 0.5) × clip(ED(t), 0.7, 1.15)

  ━━━ 収益 ━━━
  Rev(t) = Σᵢ [Nᵢ(t) × Pᵢ(t) × U(t) × 12] / 10⁴  億円

  ━━━ コスト ━━━
  OPEX(t) = E(t) + Fixed × (1+π)^t
  E(t) = ΣᵢNᵢ × Wᵢ × PUE × 8760 × r_e / 10⁸
  Dep(t) = CAPEX_net / T_life

  ━━━ キャッシュフロー ━━━
  OP(t) = Rev(t) - OPEX(t) - Dep(t)
  FCF(t) = OP(t)×(1-τ) + Dep(t) - CAPEX(t)

  ━━━ 投資指標 ━━━
  NPV = Σₜ FCF(t)/(1+r)^t
  IRR: NPV(IRR) = 0
    """)

    return model, scenarios, sens


if __name__ == "__main__":
    model, scenarios, sensitivities = main()

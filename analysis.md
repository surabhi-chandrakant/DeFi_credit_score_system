# DeFi Credit Score Analysis
        
## Overview
- **Total Wallets Analyzed**: 3,497
- **Average Credit Score**: 555.7
- **Median Credit Score**: 531.0
- **Standard Deviation**: 139.6

## Score Distribution

| Score Range | Number of Wallets | Percentage |
|-------------|------------------|------------|
| 0-100 | 11 | 0.3% |
| 100-200 | 20 | 0.6% |
| 200-300 | 67 | 1.9% |
| 300-400 | 173 | 4.9% |
| 400-500 | 735 | 21.0% |
| 500-600 | 1,507 | 43.1% |
| 600-700 | 589 | 16.8% |
| 700-800 | 172 | 4.9% |
| 800-900 | 103 | 2.9% |
| 900-1000 | 102 | 2.9% |

## High Credit Score Wallets (750+)

**Count**: 283 wallets

### Characteristics:
- **Average Transactions**: 217.1
- **Average Volume**: $916,645,075,971,476,075,577,344.00
- **Average Days Active**: 84.2
- **Average Repay Ratio**: 0.15
- **Average Asset Diversity**: 6.5

### Behavioral Patterns:
- High repayment ratios (average 15.2%)
- Consistent transaction patterns
- Low liquidation rates (average 0.31%)
- Diversified asset usage
- Sustained activity over longer periods

## Low Credit Score Wallets (250 and below)

**Count**: 58 wallets

### Characteristics:
- **Average Transactions**: 27.7
- **Average Volume**: $899,492,399,943,047,184,384.00
- **Average Days Active**: 11.0
- **Average Repay Ratio**: 0.17
- **Average Liquidation Ratio**: 0.01

### Risk Indicators:
- Higher liquidation rates (average 1.3%)
- Potential bot-like behavior patterns
- High concentration in single assets
- Irregular transaction patterns
- Shorter activity periods

## Key Findings

1. **Repayment Behavior**: Strong correlation between repay ratio and credit score
2. **Activity Consistency**: Wallets with sustained, regular activity score higher
3. **Asset Diversification**: Using multiple assets indicates sophisticated usage
4. **Volume Patterns**: Extremely high frequency transactions may indicate bot activity
5. **Liquidation Events**: Major negative impact on credit scores

## Scoring Methodology

The credit scoring model combines:
- **Behavioral Analysis** (40%): Transaction patterns, frequencies, ratios
- **Risk Assessment** (30%): Liquidations, suspicious patterns, concentrations
- **Activity Metrics** (20%): Volume, diversity, consistency
- **Time Factors** (10%): Activity duration, transaction timing

Scores range from 0 (highest risk) to 1000 (most reliable), with the distribution designed to reflect real-world credit scoring patterns.

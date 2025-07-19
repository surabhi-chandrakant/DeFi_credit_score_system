# DeFi Credit Scoring System

A comprehensive machine learning solution for assigning credit scores (0-1000) to wallet addresses based on their Aave V2 protocol transaction behavior.

## Overview

This system analyzes DeFi transaction patterns to evaluate wallet creditworthiness, identifying reliable users while flagging potentially risky or bot-like behavior. The scoring model combines behavioral analysis, risk assessment, and activity metrics to generate transparent, actionable credit scores.

## Architecture

### Core Components

1. **Data Ingestion Engine** (`load_data`)
   - Parses JSON transaction data
   - Normalizes nested structures
   - Handles timestamp conversion

2. **Feature Engineering Pipeline** (`engineer_features`)
   - Extracts 20+ behavioral features
   - Calculates risk indicators
   - Generates temporal patterns

3. **Dual Scoring System**
   - **Heuristic Base Scoring**: Rule-based initial scoring using domain expertise
   - **ML Refinement Model**: Random Forest model for score optimization

4. **Analysis & Visualization Engine**
   - Score distribution analysis
   - Behavioral pattern identification
   - Risk segmentation

## Processing Flow

```
Raw JSON Data → Feature Engineering → Base Score Calculation → ML Refinement → Final Scores → Analysis & Export
```

### Step-by-Step Process

1. **Data Loading**: Parse user transaction JSON file
2. **Feature Extraction**: Generate behavioral, temporal, and risk features per wallet
3. **Base Scoring**: Apply heuristic rules based on DeFi best practices
4. **ML Training**: Train Random Forest model on engineered features
5. **Score Refinement**: Combine base scores with ML predictions (70/30 weight)
6. **Analysis Generation**: Create comprehensive behavioral analysis
7. **Export Results**: Output final scores in JSON format

## Feature Categories

### Transaction Volume Features
- Total transaction count and volume
- Average, max, min transaction sizes
- Transaction size consistency

### Behavioral Ratios
- Deposit/Repay/Liquidation ratios
- Asset diversification metrics
- Activity consistency indicators

### Risk Indicators
- Liquidation event frequency
- Bot-like pattern detection (round numbers, high frequency)
- Asset concentration risk

### Temporal Features
- Days active on protocol
- Transaction frequency patterns
- Time consistency metrics

## Scoring Methodology

### Base Score Components (0-1000 scale)

**Positive Indicators (+)**
- High repayment ratio (+200 max)
- Regular deposits (+150 max)
- Transaction diversity (+2 per tx)
- Sustained activity (+20 per log(days))
- Asset diversification (+30 per asset)
- Consistent behavior patterns (+100 max)

**Negative Indicators (-)**
- Liquidation events (-400 max)
- Suspicious same-day activity (-200 max)
- Bot-like round number patterns (-100 max)
- Over-concentration in single asset (-150 max)
- Extreme transaction frequencies (-100 max)

### ML Refinement
- Random Forest model with 100 estimators
- 10-layer maximum depth for interpretability
- Trained on engineered features with base scores as targets
- Final score = 70% base score + 30% ML prediction

## Installation & Usage

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Quick Start
```python
python credit_scorer.py
```

### Input Format
JSON file with transaction records containing:
- `userWallet`: Wallet address
- `action`: Transaction type (deposit, borrow, repay, etc.)
- `amount`: Transaction amount
- `assetSymbol`: Asset type
- `timestamp`: Transaction timestamp
- `actionData`: Additional transaction details

### Output Files
- `wallet_scores.json`: Final credit scores for all wallets
- `analysis.md`: Comprehensive behavioral analysis
- `score_analysis.png`: Visualization charts

## Model Validation

The system includes built-in validation:
- Train/test split for model evaluation
- R² score reporting for prediction accuracy
- Cross-validation of scoring logic
- Sanity checks for score distribution

## Risk Categories

| Score Range | Risk Level | Characteristics |
|-------------|------------|-----------------|
| 800-1000 | Very Low Risk | High repayment rates, consistent activity, diversified usage |
| 650-799 | Low Risk | Good behavioral patterns, moderate activity |
| 500-649 | Medium Risk | Mixed indicators, average protocol usage |
| 350-499 | High Risk | Some concerning patterns, limited positive indicators |
| 0-349 | Very High Risk | Multiple risk flags, potential bot activity |

## Customization

### Adjusting Scoring Parameters
Modify scoring weights in `calculate_base_score()`:
```python
df['score'] += df['repay_ratio'] * 200  # Adjust repayment weight
df['score'] -= df['liquidation_ratio'] * 400  # Adjust liquidation penalty
```

### Adding New Features
Extend `engineer_features()` with additional behavioral metrics:
```python
features['new_feature'] = calculate_new_metric(wallet_data)
```

### Model Tuning
Adjust Random Forest parameters in `train_ml_model()`:
```python
self.model = RandomForestRegressor(
    n_estimators=150,  # Increase for better accuracy
    max_depth=15,      # Increase for more complexity
    random_state=42
)
```

## Interpretability

The scoring model prioritizes transparency:
- **Heuristic Base**: Clear, interpretable rules based on DeFi expertise
- **Feature Importance**: Random Forest provides feature importance rankings
- **Score Breakdown**: Individual contribution of each component
- **Risk Explanations**: Clear rationale for low scores

## Performance Considerations

- **Scalability**: Optimized for 100K+ transaction datasets
- **Memory Usage**: Efficient pandas operations with minimal memory footprint
- **Processing Time**: Vectorized operations for fast feature engineering
- **Model Size**: Lightweight Random Forest for quick predictions

## Future Enhancements

1. **Advanced ML Models**: Gradient boosting, neural networks
2. **Time Series Analysis**: Temporal pattern recognition
3. **Cross-Protocol Analysis**: Multi-protocol behavior synthesis
4. **Real-time Scoring**: Streaming score updates
5. **Explainable AI**: SHAP values for individual predictions

## Contributing

To extend or modify the system:
1. Add new features in `engineer_features()`
2. Adjust scoring logic in `calculate_base_score()`
3. Experiment with different ML models in `train_ml_model()`
4. Enhance analysis in `generate_analysis()`



---

*This system provides a robust foundation for DeFi credit assessment while maintaining transparency and interpretability essential for financial applications.*

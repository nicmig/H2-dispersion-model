# Latent Mass Flow GP: Complete Guide

## Overview

This approach treats **mass flow as a latent (hidden) variable** that must be inferred from sensor observations during operation. The GP learns the physical relationship `f(time, mass_flow, x, y, z) → h2_concentration` during training, then inverts this relationship at runtime.

```
Training:          Operation:
┌──────────┐       ┌──────────────┐      ┌─────────────┐
│  Time    │       │   Time       │      │  Predicted  │
│  Mass    │──────▶│   H2 sensors │─────▶│  Mass Flow  │
│  X,Y,Z   │       │   (observed) │      │  (inferred) │
└──────────┘       └──────────────┘      └─────────────┘
       │                                        │
       ▼                                        ▼
   H2 Conc.                              Full 3D Field
```

## Key Capabilities

### 1. Mass Flow Estimation
- **MAP Estimate**: Most likely mass flow given observations
- **Posterior Distribution**: Full uncertainty quantification
- **Credible Intervals**: 95% confidence bounds on leak rate

### 2. Field Prediction
- **Mean Prediction**: Expected H2 concentration everywhere
- **Total Uncertainty**: Epistemic (mass flow unknown) + Aleatoric (measurement noise)
- **Danger Zone Identification**: Regions above flammability limit with confidence

### 3. Sensor Fault Detection
- **Consistency Checking**: Compare each sensor to GP predictions from others
- **Anomaly Flagging**: Z-score based fault detection
- **Robust Inference**: Automatic downweighting of inconsistent sensors

## Mathematical Framework

### Forward Model (Training)
The GP defines a distribution over functions:
```
f ~ GP(μ, k)

where:
- μ: mean function (constant)
- k: composite kernel = k_time + k_massflow + k_space + k_interaction
- Input: [time, mass_flow, x, y, z]
- Output: h2_volume_fraction
```

### Inverse Problem (Runtime)
Given observations `y_obs` at sensor locations, infer mass flow `m`:

```
p(m | y_obs, t) ∝ p(y_obs | m, t) · p(m)

where:
- p(y_obs | m, t): GP predictive likelihood
- p(m): Prior (e.g., N(0.5, 0.4) based on operational knowledge)
```

### Marginal Field Prediction
Instead of point estimate, integrate over mass flow uncertainty:
```
p(c_field | y_obs) = ∫ p(c_field | m) · p(m | y_obs) dm
```

This gives **properly calibrated uncertainty** that includes:
- Epistemic: Uncertainty about true mass flow
- Aleatoric: Observation noise and model mismatch

## Usage Examples

### Basic Inference

```python
from latent_mass_flow_gp import train_h2_dispersion_gp, LatentMassFlowPredictor
import pandas as pd

# 1. Train model
df = pd.read_csv('data/unified_raw.csv')
model, likelihood = train_h2_dispersion_gp(df, n_epochs=200)

# 2. Create predictor
predictor = LatentMassFlowPredictor(model, likelihood, SENSOR_POSITIONS)

# 3. Runtime inference
sensor_readings = {
    1: 0.08,   # Sensor ID: H2 concentration
    5: 0.12,
    15: 0.03,
}

result = predictor.full_inference(
    time=60.0,
    sensor_readings=sensor_readings,
    grid_points=monitoring_grid,
    prediction_method='marginalize'  # or 'map'
)

print(f"Estimated leak rate: {result.mass_flow_map:.3f} kg/s")
print(f"Danger volume: {result.danger_zones['danger_volume_m3']:.2f} m³")
```

### Sensor Fault Detection

```python
# Check for inconsistent sensors
anomalies = predictor.detect_sensor_faults(
    time=60.0,
    sensor_readings=sensor_readings,
    threshold_sigma=3.0
)

for sensor_id, info in anomalies.items():
    print(f"Sensor {sensor_id} FAULT: observed={info['observed']:.3f}, "
          f"predicted={info['predicted']:.3f} ± {info['predicted_std']:.3f}")
```

### Safety Monitoring Loop

```python
class SafetyMonitor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.history = []
        
    def process_sensor_data(self, timestamp, readings):
        """Process new sensor data and generate alerts."""
        result = self.predictor.full_inference(
            timestamp, readings, self.monitoring_grid
        )
        
        # Track over time
        self.history.append({
            'time': timestamp,
            'mass_flow': result.mass_flow_map,
            'danger_volume': result.danger_zones['danger_volume_m3']
        })
        
        # Generate alerts
        alerts = []
        
        # Danger zone alert
        if result.danger_zones['has_danger']:
            alerts.append({
                'level': 'CRITICAL' if result.field_mean.max() > 0.1 else 'WARNING',
                'type': 'DANGER_ZONE',
                'message': f"Flammable volume: {result.danger_zones['danger_volume_m3']:.1f} m³"
            })
        
        # Sensor fault alert
        anomalies = self.predictor.detect_sensor_faults(timestamp, readings)
        for sid in anomalies:
            alerts.append({
                'level': anomalies[sid]['severity'],
                'type': 'SENSOR_FAULT',
                'sensor': sid
            })
        
        return result, alerts
```

## Inference Methods Comparison

| Method | Speed | Uncertainty | Use Case |
|--------|-------|-------------|----------|
| **MAP** | Fast (~10ms) | Underestimates | Real-time control |
| **Marginalize** | Medium (~100ms) | Accurate | Safety decisions |
| **MCMC** | Slow (~1s) | Most accurate | Post-incident analysis |

## Kernel Structure Details

```python
# Time kernel: Matern 3/2 for smooth evolution
k_time = σ²_time · Matern_3/2(time / l_time)

# Mass flow kernel: Matern 5/2 for monotonic response
k_mass = σ²_mass · Matern_5/2(mass_flow / l_mass)

# Space kernel: ARD Matern 5/2 for anisotropic dispersion
k_space = σ²_space · Matern_5/2(
    √[(x-x')/l_x]² + [(y-y')/l_y]² + [(z-z')/l_z]²
)

# Interaction: captures time-space coupling (e.g., plume spread over time)
k_int = σ²_int · RBF(time / l_t) · RBF(||position - position'|| / l_s)
```

Learned lengthscales provide physical insight:
- `l_time`: Characteristic time for concentration changes
- `l_x, l_y, l_z`: Dispersion distances (may show more spread in y than x/z)

## Computational Performance

With GPyTorch's fast variational approximations:

| Grid Size | Inference Time | Memory |
|-----------|----------------|--------|
| 1,000 points | ~20 ms | ~50 MB |
| 10,000 points | ~100 ms | ~200 MB |
| 100,000 points | ~500 ms | ~1 GB |

## Safety Applications

### 1. Early Leak Detection
**Challenge**: Detect leaks before dangerous concentrations build up.

**Solution**: Track mass flow estimate over time. Sudden increase indicates new leak.

```python
if result.mass_flow_map > 1.5 * historical_average:
    trigger_alert("POSSIBLE_NEW_LEAK", confidence=result.mass_flow_posterior_std)
```

### 2. Ventilation Control
**Challenge**: Optimize ventilation to prevent accumulation while saving energy.

**Solution**: Use predicted danger volume to modulate ventilation rate.

```python
if result.danger_zones['danger_volume_m3'] > threshold:
    increase_ventilation(flow_rate=result.mass_flow_map * safety_factor)
```

### 3. Evacuation Planning
**Challenge**: Determine when and where to evacuate.

**Solution**: Predict time until flammability limit reached at each location.

```python
# Extrapolate forward in time
future_times = np.arange(0, 600, 10)  # Next 10 minutes
for t_future in current_time + future_times:
    future_result = predictor.predict_field(t_future, ...)
    if future_result.field_mean.max() > 0.1:
        time_to_danger = t_future - current_time
        break
```

### 4. Source Localization
**Challenge**: Find which pipe/valve is leaking.

**Solution**: Infer mass flow and compare to expected values for each source.

```python
for potential_source in sources:
    prior_mean = potential_source.expected_flow
    predictor.set_mass_flow_prior(prior_mean, 0.1)
    result = predictor.full_inference(...)
    likelihood = calculate_posterior_fit(result)
    source_probabilities[potential_source] = likelihood
```

## Validation & Calibration

### Checking Calibration
The uncertainty estimates should be well-calibrated:
- ~68% of observations within ±1σ
- ~95% of observations within ±2σ

```python
def check_calibration(predictor, test_data):
    """Verify that uncertainty estimates are well-calibrated."""
    coverage_1std = []
    coverage_2std = []
    
    for _, row in test_data.iterrows():
        # Predict
        mean, std = predictor.predict_at_location(
            row['time'], row['mass_flow'], 
            (row['x'], row['y'], row['z'])
        )
        
        # Check if observation within bounds
        residual = abs(row['h2_volume_fraction'] - mean)
        coverage_1std.append(residual < std)
        coverage_2std.append(residual < 2*std)
    
    print(f"1σ coverage: {np.mean(coverage_1std):.1%} (expected 68%)")
    print(f"2σ coverage: {np.mean(coverage_2std):.1%} (expected 95%)")
```

## Troubleshooting

### High Uncertainty in Predictions
- **Cause**: Insufficient training data in that region
- **Solution**: Collect more CFD/experimental data, or reduce prediction grid resolution

### Mass Flow Estimate Bias
- **Cause**: Mismatch between training and operational conditions
- **Solution**: Update mass_flow prior based on operational experience

### Slow Inference
- **Cause**: Too many grid points or MCMC samples
- **Solution**: Use 'map' method for control, 'marginalize' only for safety decisions

### Sensor Faults Not Detected
- **Cause**: Threshold too high or sensors all drifting together
- **Solution**: Lower threshold_sigma, use redundant sensing technologies

## Future Extensions

1. **Multi-Source Leaks**: Extend to infer multiple simultaneous leaks
2. **Wind/Weather**: Include environmental conditions as inputs
3. **Time-Varying Leaks**: Model mass flow as function of time (e.g., decreasing as tank empties)
4. **Active Learning**: Automatically request sensor readings where uncertainty is highest
5. **Digital Twin Integration**: Couple with CFD for hybrid physics-ML predictions

## References

- Rasmussen & Williams (2006): Gaussian Processes for Machine Learning
- Alvarez et al. (2012): Kernels for Vector-Valued Functions
- Garnett (2023): Bayesian Optimization (for acquisition functions)

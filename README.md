# H2 Dispersion Model

*Additive Gaussian Processes for hydrogen dispersion, powered by an MLP ensemble leakage-rate estimator.*

This repository models the spatiotemporal evolution of released hydrogen by combining CFD simulations, and experimental sensor data. It combines the leakage-rate estimation via MLP ensemble and the hydrogen dispersion forecasting into a predictive framework.

## Framework Overview

The model pipeline combines an MLP ensemble for leakage-rate estimation with an additive Gaussian Process for spatiotemporal H₂ concentration prediction.

![Framework](GP_EKF_flow.png)

## Experimental Geometry

The CFD and experimental campaigns share the same sensor placement and channel geometry. This is also important to keep in mind for the limitations of this model.

![Sensor Placement](SensorPlacement.png)

## What it does

- **Predicts** H₂ volume fraction across a 3D sensor field at any time using an **additive Gaussian Process**.
- **Infers** the leakage mass flow rate from sparse early-time observations via an **MLP ensemble**.
- **Fuses** CFD scenarios and experimental campaigns into one unified dataset.
- **Anchors** predictions to a fixed release location and aligns them to the release onset.
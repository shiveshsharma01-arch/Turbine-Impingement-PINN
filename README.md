# Axisymmetric Turbine Impingement Cooling  CFD + PINN

> **CFD + PhysicsInformed Machine Learning for reconstructing flow,
> temperature, and wall heattransfer behavior in an aerospacerelevant
> jetimpingement cooling problem.**



## 1. Aerospace Motivation: Jet Impingement Cooling

Gasturbine hotsection components operate in severe thermal
environments. Cooling techniques are therefore used to protect turbine
hardware from excessive heat loading.

**Jet impingement cooling** is one such technique: a highspeed jet is
directed toward a heated surface. The jet forms a strong stagnation
region and then turns and spreads along the wall, creating a
characteristic distribution of local heattransfer rates.

This makes impinging jets a useful canonical problem for studying:

   highspeed flow,
   temperature transport,
   stagnationregion behavior,
   walljet development,
   and local convective heat transfer.

This project models a **2D axisymmetric turbinestyle impinging jet in
ANSYS Fluent** and then trains a **Fourierfeature PhysicsInformed
Neural Network (PINN)** to reconstruct the CFD solution from sparse
observations while enforcing physical constraints.



## 2. Why Add Machine Learning to a CFD Workflow?

For a single steady CFD case, a conventional CFD solution may already be
practical. The motivation for the PINN is different.

A learned physicsconstrained model can become useful when the same
underlying problem must be evaluated repeatedly for:

   designspace exploration,
   optimization,
   rapid surrogate prediction,
   inverse problems,
   sparse sensor/data assimilation,
   and future digitaltwin workflows.

The goal here is therefore **not to claim that the PINN replaces CFD for
one case**.

Instead, this project asks:

> **Can a PINN reconstruct the flow and thermal fields from sparse CFD
> observations while preserving an engineering heattransfer quantity at
> the wall?**

The workflow is:

``` text
              ANSYS FLUENT CFD
                     │
                     ▼
             Exported CFD field
                     │
                     ▼
           Normalization + sampling
                     │
                     ▼
          Fourierfeature PINN
        ┌────────────┼─────────────┐
        │            │             │
    Data loss     PDE loss     WallNu loss
        │            │             │
        └────────────┼─────────────┘
                     ▼
             Reconstructed fields
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       Velocity  Temperature  Nusselt
```



# 3. CFD Baseline

The CFD baseline is a **2D axisymmetric impinging jet**.

### Coordinate convention

   **x** = axial / jet direction
   **y** = radial direction
   **y = 0** = axis of symmetry

### Operating conditions

  Parameter                                            Value
   
  Jet diameter, D                                       2 mm
  Jet velocity, Uj                                   116 m/s
  Jet temperature, Tj                                  700 K
  Heatedwall temperature, Tw                         1100 K
  Temperature difference, ΔT                           400 K
  H/D                                                      4
  Jettowall distance, H                               8 mm
  Domain                                       20 mm × 20 mm
  Nominal Reynolds number                             10,000
  Effective hotair Reynolds number                    3,700
  Fluid                                       Air, ideal gas
  Solver                                        ANSYS Fluent
  Flow model                            Steady RANS, kω SST
  Exported field points                                3,665

The CFD solution provides the reference flow and thermal fields used by
the machinelearning workflow.



# 4. CFD Data Used by the PINN

The final training workflow uses **50% sparse, stratified CFD data**.

The sampling is deliberately not purely uniform. Half of the selected
sparse points are forced into the nearwall region (`x_norm > 0.75`) so
that the network receives more information in the region where thermal
gradients and heattransfer behavior are important.

The field variables used are:

   xcoordinate
   ycoordinate
   axial velocity, `u`
   radial velocity, `v`
   temperature, `T`

Coordinates are normalized to `[0, 1]`.

The physical outputs are normalized to `[1, 1]`.

The sparse data are split into:

   **80% training**
   **20% validation**



# 5. PINN Architecture

The final model is a **Fourierfeature MLP**.

``` text
Input
(x, y)
   │
   ▼
Random Fourier Features
σ = 1.5
   │
   ▼
96 neurons + tanh
   │
   ▼
96 neurons + tanh
   │
   ▼
96 neurons + tanh
   │
   ▼
96 neurons + tanh
   │
   ▼
Output
(u, v, T)
```

### Final architecture

**2 → 96 × 4 → 3**

with:

   Fourier features: 96
   Fourier feature scale: σ = 1.5
   hidden width: 96
   hidden depth: 4
   activation: `tanh`
   approximately 47k trainable parameters
   Xavier initialization

The network represents:

\[
(x,y)`\rightarrow`{=tex}(`\hat `{=tex}u,`\hat `{=tex}v,`\hat `{=tex}T)
\]

Fourier features are used to improve the representation of spatial
variations that can be difficult for a standard MLP to resolve.



# 6. PhysicsInformed Training

The final loss combines sparse CFD supervision with physics and wall
heattransfer information:

\[ `\mathcal `{=tex}L = `\mathcal `{=tex}L\_{data} +
`\lambda`{=tex}*{phys}`\mathcal `{=tex}L*{PDE} +
`\lambda`{=tex}*{wall}`\mathcal `{=tex}L*{wall} \]

## Data loss

The network is compared with sparse CFD observations:

\[ `\mathcal `{=tex}L\_{data} = MSE(u)+MSE(v)+MSE(T) \]

## PDE residual

The notebook calculates derivatives using **PyTorch automatic
differentiation**.

The thermal residual is:

\[ uT_x+vT_y  `\alpha`{=tex}(T\_{xx}+T\_{yy}) \]

and the continuity residual is:

\[ u_x+v_y=0 \]

The implemented physics loss is:

\[ `\mathcal `{=tex}L\_{PDE} = MSE(`\text{energy residual}`{=tex}) +
0.5,MSE(`\text{continuity residual}`{=tex}) \]

with the normalized thermal diffusivity parameter:

\[ `\alpha=0.01`{=tex} \]

### Important scope

The notebook does **not** implement a full momentumequation residual.
The PINN physics term is specifically based on the implemented **thermal
advectiondiffusion residual + continuity constraint**.



# 7. Wall HeatFlux Assimilation

A key part of the final model is the wall heattransfer constraint.

The notebook reads the CFD wall heatflux export and converts it to a
Nusseltnumber target using:

\[ Nu = `\frac{|q''|D}`{=tex} {k,`\Delta `{=tex}T} \]

with:

   `D = 0.002 m`
   `ΔT = 400 K`
   `k = 0.052 W/(m·K)`

The wall data are cleaned by:

1.  selecting the heatedwall region,
2.  filtering the heatflux data,
3.  converting to Nusselt number,
4.  binning by `r/D`,
5.  taking the maximum value per bin,
6.  applying movingaverage smoothing.

The PINN then obtains its own wall heattransfer estimate from the
**autograd temperature gradient at the wall**.

This creates the additional wall loss:

\[ `\mathcal `{=tex}L\_{wall} \]

which encourages the network to reproduce the CFD wall heattransfer
behavior.



# 8. Why the Nusselt Number Matters

The Nusselt number is a dimensionless measure of convective heat
transfer.

For turbine cooling, the temperature field alone is not enough.
Engineers ultimately care about **how effectively heat is removed from
the component surface**.

The local Nusselt number helps identify:

   regions of high heattransfer effectiveness,
   the stagnationregion cooling intensity,
   how heat transfer changes along the wall,
   and the spatial distribution of thermal protection.

This makes `Nu(r/D)` a valuable engineering validation metric.

It is also a harder test for a PINN than simply matching temperature
values.

A model can produce a visually convincing temperature field while still
producing an inaccurate wall temperature gradient.

That is why this project explicitly evaluates the reconstructed
**Nusseltnumber distribution**.



# 9. Final Training Strategy

The final training configuration in the notebook is:

  Setting                                                           Value
   
  Sparse CFD data                                                     50%
  Train/validation split                                            80/20
  Optimizer                                                          Adam
  Learning rate                                                      1e3
  Weight decay                                                       1e5
  Epochs                                                           12,000
  Data batch                                                          512
  Collocation points                                                  512
  Wallloss batch                                                     128
  Maximum physics weight                                            0.001
  Wallloss weight                                                    3.0
  Gradient clipping                                                   1.0
  LR scheduler                                          ReduceLROnPlateau
  Physics warmup            starts at epoch 800, ramps over 2,500 epochs

The collocation points are also biased toward the nearwall region.

The model checkpoint is selected using a score that prioritizes
wallNusselt agreement while retaining reasonable validationfield
performance:

\[ score = `\frac{MAE_{Nu}}{Nu_{scale}}`{=tex} + 0.3,L\_{val} \]

The best checkpoint is saved as `model_final.pth`.



# 10. Results

The final recorded metrics in `metrics_summary.json` are:

  Metric                            Result
   
  Temperature MAE                  38.86 K
  Temperature RMSE                 62.64 K
  Relative temperature error         9.71%
  Axial velocity MAE             11.95 m/s
  Radial velocity MAE             1.36 m/s
  CFD peak Nu                         22.3
  PINN peak Nu                        16.6
  Peak Nu error                      25.5%

The PINN therefore reconstructs the primary flow and thermal variables
while also producing a wall heattransfer distribution that can be
directly compared with the CFD reference.

The Nusseltnumber comparison is especially important because it tests a
**derived engineering quantity obtained from the predicted temperature
gradient**, rather than only comparing the network's direct outputs.



# 11. Training Evolution

The notebook contains multiple PINN experiments before the final model.

The development path includes:

``` text
Initial Fourier PINN
       ↓
Sparsedata reconstruction
       ↓
Nearwall stratified sampling
       ↓
Fourierfeature refinement
       ↓
Wall heatflux / Nu assimilation
       ↓
Final wallNuaware model
```

The final model uses:

   Fourier features with `σ = 1.5`
   96neuron hidden layers
   50% stratified sparse data
   PDE residual
   wall heatflux/Nusselt loss
   validationbased checkpoint scoring

This progression reflects an important observation from the project:

> **Matching field values and matching their wall derivatives are not
> the same problem.**



# 12. Repository Data

The repository is intentionally compact. It contains the model, data,
results, and notebook rather than every intermediate figure or CFD
artifact.

``` text
turbineimpingementpinn/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── notebook/
│   └── Jet_Impingement_project.ipynb
│
├── data/
│   ├── predictions_full_field.csv
│   ├── Nu_CFD_reference.csv
│   └── Nu_PINN_curve.csv
│
├── model/
│   └── model_final.pth
│
└── results/
    └── metrics_summary.json
```

### `notebook/`

Contains the complete Python/PyTorch workflow for loading the CFD data,
preprocessing it, training the PINN, reconstructing the fields,
calculating Nusselt number, and saving the final outputs.

### `data/`

   `predictions_full_field.csv`  CFD reference values and PINN
    predictions for the full exported field.
   `Nu_CFD_reference.csv`  processed CFD wall Nusselt distribution.
   `Nu_PINN_curve.csv`  PINN wall Nusselt prediction.

### `model/`

   `model_final.pth`  trained PINN state dictionary.

### `results/`

   `metrics_summary.json`  final CFD baseline, PINN configuration,
    and evaluation metrics.



# 13. Limitations

This is a proofofconcept computational study rather than an industrial
turbinecooling model.

Current limitations include:

   steady RANS CFD rather than transient/highfidelity turbulence
    simulation,
   simplified 2D axisymmetric geometry,
   a single primary operating condition,
   sparse CFD supervision,
   simplified normalized thermal physics in the PINN,
   no full momentumequation residual,
   and remaining error in the reconstructed Nusselt distribution.

The PINN should therefore be interpreted as a **physicsinformed
reconstruction model**, not as a replacement for validated industrial
CFD.



# 14. Key Takeaway

This project sits at the intersection of:

**Aerospace Thermal Engineering + CFD + PhysicsInformed Machine
Learning**

The complete workflow is:

``` text
Turbine cooling problem
        ↓
Jet impingement physics
        ↓
ANSYS Fluent CFD
        ↓
Sparse CFD observations
        ↓
Fourierfeature PINN
        ↓
Automaticdifferentiation physics loss
        ↓
Wall heatflux assimilation
        ↓
Velocity + temperature reconstruction
        ↓
Nusseltnumber prediction
        ↓
Engineering validation
```

The main engineering lesson is:

> **A PINN that matches field values is not automatically a PINN that
> matches engineering quantities derived from field gradients.**

By explicitly incorporating wall heattransfer information, this project
investigates that second, more demanding problem.

The broader opportunity is to develop learned physicsbased models that
can eventually support **rapid design exploration, inverse problems,
sparsedata assimilation, and thermal digitaltwin applications**.



## Author

**Shivesh**

Mechanical Engineering Undergraduate\
Interests: CFD • Heat Transfer • PhysicsInformed Machine Learning •
Aerospace Applications



## Project Status

**Completed  CFD baseline + PINN reconstruction**

The repository contains the final notebook, trained model, reconstructed
field data, Nusseltnumber data, and quantitative metrics.

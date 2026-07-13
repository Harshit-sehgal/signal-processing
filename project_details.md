# Physics-Guided Adaptive Multi-Stage Chatter Detection (PG-AMCD)

## 📌 Framework Overview
The **PG-AMCD** framework is a physics-guided, multi-stage signal processing pipeline designed for vibration-based chatter detection in turning/milling operations. 

Chatter is a self-excited, unstable vibration that occurs near the natural frequencies of the machine-tool-workpiece system, causing poor surface finish, accelerated tool wear, and potential spindle damage. In contrast, stable cutting is dominated by forced vibrations at lower frequencies (spindle rotation frequency and tooth-passing frequencies).

Traditional Empirical Mode Decomposition (EMD) or Ensemble EMD (EEMD) suffer from **mode mixing** (where components of different frequencies are mixed into a single IMF, or a single frequency component is split across multiple IMFs). PG-AMCD solves this by combining:
1. **Adaptive Bandpass Filtering**: Dynamically isolating the chatter resonance band.
2. **CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)**: High-resolution decomposition.
3. **MAIW (Multi-Criteria Adaptive IMF Weighting)**: Adaptive reconstruction based on physical and statistical features.
4. **Bayesian Wavelet Denoising**: Subband-adaptive thresholding to remove remaining noise while preserving chatter transients.

---

## 🧮 Mathematical Formulation

### Stage 1: Adaptive Preprocessing & Decomposition

#### 1. Butterworth Bandpass Filter
A 3rd-order Butterworth bandpass filter is used to filter out low-frequency forced vibrations and high-frequency noise. The passband is defined as $[f_{low}, f_{high}]$, where $f_{high} = 4000\text{ Hz}$ and $f_{low}$ is adaptively selected from $\{50, 100, 150, 200\}\text{ Hz}$ for each file to minimize the Mode-Mixing Index (MMI).
The filter transfer function is:
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f^2 - f_0^2}{f \cdot B}\right)^{2N}}$$
where $B$ is the bandwidth, $f_0$ is the center frequency, and $N = 3$.

#### 2. Complete Ensemble EMD with Adaptive Noise (CEEMDAN)
CEEMDAN decomposes the preprocessed, maximum-energy 1-second segment $x(t)$ into a set of Intrinsic Mode Functions (IMFs).
Let $E_j(\cdot)$ be the operator producing the $j$-th IMF of a signal, and let $w^i(t)$ be zero-mean white noise.
1. The first IMF $IMF_1(t)$ is computed by:
   $$IMF_1(t) = \frac{1}{I} \sum_{i=1}^{I} E_1(x(t) + \epsilon_0 w^i(t))$$
   where $I = 300$ is the number of ensemble trials, and $\epsilon_0 = 0.05$ is the noise scale coefficient.
2. The first residue $r_1(t)$ is:
   $$r_1(t) = x(t) - IMF_1(t)$$
3. The $k$-th IMF ($k \ge 2$) and residue are iteratively computed:
   $$IMF_k(t) = \frac{1}{I} \sum_{i=1}^{I} E_1(r_{k-1}(t) + \epsilon_{k-1} E_{k-1}(w^i(t)))$$
   $$r_k(t) = r_{k-1}(t) - IMF_k(t)$$

#### 3. Mode Mixing Index (MMI)
The adaptive loop selects the filter low-cutoff $f_{low}$ that minimizes the Mean Adjacent IMF Correlation:
$$\text{MMI} = \frac{1}{M-1} \sum_{k=1}^{M-1} |\text{Corr}(IMF_k(t), IMF_{k+1}(t))|$$
where $M$ is the number of physical IMFs (excluding the residual). A lower MMI indicates cleaner spectral separation.

---

### Stage 2: Multi-Criteria Adaptive IMF Weighting (MAIW)

After decomposition, the signal is reconstructed as a weighted sum of IMFs:
$$x_{MAIW}(t) = \sum_{k=1}^{M} W_k \cdot IMF_k(t)$$
where $W_k$ is the weight of the $k$-th IMF, calculated as:
$$W_k = \alpha C_k + \beta E_k + \gamma K_k + \delta F_k$$
with weighting coefficients $\alpha = \beta = \gamma = \delta = 0.25$.

The individual criteria are defined as:
1. **Correlation Coefficient ($C_k$)**: Pearson correlation with the preprocessed input signal $x(t)$:
   $$C_k = \frac{|\text{Cov}(IMF_k(t), x(t))|}{\sigma_{IMF_k} \sigma_x}$$
2. **Energy Ratio ($E_k$)**: Represents the energy contribution of the mode:
   $$E_k = \frac{\sum_{t} [IMF_k(t)]^2}{\sum_{j=1}^{M} \sum_{t} [IMF_j(t)]^2}$$
3. **Kurtosis ($K_k$)**: Capture the impulsiveness of chatter impacts (non-impulsive signals have $fisher\_kurtosis \approx 0$, while chatter signals exhibit high kurtosis):
   $$K_k = \frac{\text{Kurt}(IMF_k)}{\sum_{j=1}^{M} \text{Kurt}(IMF_j)} \quad \text{where } \text{Kurt}(y) = \frac{\frac{1}{T}\sum_t(y(t)-\mu_y)^4}{\sigma_y^4}$$
4. **Frequency Proximity ($F_k$)**: Focuses on the natural chatter frequency band ($500\text{--}2000$ Hz):
   $$F_k = \exp\left( -\frac{(f_{\text{dom}, k} - 1250)^2}{2 \cdot 500^2} \right)$$
   where $f_{\text{dom}, k}$ is the dominant frequency of $IMF_k(t)$ computed via Welch's Power Spectral Density (PSD).

The weights are normalized: $W_k = \frac{W_k}{\sum_j W_j}$ to maintain the signal scale.

---

### Stage 3: Bayesian Adaptive Wavelet Denoising

Wavelet denoising is applied to $x_{MAIW}(t)$ using the Daubechies 8 (`db8`) wavelet at level 4.

1. **Noise Variance Estimation**:
   The noise standard deviation $\sigma_n$ is estimated using the Median Absolute Deviation (MAD) of the first-level detail coefficients $d_1$:
   $$\sigma_n = \frac{\text{median}(|d_1|)}{0.6745}$$
2. **BayesShrink Threshold Calculation**:
   For each subband $j$ with coefficients $d_j$, the variance of the subband signal $\sigma_y^2$ is:
   $$\sigma_y^2 = \frac{1}{N_j} \sum_{n=1}^{N_j} (d_j[n])^2$$
   The true signal standard deviation $\sigma_x$ is estimated as:
   $$\sigma_x = \sqrt{\max(0.0, \sigma_y^2 - \sigma_n^2)}$$
   The adaptive soft threshold $T_j$ for subband $j$ is:
   $$T_j = \begin{cases} 
      \max(|d_j|) & \text{if } \sigma_x = 0 \\
      \frac{\sigma_n^2}{\sigma_x} & \text{if } \sigma_x > 0 
   \end{cases}$$
3. **Soft Thresholding Application**:
   $$d_j'[n] = \text{sgn}(d_j[n]) \cdot \max(0.0, |d_j[n]| - T_j)$$
4. **Inverse Wavelet Transform**:
   Reconstruct the clean signal $x_{clean}(t)$ from the thresholded detail coefficients and unchanged approximation coefficients.

---

## 📊 Verification and Run Parameters

### CEEMDAN Optimizations
*   **Ensemble Trials**: 300 (to achieve high statistical convergence and noise cancellation).
*   **Noise Standard Deviation**: 0.05 (ideal for separating low-level vibration modes).
*   **Multiprocessing**: Enabled via `parallel=True` to compute the ensemble trials across all CPU cores.

# Feature: Normalize segment confidence to 0–1 and UI badges

## Why normalize confidence to 0–1
- Makes confidence intuitive and comparable across files/refs.
- Ties directly to the threshold that triggers detection.
- 0 = at threshold; 1 = well above (≥2× threshold), independent of absolute scale.

## How it’s computed
- raw_max = max correlation inside the segment.
- thr_baseline = max over refs of (threshold × mean(correlation_ref)).
- confidence = clamp01(raw_max/thr_baseline − 1).

## Example
- threshold = 4.5; ref means: 0.020 → thr1=0.09, 0.015 → thr2=0.0675 → thr_baseline=0.09.
- A: raw_max=0.135 → 0.135/0.09−1 = 0.50.
- B: raw_max=0.081 → −0.10 → 0.00.
- C: raw_max=0.200 → 1.22 → 1.00.

## Implementation
- Backend returns per-ref thresholds and stores normalized confidence (0–1).
- Frontend shows it as a badge on each Wavesurfer region and a top-left badge on every grid card.

## Screenshots
Place screenshots at `docs/images/` and ensure they are committed.

![Region badge](../images/region-badge.png)
![Grid badge](../images/grid-badge.png)



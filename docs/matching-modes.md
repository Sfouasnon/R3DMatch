# Matching Domains

R3DMatch uses two related concepts:

- analysis `mode`
- review `matching_domain`

The lower-level `analyze` command still exposes legacy `scene` / `view` modes. The operator review workflow uses normalized matching-domain language:

- `scene`
- `perceptual`

## Scene

Scene matching analyzes the neutral target before any creative monitoring transform. Use it when the goal is technical array matching in a camera-space style domain.

## Perceptual

Perceptual matching follows the preview/display path used for operator review. Use it when you want the exposure comparison to track the rendered viewing conditions more closely.

## Practical Guidance

- use `scene` when the review should stay closest to technical calibration intent
- use `perceptual` when operator-visible preview behavior matters more than raw technical neutrality
- keep preview settings consistent across the set so the comparison remains meaningful

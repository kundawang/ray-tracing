# Ray Tracing

An implementation of [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) in Go.
![Original Render Image](original-render-small.png?raw=true)

If you're exprementing/debugging code, you can reduce `samplesPerPixel` to make rendering faster (e.g `1`), and if you want to have precise rendering use `500`.

## Volumetric absorption in dielectric media

A thick piece of colored glass looks darker than a thin one. `Dielectric`
therefore accepts an optional RGB absorption coefficient (per unit distance)
and attenuates transmitted light with the Beer–Lambert law while the ray
travels inside the medium:

```go
// No absorption: identical to the original behavior (white attenuation).
clearGlass := NewDielectric(1.5, nil)

// Absorbs blue most strongly, red least: thick sections look warm.
amberGlass := NewDielectric(1.5, &Color{R: 0.1, G: 0.5, B: 1.2})
```

For a channel with coefficient `a` and an in-medium path length `d`, the
transmitted fraction is `exp(-a*d)`. Distance is accumulated per ray on a
stack of dielectric volumes, so:

- entry, exit and (total or Schlick) reflection paths all accumulate
  correctly — a ray reflected inside the glass keeps accruing distance;
- concentric layers only charge each material for its own segments, and an
  air cavity between glass layers is never counted as glass;
- nested shells sharing the same material behave as one continuous volume.

Negative, `NaN` or infinite coefficients are treated as zero (never
amplifying light); non-finite path lengths and extremely thick media produce
finite black rather than `NaN`. Rays with no absorption configured keep the
original result exactly. Run `go test ./...` for the full behavior suite;
`RENDER_ABSORPTION_PPM=1 go test ./...` additionally writes a comparison
image to `artifacts/absorption-spheres.ppm`.

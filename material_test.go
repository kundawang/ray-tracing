package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The dielectric branch is `reflectance(...) > randFloat()`, so returning 1
// forces the refraction branch (when refraction is possible) and returning 0
// forces reflection.
func forceRefraction(t *testing.T) {
	t.Helper()
	old := randFloat
	randFloat = func() float64 { return 1 }
	t.Cleanup(func() { randFloat = old })
}

func forceReflection(t *testing.T) {
	t.Helper()
	old := randFloat
	randFloat = func() float64 { return 0 }
	t.Cleanup(func() { randFloat = old })
}

func approxEq(t *testing.T, got, want float64, tol float64, msg string) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %g)", msg, got, want, tol)
	}
}

func approxColor(t *testing.T, got, want *Color, tol float64, msg string) {
	t.Helper()
	approxEq(t, got.R, want.R, tol, msg+" R")
	approxEq(t, got.G, want.G, tol, msg+" G")
	approxEq(t, got.B, want.B, tol, msg+" B")
}

func finiteColor(t *testing.T, c *Color, msg string) {
	t.Helper()
	if math.IsNaN(c.R) || math.IsNaN(c.G) || math.IsNaN(c.B) ||
		math.IsInf(c.R, 0) || math.IsInf(c.G, 0) || math.IsInf(c.B, 0) {
		t.Errorf("%s: non-finite color %v", msg, c)
	}
}

// transmit drives a deterministic (refraction-forced) ray through the hittable
// world until it leaves every dielectric, multiplying the attenuation of each
// scattering event. It returns the accumulated transmittance and the number of
// scatters.
func transmit(t *testing.T, world *HitterList, origin *Point3, dir *Vec3) (*Color, int) {
	t.Helper()
	r := &Ray{Orig: origin, Dir: dir}
	product := &Color{1, 1, 1}
	scatters := 0
	for i := 0; i < 100; i++ {
		ok, rec := world.Hit(r, 0.001, Infinity, new(HitRecord))
		if !ok {
			break
		}
		ok2, attenuation, scattered := rec.Mat.Scatter(r, rec)
		if !ok2 {
			return &Color{0, 0, 0}, scatters
		}
		product = product.Mul(attenuation)
		r = scattered
		scatters++
		if len(r.mediumStack) == 0 {
			break
		}
	}
	return product, scatters
}

// ---------------------------------------------------------------------------
// Constructor / coefficient validation
// ---------------------------------------------------------------------------

func TestValidAbsorption(t *testing.T) {
	if validAbsorption(nil) != nil {
		t.Fatal("nil absorption should stay nil")
	}

	nan, posInf, negInf := math.NaN(), math.Inf(1), math.Inf(-1)
	got := validAbsorption(&Color{-1, nan, posInf})
	approxColor(t, got, &Color{0, 0, 0}, 0, "negative/NaN/+Inf sanitize to zero")

	got = validAbsorption(&Color{negInf, -0.0, 0.5})
	approxColor(t, got, &Color{0, 0, 0.5}, 0, "-Inf and negative sanitize, valid kept")

	// Sanitization must not mutate the caller's color.
	src := &Color{-3, 1, 2}
	_ = validAbsorption(src)
	approxColor(t, src, &Color{-3, 1, 2}, 0, "source color must be untouched")
}

func TestNewDielectricSanitizes(t *testing.T) {
	d := NewDielectric(1.5, &Color{math.NaN(), -2, math.Inf(1)})
	approxColor(t, d.Absorption, &Color{0, 0, 0}, 0, "constructor sanitizes bad channels")

	if NewDielectric(1.5, nil).Absorption != nil {
		t.Fatal("nil absorption should remain nil")
	}
}

// ---------------------------------------------------------------------------
// Beer-Lambert transmittance
// ---------------------------------------------------------------------------

func TestBeerLambert(t *testing.T) {
	a := &Color{0.5, 1.0, 1.5}
	d := NewDielectric(1.5, a)

	// No absorption configured: always white (legacy behavior).
	none := NewDielectric(1.5, nil)
	approxColor(t, none.beerLambert(42), &Color{1, 1, 1}, 0, "nil absorption is white at any distance")

	zero := NewDielectric(1.5, &Color{0, 0, 0})
	approxColor(t, zero.beerLambert(42), &Color{1, 1, 1}, 0, "zero absorption is white at any distance")

	// Zero/negative distance: no travel, no attenuation.
	approxColor(t, d.beerLambert(0), &Color{1, 1, 1}, 0, "zero distance")
	approxColor(t, d.beerLambert(-5), &Color{1, 1, 1}, 0, "negative distance clamped to no attenuation")

	// Known Beer-Lambert values at distance 2.
	want := &Color{math.Exp(-1), math.Exp(-2), math.Exp(-3)}
	approxColor(t, d.beerLambert(2), want, 1e-15, "exp(-a*d) per channel")

	// Non-finite distance means no surviving light, never a NaN color.
	black := d.beerLambert(math.Inf(1))
	approxColor(t, black, &Color{0, 0, 0}, 0, "infinite distance -> black")
	finiteColor(t, black, "infinite distance")
	blackNaN := d.beerLambert(math.NaN())
	approxColor(t, blackNaN, &Color{0, 0, 0}, 0, "NaN distance -> black")

	// Extremely thick medium: underflow to zero must stay finite.
	thick := d.beerLambert(1e308)
	approxColor(t, thick, &Color{0, 0, 0}, 0, "extreme distance -> black")
	finiteColor(t, thick, "extreme distance")

	huge := NewDielectric(1.5, &Color{1e308, 1e308, 1e308})
	strong := huge.beerLambert(1)
	approxColor(t, strong, &Color{0, 0, 0}, 0, "extreme coefficient -> black")
	finiteColor(t, strong, "extreme coefficient")

	// Defensive: a struct literal bypassing NewDielectric with bad
	// coefficients must never amplify or corrupt light.
	bypass := &Dielectric{Ir: 1.5, Absorption: &Color{-1, math.NaN(), math.Inf(1)}}
	approxColor(t, bypass.beerLambert(3), &Color{1, 1, 1}, 0, "bad literal coefficients treated as no absorption")
}

// ---------------------------------------------------------------------------
// Legacy behavior: no absorption configured => identical white attenuation
// ---------------------------------------------------------------------------

func TestNoAbsorptionMatchesLegacy(t *testing.T) {
	forceRefraction(t)

	// Original struct literal style (nil Absorption) vs new constructor.
	legacy := &Dielectric{Ir: 1.5}
	constructor := NewDielectric(1.5, nil)
	world1 := &HitterList{Hitters: []Hitter{&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: legacy}}}
	world2 := &HitterList{Hitters: []Hitter{&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: constructor}}}

	got1, n1 := transmit(t, world1, &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	got2, n2 := transmit(t, world2, &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	if n1 != 2 || n2 != 2 {
		t.Fatalf("want enter+exit (2 scatters), got %d and %d", n1, n2)
	}
	approxColor(t, got1, &Color{1, 1, 1}, 0, "legacy dielectric stays white")
	approxColor(t, got2, &Color{1, 1, 1}, 0, "nil-absorption dielectric stays white")
}

// ---------------------------------------------------------------------------
// Single absorbing sphere: entry, distance accumulation, exit
// ---------------------------------------------------------------------------

func TestSingleSphereAbsorption(t *testing.T) {
	forceRefraction(t)

	abs := &Color{0.5, 1.0, 1.5}
	mat := NewDielectric(1.5, abs)
	world := &HitterList{Hitters: []Hitter{&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}}}

	// Center ray through a unit sphere: chord length is exactly 2.
	got, n := transmit(t, world, &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	if n != 2 {
		t.Fatalf("want 2 scatters, got %d", n)
	}
	want := &Color{math.Exp(-1), math.Exp(-2), math.Exp(-3)}
	approxColor(t, got, want, 1e-12, "center ray attenuated over diameter 2R")

	// Inspect the bookkeeping directly, too.
	r := &Ray{Orig: &Point3{0, 0, -3}, Dir: &Vec3{0, 0, 1}}
	rec := new(HitRecord)
	if !world.Hitters[0].Hit(r, 0.001, Infinity, rec) {
		t.Fatal("entry hit expected")
	}
	_, _, scattered := mat.Scatter(r, rec)
	if len(scattered.mediumStack) != 1 || scattered.mediumStack[0].mat != mat {
		t.Fatal("entry must push one frame for this dielectric")
	}
	if scattered.mediumStack[0].depth != 1 {
		t.Fatalf("entry depth want 1, got %d", scattered.mediumStack[0].depth)
	}

	rec2 := new(HitRecord)
	if !world.Hitters[0].Hit(scattered, 0.001, Infinity, rec2) {
		t.Fatal("exit hit expected")
	}
	_, att, out := mat.Scatter(scattered, rec2)
	approxColor(t, att, want, 1e-12, "exit attenuation")
	if len(out.mediumStack) != 0 {
		t.Fatal("exit must pop the frame")
	}
}

// ---------------------------------------------------------------------------
// Center vs edge coloration, and different radii of the same material
// ---------------------------------------------------------------------------

func TestThickerPathsAreDarker(t *testing.T) {
	forceRefraction(t)

	abs := &Color{0.7, 0.7, 0.7}

	// Ir = 1 means the ray does not bend, so geometric chord lengths are
	// exact and isolate the absorption bookkeeping from Snell's law.
	noBend := NewDielectric(1.0, abs)
	mkWorld := func(radius float64, mat *Dielectric) *HitterList {
		return &HitterList{Hitters: []Hitter{&Sphere{Center: &Point3{0, 0, 0}, Radius: radius, Mat: mat}}}
	}

	// Same unit sphere: a central chord (2) is longer than an offset chord.
	// Offset b=0.8 => chord = 2*sqrt(1-0.64) = 1.2.
	center, _ := transmit(t, mkWorld(1, noBend), &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	edge, _ := transmit(t, mkWorld(1, noBend), &Point3{0.8, 0, -3}, &Vec3{0, 0, 1})
	approxColor(t, edge, &Color{math.Exp(-0.7 * 1.2), math.Exp(-0.7 * 1.2), math.Exp(-0.7 * 1.2)}, 1e-12,
		"edge chord length 1.2")
	if !(center.R < edge.R && center.G < edge.G && center.B < edge.B) {
		t.Errorf("thicker center path must be darker: center=%v edge=%v", center, edge)
	}

	// Different radii, same material: the bigger sphere's center ray
	// (diameter 4) is darker than the small one's (diameter 2).
	big, _ := transmit(t, mkWorld(2, noBend), &Point3{0, 0, -5}, &Vec3{0, 0, 1})
	small, _ := transmit(t, mkWorld(1, noBend), &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	approxColor(t, big, &Color{math.Exp(-0.7 * 4), math.Exp(-0.7 * 4), math.Exp(-0.7 * 4)}, 1e-12,
		"big sphere chord 4")
	if !(big.R < small.R && big.G < small.G && big.B < small.B) {
		t.Errorf("bigger sphere must be darker: big=%v small=%v", big, small)
	}

	// With a real refractive index the off-axis ray bends, so the absorbed
	// distance must follow the *refracted* path, not the straight chord.
	// b=0.8 into Ir=1.5: sin(alpha)=0.8, sin(beta)=0.8/1.5, hence the
	// internal chord is 2*cos(beta) = 2*sqrt(1-(0.8/1.5)^2) ~= 1.6919.
	glass := NewDielectric(1.5, abs)
	refractedWorld := mkWorld(1, glass)
	r := &Ray{Orig: &Point3{0.8, 0, -3}, Dir: &Vec3{0, 0, 1}}
	rec := new(HitRecord)
	if !refractedWorld.Hitters[0].Hit(r, 0.001, Infinity, rec) {
		t.Fatal("entry hit expected")
	}
	_, _, insideRay := glass.Scatter(r, rec)
	rec2 := new(HitRecord)
	if !refractedWorld.Hitters[0].Hit(insideRay, 0.001, Infinity, rec2) {
		t.Fatal("exit hit expected")
	}
	_, exitAtt, _ := glass.Scatter(insideRay, rec2)
	bentLen := 2 * math.Sqrt(1-math.Pow(0.8/1.5, 2))
	approxColor(t, exitAtt,
		&Color{math.Exp(-0.7 * bentLen), math.Exp(-0.7 * bentLen), math.Exp(-0.7 * bentLen)},
		1e-12, "off-axis absorption follows the refracted chord")
	if bentLen <= 1.2 {
		t.Fatalf("sanity: refracted chord %v should exceed straight chord 1.2", bentLen)
	}

	// Center rays hit at normal incidence and never bend, even with Ir=1.5:
	// they still absorb over exactly the diameter.
	centerGlass, n := transmit(t, mkWorld(1, glass), &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	if n != 2 {
		t.Fatalf("want 2 scatters, got %d", n)
	}
	approxColor(t, centerGlass,
		&Color{math.Exp(-0.7 * 2), math.Exp(-0.7 * 2), math.Exp(-0.7 * 2)},
		1e-12, "normal-incidence center ray absorbs over diameter")
}

// ---------------------------------------------------------------------------
// Total internal reflection: stack retained, reflected segment counted later
// ---------------------------------------------------------------------------

func TestTotalInternalReflection(t *testing.T) {
	// cannotRefract short-circuits to reflection regardless of randFloat;
	// force refraction anyway to prove the TIR branch itself fires.
	forceRefraction(t)

	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	sphere := &Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}

	// Ray already inside the glass, hitting the inner surface at a grazing
	// angle where refraction is impossible (1.5 * sin(theta) > 1).
	inside := &Ray{
		Orig:        &Point3{0.9, 0, 0},
		Dir:         &Vec3{0, 1, 0},
		mediumStack: []*mediumFrame{{mat: mat, travelled: 0, depth: 1}},
	}
	rec := new(HitRecord)
	if !sphere.Hit(inside, 0.001, Infinity, rec) {
		t.Fatal("grazing hit expected")
	}
	if rec.FrontFace {
		t.Fatal("inner hit must be a back face")
	}
	_, att, reflected := mat.Scatter(inside, rec)
	approxColor(t, att, &Color{1, 1, 1}, 0, "TIR must not attenuate on reflection")
	if len(reflected.mediumStack) != 1 || reflected.mediumStack[0].mat != mat {
		t.Fatal("TIR must retain the medium frame")
	}
	segLen := math.Sqrt(0.19) // distance from (0.9,0,0) to the grazing hit
	approxEq(t, reflected.mediumStack[0].travelled, segLen, 1e-12,
		"TIR segment is accumulated inside the medium")

	// From the grazing point straight through the center to the opposite
	// wall: exit attenuation must cover BOTH segments.
	toCenter := rec.P.Sub(&Point3{0, 0, 0}).ToVec3().Neg().UnitVector()
	exitRay := &Ray{Orig: rec.P, Dir: toCenter, mediumStack: reflected.mediumStack}
	rec2 := new(HitRecord)
	if !sphere.Hit(exitRay, 0.001, Infinity, rec2) {
		t.Fatal("exit hit expected")
	}
	_, exitAtt, out := mat.Scatter(exitRay, rec2)
	total := segLen + 2.0 // grazing point -> through center -> opposite wall
	approxColor(t, exitAtt,
		&Color{math.Exp(-0.5 * total), math.Exp(-0.5 * total), math.Exp(-0.5 * total)},
		1e-12, "exit attenuation includes the reflected segment")
	if len(out.mediumStack) != 0 {
		t.Fatal("exit must pop the frame")
	}
}

// A Schlick reflection event inside the glass behaves like TIR for
// bookkeeping: the frame is retained and the reflected path keeps accruing.
func TestSchlickReflectionKeepsFrame(t *testing.T) {
	forceReflection(t)

	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	sphere := &Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}
	inside := &Ray{
		Orig:        &Point3{0, 0, 0},
		Dir:         &Vec3{0, 0, 1},
		mediumStack: []*mediumFrame{{mat: mat, travelled: 0.3, depth: 1}},
	}
	rec := new(HitRecord)
	if !sphere.Hit(inside, 0.001, Infinity, rec) {
		t.Fatal("back wall hit expected")
	}
	_, att, reflected := mat.Scatter(inside, rec)
	approxColor(t, att, &Color{1, 1, 1}, 0, "reflection does not attenuate")
	if len(reflected.mediumStack) != 1 || reflected.mediumStack[0].depth != 1 {
		t.Fatal("reflection must retain the frame and depth")
	}
	approxEq(t, reflected.mediumStack[0].travelled, 1.3, 1e-12,
		"reflected segment accumulates in the retained frame")
}

// ---------------------------------------------------------------------------
// Concentric layers: an air cavity must never be charged to the glass, and
// distinct nested materials only count their own segments.
// ---------------------------------------------------------------------------

func traceInterfaces(t *testing.T, world *HitterList, r *Ray) ([]*Color, []*Dielectric, *Ray) {
	t.Helper()
	var events []*Color
	var mats []*Dielectric
	for i := 0; i < 20; i++ {
		ok, rec := world.Hit(r, 0.001, Infinity, new(HitRecord))
		if !ok {
			break
		}
		d := rec.Mat.(*Dielectric)
		_, att, scattered := d.Scatter(r, rec)
		events = append(events, att)
		mats = append(mats, d)
		r = scattered
		if len(r.mediumStack) == 0 {
			break
		}
	}
	return events, mats, r
}

func TestConcentricAirCavity(t *testing.T) {
	forceRefraction(t)

	outer := NewDielectric(1.5, &Color{0.1, 0.2, 0.3}) // glass shell, R=2
	air := NewDielectric(1.0, nil)                     // air cavity, R=1, no absorption
	world := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 2, Mat: outer},
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: air},
	}}

	events, mats, finalRay := traceInterfaces(t, world, &Ray{Orig: &Point3{0, 0, -5}, Dir: &Vec3{0, 0, 1}})
	if len(events) != 4 {
		t.Fatalf("want 4 interface events, got %d", len(events))
	}
	approxColor(t, events[0], &Color{1, 1, 1}, 0, "outer entry white")
	approxColor(t, events[1], &Color{1, 1, 1}, 0, "air entry white")
	approxColor(t, events[2], &Color{1, 1, 1}, 0, "air exit white (no absorption)")
	if mats[2] != air {
		t.Fatal("third event must close the air frame")
	}

	// Glass thickness along the axis: (2-1) on each side = 2. If the air
	// cavity (length 2) leaked into the glass frame this would be exp(-a*4).
	approxColor(t, events[3],
		&Color{math.Exp(-0.1 * 2), math.Exp(-0.2 * 2), math.Exp(-0.3 * 2)},
		1e-12, "glass attenuation excludes the air cavity")
	if mats[3] != outer {
		t.Fatal("fourth event must close the outer glass")
	}
	if len(finalRay.mediumStack) != 0 {
		t.Fatal("all frames must be closed")
	}
}

func TestConcentricDistinctMaterials(t *testing.T) {
	forceRefraction(t)

	outer := NewDielectric(1.5, &Color{0.1, 0.2, 0.3}) // outer glass, R=2
	inner := NewDielectric(1.3, &Color{0.9, 0.8, 0.7}) // inner glass, R=1
	world := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 2, Mat: outer},
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: inner},
	}}

	events, mats, finalRay := traceInterfaces(t, world, &Ray{Orig: &Point3{0, 0, -5}, Dir: &Vec3{0, 0, 1}})
	if len(events) != 4 {
		t.Fatalf("want 4 interface events, got %d", len(events))
	}

	// Inner sphere diameter = 2.
	approxColor(t, events[2],
		&Color{math.Exp(-0.9 * 2), math.Exp(-0.8 * 2), math.Exp(-0.7 * 2)},
		1e-12, "inner glass attenuated over its own diameter")
	if mats[2] != inner {
		t.Fatal("third event must close the inner material")
	}

	// Outer glass thickness along the axis = 2 total; inner volume excluded.
	approxColor(t, events[3],
		&Color{math.Exp(-0.1 * 2), math.Exp(-0.2 * 2), math.Exp(-0.3 * 2)},
		1e-12, "outer glass excludes the inner volume")
	if mats[3] != outer {
		t.Fatal("fourth event must close the outer material")
	}
	if len(finalRay.mediumStack) != 0 {
		t.Fatal("all frames must be closed")
	}
}

func TestConcentricSameMaterialOneVolume(t *testing.T) {
	forceRefraction(t)

	mat := NewDielectric(1.5, &Color{0.4, 0.4, 0.4})
	world := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 2, Mat: mat},
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}, // same material pointer
	}}

	got, n := transmit(t, world, &Point3{0, 0, -5}, &Vec3{0, 0, 1})
	if n != 4 {
		t.Fatalf("want 4 interface scatters, got %d", n)
	}
	// One continuous volume spanning the outer diameter 4; a single
	// attenuation event at the final exit.
	want := math.Exp(-0.4 * 4)
	approxEq(t, got.R, want, 1e-12, "same-material shells form one volume (R)")
	approxEq(t, got.G, want, 1e-12, "same-material shells form one volume (G)")
	approxEq(t, got.B, want, 1e-12, "same-material shells form one volume (B)")
}

// ---------------------------------------------------------------------------
// Tangent rays, zero thickness, recursion limit
// ---------------------------------------------------------------------------

func TestTangentRay(t *testing.T) {
	forceRefraction(t)

	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	sphere := &Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}

	// Ray one radius off-axis: discriminant is exactly zero, one tangent hit.
	r := &Ray{Orig: &Point3{1, 0, -3}, Dir: &Vec3{0, 0, 1}}
	rec := new(HitRecord)
	if !sphere.Hit(r, 0.001, Infinity, rec) {
		t.Fatal("tangent hit expected")
	}
	_, att, scattered := mat.Scatter(r, rec)
	// The codebase's face convention marks a dot==0 tangent as a back face;
	// with an empty medium stack that is an orphaned back face. Total
	// reflection at the grazing angle keeps the (empty) stack: defined,
	// finite, white, and no frame is created.
	approxColor(t, att, &Color{1, 1, 1}, 0, "tangent ray does not absorb")
	finiteColor(t, att, "tangent ray")
	if len(scattered.mediumStack) != 0 {
		t.Fatal("tangent ray must not open a medium frame")
	}
}

func TestZeroThickness(t *testing.T) {
	forceRefraction(t)

	// (a) A vanishingly thin shell. The tracer's 0.001 hit epsilon requires
	// the shell to be thicker than that; 0.01 still demonstrates attenuation
	// proportional only to shell thickness.
	outer := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	air := NewDielectric(1.0, nil)
	const shell = 0.01
	world := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: outer},
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1 - shell, Mat: air},
	}}
	got, _ := transmit(t, world, &Point3{0, 0, -3}, &Vec3{0, 0, 1})
	want := math.Exp(-0.5 * 2 * shell)
	approxEq(t, got.R, want, 1e-12, "thin shell attenuation over 2*shell only")
	approxEq(t, got.G, want, 1e-12, "thin shell attenuation over 2*shell only")
	approxEq(t, got.B, want, 1e-12, "thin shell attenuation over 2*shell only")

	// (b) Exactly zero travelled distance (entry and exit coincide): white.
	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	zeroRay := &Ray{
		Orig:        &Point3{0, 0, 1},
		Dir:         &Vec3{0, 0, 1},
		mediumStack: []*mediumFrame{{mat: mat, travelled: 0, depth: 1}},
	}
	zeroRec := &HitRecord{
		P:         &Point3{0, 0, 1},
		Normal:    &Vec3{0, 0, -1}, // back-face normal points inward
		T:         0,
		FrontFace: false,
	}
	_, att, out := mat.Scatter(zeroRay, zeroRec)
	approxColor(t, att, &Color{1, 1, 1}, 0, "zero segment exits unabsorbed")
	if len(out.mediumStack) != 0 {
		t.Fatal("zero-thickness exit must still close the frame")
	}

	// (c) Radius-zero sphere never crashes and yields finite colors.
	zeroWorld := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 0, Mat: NewDielectric(1.5, &Color{1, 1, 1})},
	}}
	c := rayColor(&Ray{Orig: &Point3{0, 0, -1}, Dir: &Vec3{0, 0, 1}}, zeroWorld, 10)
	finiteColor(t, c, "radius-zero sphere")
}

func TestRecursionLimit(t *testing.T) {
	forceRefraction(t)

	world := &HitterList{Hitters: []Hitter{
		&Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: NewDielectric(1.5, &Color{0.5, 0.5, 0.5})},
	}}
	r := &Ray{Orig: &Point3{0, 0, -3}, Dir: &Vec3{0, 0, 1}}

	// depth 0: bounce limit reached before any light is gathered.
	c0 := rayColor(r, world, 0)
	approxColor(t, c0, &Color{0, 0, 0}, 0, "depth 0 returns black")
	finiteColor(t, c0, "depth 0")

	// depth 1: one scatter then the limit returns black; the unclosed
	// medium frame is abandoned without panic or NaN.
	c1 := rayColor(r, world, 1)
	approxColor(t, c1, &Color{0, 0, 0}, 0, "depth 1 returns black")
	finiteColor(t, c1, "depth 1")
}

// ---------------------------------------------------------------------------
// Non-unit ray directions: geometric distance, not ray parameter t
// ---------------------------------------------------------------------------

func TestNonUnitRayDirection(t *testing.T) {
	forceRefraction(t)

	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	sphere := &Sphere{Center: &Point3{0, 0, 0}, Radius: 1, Mat: mat}

	// Inside ray with direction length 3: parametric t to the back wall is
	// 2/3, but the geometric chord is 2.
	inside := &Ray{
		Orig:        &Point3{0, 0, -1},
		Dir:         &Vec3{0, 0, 3},
		mediumStack: []*mediumFrame{{mat: mat, travelled: 0, depth: 1}},
	}
	rec := new(HitRecord)
	if !sphere.Hit(inside, 0.001, Infinity, rec) {
		t.Fatal("back wall hit expected")
	}
	approxEq(t, rec.T, 2.0/3.0, 1e-15, "parametric t is 2/3 for a length-3 direction")
	_, att, out := mat.Scatter(inside, rec)
	approxColor(t, att,
		&Color{math.Exp(-0.5 * 2), math.Exp(-0.5 * 2), math.Exp(-0.5 * 2)},
		1e-12, "absorption uses geometric distance 2, not t")
	if len(out.mediumStack) != 0 {
		t.Fatal("frame must close on exit")
	}
}

// ---------------------------------------------------------------------------
// Other materials carry the medium stack unchanged
// ---------------------------------------------------------------------------

func TestOtherMaterialsPreserveMediumStack(t *testing.T) {
	mat := NewDielectric(1.5, &Color{0.5, 0.5, 0.5})
	frame := &mediumFrame{mat: mat, travelled: 1.25, depth: 1}
	r := &Ray{Orig: &Point3{0, 0, 0}, Dir: &Vec3{1, 0, 0}, mediumStack: []*mediumFrame{frame}}
	rec := &HitRecord{P: &Point3{1, 0, 0}, Normal: &Vec3{1, 0, 0}, FrontFace: true}

	_, _, lRay := (&Lambertian{Albedo: &Color{0.8, 0.8, 0.8}}).Scatter(r, rec)
	if len(lRay.mediumStack) != 1 || lRay.mediumStack[0] != frame {
		t.Fatal("lambertian must preserve the ray's medium stack")
	}

	_, _, mRay := NewMetal(&Color{0.8, 0.8, 0.8}, 0).Scatter(r, rec)
	if len(mRay.mediumStack) != 1 || mRay.mediumStack[0] != frame {
		t.Fatal("metal must preserve the ray's medium stack")
	}
}

// ---------------------------------------------------------------------------
// End-to-end smoke test of the original scene: everything stays finite.
// ---------------------------------------------------------------------------

func TestRandomSceneRendersFiniteColors(t *testing.T) {
	forceRefraction(t)
	world := randomScene()
	for i := 0; i < 25; i++ {
		origin := &Point3{13, 2, 3}
		target := &Point3{
			X: RandFloat(-6, 6),
			Y: RandFloat(-0.5, 2.5),
			Z: RandFloat(-6, 6),
		}
		dir := target.Sub(origin).ToVec3().UnitVector()
		c := rayColor(&Ray{Orig: origin, Dir: dir}, world, 6)
		finiteColor(t, c, "random scene ray")
		if c.R < 0 || c.G < 0 || c.B < 0 {
			t.Errorf("negative color channel: %v", c)
		}
	}
}

// ---------------------------------------------------------------------------
// Render-level check: trace whole images against a white background so the
// transmitted color is the Beer-Lambert transmittance directly, and verify the
// visual claims (center darker than edge, bigger sphere darker, tinting,
// no-absorption control). Set RENDER_ABSORPTION_PPM to also dump the image.
// ---------------------------------------------------------------------------

// rayColorWhite is rayColor with a white background instead of the sky
// gradient, so a transmitted pixel equals the product of attenuations.
func rayColorWhite(r *Ray, world *HitterList, depth int) *Color {
	if depth <= 0 {
		return &Color{0, 0, 0}
	}
	if ok, rec := world.Hit(r, 0.001, Infinity, new(HitRecord)); ok {
		if ok2, attenuation, scattered := rec.Mat.Scatter(r, rec); ok2 {
			return attenuation.Mul(rayColorWhite(scattered, world, depth-1))
		}
		return &Color{0, 0, 0}
	}
	return &Color{1, 1, 1}
}

// renderSphereGrid renders one row of equal-size spheres with the given
// materials, centered on the x axis, viewed from straight ahead. Pixel (i,j)
// is the linear (un-gamma-corrected) color.
func renderSphereGrid(t *testing.T, mats []*Dielectric, radius float64, width, height int) [][]*Color {
	t.Helper()
	world := new(HitterList)
	spacing := radius * 3
	for k, m := range mats {
		x := (float64(k) - float64(len(mats)-1)/2.0) * spacing
		world.Add(&Sphere{Center: &Point3{x, 0, 0}, Radius: radius, Mat: m})
	}

	const camZ = -8.0
	// Frame the whole row: widest extent plus a margin, keeping square pixels.
	halfExtent := spacing*float64(len(mats)-1)/2 + radius
	viewW := 2 * halfExtent * 1.25
	viewH := viewW * float64(height) / float64(width)
	pixels := make([][]*Color, height)
	for j := 0; j < height; j++ {
		pixels[j] = make([]*Color, width)
		for i := 0; i < width; i++ {
			u := (float64(i) + 0.5) / float64(width)
			v := 1 - (float64(j)+0.5)/float64(height)
			dir := (&Vec3{
				X: (u - 0.5) * viewW,
				Y: (v - 0.5) * viewH,
				Z: -camZ,
			}).UnitVector()
			pixels[j][i] = rayColorWhite(&Ray{Orig: &Point3{0, 0, camZ}, Dir: dir}, world, 20)
		}
	}
	return pixels
}

// columnMean returns the per-channel mean color of image column i.
func columnMean(img [][]*Color, i int) *Color {
	c := &Color{}
	for j := range img {
		c = c.Add(img[j][i])
	}
	n := float64(len(img))
	return &Color{c.R / n, c.G / n, c.B / n}
}

func TestRenderedAbsorptionImage(t *testing.T) {
	forceRefraction(t)

	const W, H, R = 240, 80, 1.0
	clear := NewDielectric(1.5, nil)
	gray := NewDielectric(1.5, &Color{0.6, 0.6, 0.6})
	tinted := NewDielectric(1.5, &Color{0.1, 0.5, 1.2}) // absorbs blue most, red least
	img := renderSphereGrid(t, []*Dielectric{clear, gray, tinted}, R, W, H)

	// Each sphere occupies one third of the width; sample central columns.
	centerCol := func(third int) int { return third*W/3 + W/6 }
	clearCenter := columnMean(img, centerCol(0))
	grayCenter := columnMean(img, centerCol(1))
	tintCenter := columnMean(img, centerCol(2))

	// No-absorption control: the clear sphere transmits background white.
	approxColor(t, clearCenter, &Color{1, 1, 1}, 1e-12, "control sphere renders white")

	// Gray absorber darkens every channel equally and never inverts.
	if grayCenter.R >= 1 || grayCenter.G >= 1 || grayCenter.B >= 1 {
		t.Errorf("absorbing sphere must be darker than background: %v", grayCenter)
	}
	approxEq(t, grayCenter.R, grayCenter.G, 1e-12, "gray stays neutral (R=G)")
	approxEq(t, grayCenter.G, grayCenter.B, 1e-12, "gray stays neutral (G=B)")

	// Tinted absorber: red survives best, blue worst, matching the
	// coefficient ordering 0.1 < 0.5 < 1.2.
	if !(tintCenter.R > tintCenter.G && tintCenter.G > tintCenter.B) {
		t.Errorf("tint ordering want R>G>B, got R=%v G=%v B=%v", tintCenter.R, tintCenter.G, tintCenter.B)
	}
	if tintCenter.B >= grayCenter.B {
		t.Errorf("stronger blue absorption must be darker than the gray sphere: tintB=%v grayB=%v",
			tintCenter.B, grayCenter.B)
	}

	// Center vs edge inside the gray sphere: the central column is darker
	// than a column near the silhouette (shorter refracted chord).
	edgeCol := centerCol(1) - W/6 + 4 // inside the sphere, near its left edge
	grayEdge := columnMean(img, edgeCol)
	if !(grayCenter.R < grayEdge.R) {
		t.Errorf("sphere center must be darker than edge: center=%v edge=%v", grayCenter, grayEdge)
	}

	// Different radii of the same material: the large sphere is darker at
	// its center than the small one.
	bigImg := renderSphereGrid(t, []*Dielectric{gray}, 2.0, 90, H)
	smallImg := renderSphereGrid(t, []*Dielectric{gray}, 0.5, 90, H)
	bigCenter := columnMean(bigImg, 45)
	smallCenter := columnMean(smallImg, 45)
	if !(bigCenter.R < smallCenter.R && bigCenter.G < smallCenter.G && bigCenter.B < smallCenter.B) {
		t.Errorf("larger sphere must be darker at center: big=%v small=%v", bigCenter, smallCenter)
	}

	// Every pixel stays finite and within the physical [0,1] range.
	for j := range img {
		for i := range img[j] {
			finiteColor(t, img[j][i], "rendered pixel")
			if img[j][i].R < 0 || img[j][i].G < 0 || img[j][i].B < 0 ||
				img[j][i].R > 1+1e-9 || img[j][i].G > 1+1e-9 || img[j][i].B > 1+1e-9 {
				t.Fatalf("pixel out of physical range: %v", img[j][i])
			}
		}
	}

	// Optional visual artifact for manual inspection.
	if os.Getenv("RENDER_ABSORPTION_PPM") != "" {
		path := filepath.Join("artifacts", "absorption-spheres.ppm")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		fmt.Fprintf(f, "P3\n%d %d\n255\n", W, H)
		for j := 0; j < H; j++ {
			for i := 0; i < W; i++ {
				c := img[j][i]
				ch := func(x float64) int { return int(256 * Clamp(math.Sqrt(x), 0, 0.999)) }
				fmt.Fprintf(f, "%d %d %d\n", ch(c.R), ch(c.G), ch(c.B))
			}
		}
		t.Logf("wrote %s", path)
	}
}

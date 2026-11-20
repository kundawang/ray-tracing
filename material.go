package main

import (
	"math"
	"math/rand"
)

// randFloat is indented so tests can make scattering deterministic. The
// branch is `reflectance(...) > randFloat()`, so returning 1 forces the
// refraction branch (when physically possible) and 0 forces reflection.
var randFloat = rand.Float64

// Materialer is any material that might scatter the light.
type Materialer interface {
	Scatter(rIn *Ray, rec *HitRecord) (ok bool, attenuation *Color, scattered *Ray)
}

// Lambertian is a lambertian reflectance material (like a solid object).
type Lambertian struct {
	Albedo *Color
}

// Scatter returns always true since lambertian reflectance material scatters the light.
func (l *Lambertian) Scatter(rIn *Ray, rec *HitRecord) (ok bool, attenuation *Color, scattered *Ray) {
	scatterDirection := rec.Normal.Add(NewRandomUnitVec3())

	// Catch degenerate scatter direction
	if scatterDirection.NearZero() {
		scatterDirection = rec.Normal
	}

	return true, l.Albedo, &Ray{Orig: rec.P, Dir: scatterDirection, mediumStack: rIn.mediumStack}
}

// Metal is a metalic material.
type Metal struct {
	Albedo *Color
	Fuzz   float64
}

// NewMetal creates a new metal.
func NewMetal(c *Color, f float64) *Metal {
	if f > 1 {
		f = 1
	}
	return &Metal{
		Albedo: c,
		Fuzz:   f,
	}
}

// Scatter returns true if metal material scatters the light.
func (m *Metal) Scatter(rIn *Ray, rec *HitRecord) (ok bool, attenuation *Color, scattered *Ray) {
	reflected := rIn.Dir.UnitVector().Reflect(rec.Normal)
	scattered = &Ray{
		Orig:        rec.P,
		Dir:         reflected.Add(NewRandomInUnitSphereVec3().Mult(m.Fuzz)),
		mediumStack: rIn.mediumStack,
	}
	return scattered.Dir.Dot(rec.Normal) > 0, m.Albedo, scattered
}

// Dielectric is a material with refraction.
type Dielectric struct {
	// Index of refraction.
	Ir float64

	// Absorption is the optional RGB absorption (attenuation)
	// coefficient, per unit distance, of the medium. A transmitted
	// channel of a ray that travels distance d inside the medium is
	// multiplied by exp(-Absorption.channel * d) (Beer-Lambert law).
	// A nil value (or an all-zero coefficient) leaves the medium fully
	// transparent, preserving the original white-attenuation result.
	Absorption *Color
}

// NewDielectric creates a dielectric with the given index of refraction and
// optional RGB absorption coefficient. A nil absorption means no absorption.
// Negative or non-finite absorption components are treated as zero, i.e. an
// invalid coefficient never amplifies or corrupts transmitted light.
func NewDielectric(ir float64, absorption *Color) *Dielectric {
	return &Dielectric{
		Ir:         ir,
		Absorption: validAbsorption(absorption),
	}
}

// validAbsorption returns a sanitized copy of c in which every negative or
// non-finite component is replaced by zero. nil stays nil.
func validAbsorption(c *Color) *Color {
	if c == nil {
		return nil
	}
	sanitize := func(x float64) float64 {
		if math.IsNaN(x) || math.IsInf(x, 0) || x < 0 {
			return 0
		}
		return x
	}
	return &Color{sanitize(c.R), sanitize(c.G), sanitize(c.B)}
}

// beerLambert returns the transmittance color for distance d through the
// medium: exp(-absorption * d) per channel. With no absorption configured it
// is always white, so legacy renders are unchanged.
func (d *Dielectric) beerLambert(distance float64) *Color {
	if d.Absorption == nil || distance <= 0 {
		return &Color{1, 1, 1}
	}
	// Non-finite distance must not produce NaN colors; a non-finite path
	// means no light survives.
	if math.IsNaN(distance) || math.IsInf(distance, 0) {
		return &Color{0, 0, 0}
	}
	channel := func(a float64) float64 {
		if math.IsNaN(a) || math.IsInf(a, 0) || a < 0 {
			return 1 // Defensive: constructors already reject bad coefficients.
		}
		t := math.Exp(-a * distance)
		if t < 0 || math.IsNaN(t) {
			return 0
		}
		return t
	}
	return &Color{
		R: channel(d.Absorption.R),
		G: channel(d.Absorption.G),
		B: channel(d.Absorption.B),
	}
}

// Scatter returns always true since dielectric scatters the light.
//
// Absorption bookkeeping follows the dielectric volume the ray travels in.
// The ray carries a stack of medium frames; each scattering event first adds
// the segment just travelled to the innermost frame, then:
//   - refraction into a front face pushes a frame for this dielectric (or,
//     when the innermost frame is already the same material, deepens it, so a
//     back-to-back interface with the same material never inserts an air
//     gap into the bookkeeping),
//   - refraction out of a back face pops the frame only once every nested
//     shell of the same material has been closed, and applies Beer-Lambert
//     attenuation over the whole distance accumulated in it,
//   - reflection (including total internal reflection) keeps the stack, so
//     the distance covered by a reflected ray inside the medium is counted
//     when the ray eventually leaves.
func (d *Dielectric) Scatter(rIn *Ray, rec *HitRecord) (ok bool, attenuation *Color, scattered *Ray) {
	var refractionRatio float64
	if rec.FrontFace {
		refractionRatio = 1.0 / d.Ir
	} else {
		refractionRatio = d.Ir
	}
	unitDir := rIn.Dir.UnitVector()
	cosTheta := math.Min(unitDir.Neg().Dot(rec.Normal), 1.0)
	sinTheta := math.Sqrt(1.0 - cosTheta*cosTheta)
	cannotRefract := refractionRatio*sinTheta > 1.0

	// Length of the segment of rIn ending at this hit, in scene units.
	// Rays are always built from unit directions at scatter time, so the
	// hit parameter t equals the geometric distance; prefer it and fall
	// back to the point difference for non-unit directions.
	segment := rec.T
	if dirLen := rIn.Dir.Len(); dirLen > 0 && math.Abs(dirLen-1) > 1e-12 {
		segment = rec.P.Sub(rIn.Orig).ToVec3().Len()
	}
	if segment < 0 {
		segment = 0
	}

	// Accumulate the segment inside the medium the ray was travelling in.
	stack := rIn.mediumStack
	if n := len(stack); n > 0 {
		stack[n-1].travelled += segment
	}

	var dir *Vec3
	if cannotRefract || reflectance(cosTheta, refractionRatio) > randFloat() {
		// Reflection (probabilistic or total internal reflection): the ray
		// stays inside the same volume, the frame is retained.
		dir = unitDir.Reflect(rec.Normal)
		return true, &Color{1.0, 1.0, 1.0}, &Ray{Orig: rec.P, Dir: dir, mediumStack: stack}
	}

	dir = unitDir.Refract(rec.Normal, refractionRatio)

	if rec.FrontFace {
		// Entering this dielectric from outside (or from a different
		// medium). Nested surfaces that share the exact same material are
		// treated as one continuous volume: deepen the existing frame
		// instead of pushing a second one, so no air gap is ever recorded
		// between concentric shells of the same glass.
		if n := len(stack); n > 0 && stack[n-1].mat == d {
			stack[n-1].depth++
			return true, &Color{1.0, 1.0, 1.0}, &Ray{Orig: rec.P, Dir: dir, mediumStack: stack}
		}
		newStack := make([]*mediumFrame, 0, len(stack)+1)
		newStack = append(newStack, stack...)
		newStack = append(newStack, &mediumFrame{mat: d, depth: 1})
		return true, &Color{1.0, 1.0, 1.0}, &Ray{Orig: rec.P, Dir: dir, mediumStack: newStack}
	}

	// Leaving through a back face: close one nesting level of the innermost
	// frame. The frame is popped (and Beer-Lambert attenuation applied over
	// everything accumulated in it) only when the outermost shell of the
	// material is closed; an inner shell just shrinks the depth. An empty
	// stack means the ray was already outside every dielectric (e.g. an
	// orphaned back face); treat it as a no-op rather than crashing.
	if len(stack) == 0 {
		return true, &Color{1.0, 1.0, 1.0}, &Ray{Orig: rec.P, Dir: dir}
	}
	n := len(stack)
	frame := stack[n-1]
	if frame.depth > 1 {
		frame.depth--
		return true, &Color{1.0, 1.0, 1.0}, &Ray{Orig: rec.P, Dir: dir, mediumStack: stack}
	}
	newStack := stack[: n-1 : n-1]
	return true, frame.mat.beerLambert(frame.travelled), &Ray{Orig: rec.P, Dir: dir, mediumStack: newStack}
}

func reflectance(cosine, refIdx float64) float64 {
	// Use Schlick's approximation for reflectance.
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*math.Pow((1-cosine), 5)
}

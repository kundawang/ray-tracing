package main

// Ray is a ray line in 3D dimension.
type Ray struct {
	Orig *Point3
	Dir  *Vec3

	// mediumStack holds the dielectric volumes the ray is currently
	// travelling inside, innermost last. A ray outside every dielectric
	// carries an empty stack. The stack is only ever touched by
	// Dielectric.Scatter, so non-dielectric materials leave it untouched.
	mediumStack []*mediumFrame
}

// mediumFrame records a dielectric volume a ray is inside and the distance
// the ray has already accumulated inside that volume. depth counts nested
// surfaces sharing the same dielectric material, so coincident or concentric
// shells of one material behave as one continuous volume.
type mediumFrame struct {
	mat       *Dielectric
	travelled float64
	depth     int
}

// At calculate ray position with parameter t.
func (r *Ray) At(t float64) *Point3 {
	return r.Orig.Addv(r.Dir.Mult(t))
}

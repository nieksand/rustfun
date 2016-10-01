/*
 * Bloom filter implementation with double-hashing
 *
 * Todo:
 *   1) create bloom filter with supplied hasher
 *   2) compute optimal k count
 */

use std::cmp;
use std::f64;



/*
 * Optimal hash count for given bloom filter.  This is determined by the size (#
 * bits) together with the epected cardinality of items in the filter.
 */
pub fn get_optimal_hash_count(nbits: i64, nelem: i64) -> i32 {

	// corner case
	assert!(nelem>=0 && nbits >0);
	if nelem == 0 {
		return 1;
	}

    // floor[(m/n) * ln(2)]
    let bits_per_elem = (nbits as f64) / (nelem as f64);
    let optimal_k = ( bits_per_elem * f64::consts::LN_2 ) as i32;

	// ensure at least one hash
	cmp::max(optimal_k, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_get_optimal_hash_count() {
		assert!(get_optimal_hash_count(2500, 500) == 3);
		assert!(get_optimal_hash_count(1000, 100) == 6);
		assert!(get_optimal_hash_count(3600, 300) == 8);
	}
}

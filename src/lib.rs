/*
 * Completed:
 *   - Fisher-Yates Shuffle
 *   - Boyer-Moore Majority Vote
 *   - Quick Select
 *   - Quick Sort
 *   - Insertion Sort
 *   - Bogo Sort
 *
 * Todo:
 *   1) Merge sort	- scratch implementation exists
 *   2) Heap sort
 *   3) Radix sort
 */
extern crate rand;
use rand::Rng;


/*
 * Randomly shuffle vector using Durstenfeld's variant of Fisher-Yates.
 */
fn fisher_yates_shuffle(dat: &mut Vec<i32>) {
	let mut rng = rand::thread_rng();

	let mut j = dat.len();	// offset to shuffled region
	for _ in 0..dat.len()-2 {
		let pick = rng.gen_range(0,j);
		dat.swap(pick,j-1);
		j -= 1;
	}
}

/*
 * Boyer-Moore majority vote algorithm.
 *
 * Indicates which element has a strict majority--that is more occurrences than
 * all other elements combined.  This implementation combines the pairing phase
 * from the original BM paper with the counting phase needed to verify that the
 * strict majority exists.
 *
 * The result is None if no strict majority element exists.
 *
 * It is linear time (2 passes over data) and has a tiny, fixed memory usage
 * regardless of the number of candidates.
 *
 * MJRTY - A Fast Majority Vote Algorithm, with R.S. Boyer. In R.S. Boyer (ed.),
 * Automated Reasoning: Essays in Honor of Woody Bledsoe, Automated Reasoning
 * Series, Kluwer Academic Publishers, Dordrecht, The Netherlands, 1991, pp.
 * 105-117.
 */
fn bm_majority_vote(dat: &Vec<i32>) -> Option<i32> {
	// no majority on empty set
	if dat.len() == 0 {
		return None;
	}

	// pairing phase -- find candidate assuming strict majority exists
	let mut candidate : i32 = dat[0];
	let mut count : i64 = 1;

	for i in 1..dat.len() {
		if count == 0 {
			candidate = dat[i];
			count = 1;
		} else if candidate == dat[i] {
			count += 1;
		} else {
			count -= 1;
		}
	}

	// counting phase -- verify that candidate actually has strict majority
	count = 0;
	let threshold : i64 = (dat.len() as i64) / 2;
	for v in dat {
		if *v == candidate {
			count += 1;
			if count > threshold {
				return Some(candidate);
			}
		}
	}

	// no strict majority existed
	return None;
}

/*
 * Quicksort with random pivoting.  Kept it as a pure implementation; does not
 * switch to a non-recursive sort at small partition sizes.
 */
fn quick_sort(dat: &mut Vec<i32>) {
	let max = dat.len();
	quick_sort_int(dat, 0, max);
}

fn quick_sort_int(dat: &mut Vec<i32>, min: usize, max: usize) {
	assert!(min <= max, "qsort min extent gt max extent");
	assert!(max <= dat.len(), "qsort max extent gt vector len");

	// recursion base case
	if max - min <= 1 {
		return;
	}

	// random pivot
	dat.swap(min, rand::thread_rng().gen_range(min, max));
	let pidx = partition(dat, min, max);

	// sort subpartitions
	quick_sort_int(dat, min, pidx);
	quick_sort_int(dat, pidx+1, max);
}

/*
 * Partition orders an array around a pivot value.  This logic is key to both
 * the quicksort and quickselect algorithms.
 *
 * The input has the pivot as the 0th element:
 *   [pivot, other values]
 *
 * The output has the array partitioned:
 *   [values <= pivot, pivot, values > pivot]
 *
 * It returns the final index of the pivot.
 *
 * Note that you can control the pivot (e.g. choose one randomly) by swapping
 * the desired value to the 0th position before invoking this routine.
 */
fn partition(dat: &mut Vec<i32>, min: usize, max: usize) -> usize {
	assert!(max-min > 0, "partition requires pivot in 0th position");

	// bounds (min,pleft] and (pright,max) are partitioned.
	let mut pleft = min;
	let mut pright = max;

	while pleft+1 < pright {
		// value already in correct partition
		if dat[pleft+1] <= dat[min] {
			pleft += 1;
		}
		// value belongs in other partition
		else {
			dat.swap(pleft+1,pright-1);
			pright -= 1;
		}
	}

	// pivot lives at end of left partition
	dat.swap(min,pleft);
	pleft
}

/*
 * Quickselect with random pivoting.
 *
 * This picks the k-th smallest element from values in the array.
 *
 * Algorithm has same basic idea of quicksort, but you don't have to fully sort
 * the array.  Instead observe that each partition sweep guarantees that the
 * pivot will be placed in the correct, final location.  So each time you check
 * where the kth element falls relative to the pivot and only recurse on that
 * side.  You stop once the pivot=kth element.
 */
fn quick_select(dat: &mut Vec<i32>, k: usize) -> i32 {
	assert!(k >= 0 && k < dat.len(), "k-th element not in data bounds");

	let max = dat.len();
	quick_select_int(dat, 0, max, k);
	dat[k]
}

fn quick_select_int(dat: &mut Vec<i32>, min: usize, max: usize, k: usize) {
	assert!(min <= max, "qselect min extent gt max extent");
	assert!(max <= dat.len(), "qselect max extent gt vector len");

	// recursion base case
	if max - min <= 1 {
		return;
	}

	// random pivot
	dat.swap(min, rand::thread_rng().gen_range(min, max));
	let pidx = partition(dat, min, max);

	// process only subpartition needed to position kth element
	if k < pidx {
		quick_select_int(dat, min, pidx, k);
	} else if k > pidx {
		quick_select_int(dat, pidx+1, max, k);
	} else {
		return;
	}
}

/*
 * Insertion sort.  Stable and can support online sorts, but quadratic.
 * Decrementing loops are a suprising pita in Rust.
 */
fn insertion_sort(dat: &mut Vec<i32>) {
	// outer loop tracks sorted region
	for i in 1..dat.len() {

		// slide next unsorted element to correct place
		// [1,i+1) reversed is (i+i,1] = [i,0)
		for j in (1..i+1).rev() {
			if dat[j] < dat[j-1] {
				dat.swap(j,j-1);
			} else {
				break
			}
		}
	}
}

/*
 * Bogosort!  Just for fun.  Optimized build can handle about size 10 inputs.
 */
fn bogo_sort(dat: &mut Vec<i32>) {
	let mut sorted = false;
	while !sorted {
		// randomly shuffle input
		fisher_yates_shuffle(dat);

		// check if we got a sorted set
		sorted = is_sorted(&dat);
	}
}

/*
 * [incomplete] Merge sort implementation.
 */
fn merge_sort(dat: &mut Vec<i32>) {

	let mut scratch : Vec<i32> = Vec::with_capacity(dat.len());

	let max = dat.len();
	merge_sort_int(dat, 0, max, &mut scratch);
}

fn merge_sort_int(dat: &mut Vec<i32>, min: usize, max: usize, scratch: &mut Vec<i32>) {

	// empty and single-element list already sorted
	if max-min <= 1 {
		return;
	}

	// split in two and sort each chunk
	let mid = (max+min)/2;
	merge_sort_int(dat, min, mid, scratch);
	merge_sort_int(dat, mid, max, scratch);

	// combine sorted chunks
	combine_chunks(dat, min, mid, mid, max, scratch);
}

fn combine_chunks(dat: &mut Vec<i32>, lmin : usize, lmax : usize, rmin : usize, rmax : usize, scratch: &mut Vec<i32>) {

	assert!(lmax == rmin, "mergesort combine chunks must be adjacent");

	scratch.clear();
	let mut li : usize = lmin;
	let mut ri : usize = lmax;

	// merge two sorted lists
	while li < lmax && ri < rmax {
		if dat[li] < dat[ri] {
			scratch.push(dat[li]);
			li += 1;
		} else {
			scratch.push(dat[ri]);
			ri += 1;
		}
	}

	// drain remainder
	for i in li..lmax {
		scratch.push(dat[i]);
	}
	for i in ri..rmax {
		scratch.push(dat[i]);
	}

	// scratch back to output
	for i in lmin..rmax {
		dat[i] = scratch[i-lmin]
	}
}

/*
 * Verify if vector is sorted.
 */
fn is_sorted(dat: &Vec<i32>) -> bool {
	for i in 1..dat.len() {
		if dat[i] < dat[i-1] {
			return false;
		}
	}
	true
}


#[test]
fn test_quick_sort() {
	let mut dat: Vec<i32> = (0..5000).collect();
	fisher_yates_shuffle(&mut dat);
	quick_sort(&mut dat);
	assert!(is_sorted(&dat), "result not properly sorted");
}

#[test]
fn test_merge_sort() {
	let mut dat: Vec<i32> = (0..5000).collect();
	fisher_yates_shuffle(&mut dat);
	merge_sort(&mut dat);
	assert!(is_sorted(&dat), "result not properly sorted");
}

#[test]
fn test_insertion_sort() {
	let mut dat: Vec<i32> = (0..5000).collect();
	fisher_yates_shuffle(&mut dat);
	insertion_sort(&mut dat);
	assert!(is_sorted(&dat), "result not properly sorted");
}

#[test]
fn test_bogo_sort() {
	let mut dat: Vec<i32> = (0..8).collect();
	fisher_yates_shuffle(&mut dat);
	bogo_sort(&mut dat);
	assert!(is_sorted(&dat), "result not properly sorted");
}

#[test]
fn test_is_sorted() {
	let v1 = vec![];
	assert!(is_sorted(&v1), "empty array always sorted");

	let v2 = vec![3];
	assert!(is_sorted(&v2), "single element array always sorted");

	let v3 = vec![-1, 0, 5];
	assert!(is_sorted(&v3), "rejected a sorted sequence");

	let v4 = vec![0, 0, 0];
	assert!(is_sorted(&v4), "rejected all-same sequence");

	let v5 = vec![5, 3, 8];
	assert!(!is_sorted(&v5), "accepted unsorted sequence");
}

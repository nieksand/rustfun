/*
 * Completed:
 *   - Fisher-Yates Shuffle
 *   - Boyer-Moore Majority Vote
 *   - Partition by Pivot
 *   - Quick Sort
 *   - Quick Select
 *   - Naive Select
 *   - Implicit Max Heap
 *   - Heap Sort
 *   - Merge Sort
 *   - Insertion Sort
 *   - Selection Sort
 *   - Shaker Sort
 *   - Bubble Sort
 *   - Bogo Sort
 *   - Binary Search
 *   - Jump Search
 *   - Reverse Sequence
 *
 * Todo:
 *   1) Radix sort
 *   2) Heap's Permutations
 *   3) Shell sort
 *   4) In-situ permutation
 *   5) Indirect sort
 *   6) List sorts
 *   7) Templatized hybrid sort helper
 *   8) External sort
 *   9) Three way quicksort
 *   10) Templatize the sorts
 *
 * Maybe:
 *   1) Interpolation Search
 *   2) Variance calculation
 *   3) Approximate Counting (Morris)
 */
extern crate rand;

use self::rand::Rng;


/*
 * Randomly shuffle vector using Durstenfeld's variant of Fisher-Yates.
 */
pub fn fisher_yates_shuffle(dat: &mut [i32]) {
	// nothing to do
	if dat.len() < 2 {
		return;
	}

	// offset to shuffled region
	let mut j = dat.len();

	// randomly pick element to move to shuffled region
	let mut rng = rand::thread_rng();
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
pub fn bm_majority_vote(dat: &[i32]) -> Option<i32> {
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
 * Partition orders an array around a pivot value.  This logic is key to both
 * the quicksort and quickselect algorithms.
 *
 * The input has the pivot as the 0th element:
 *   [pivot, other values]
 *
 * The output has the array partitioned:
 *   [values <= pivot, pivot, values > pivot]
 *
 * It returns the final index of the pivot relative to the slice origin.
 *
 * Note that you can control the pivot (e.g. choose one randomly) by swapping
 * the desired value to the 0th position before invoking this routine.
 */
pub fn partition(dat: &mut [i32]) -> usize {
	assert!(dat.len() > 0, "partition requires pivot in 0th position");

	// bounds (min,pleft] and (pright,max) are partitioned.
	let mut pleft = 0;
	let mut pright = dat.len();

	while pleft+1 < pright {
		// value already in correct partition
		if dat[pleft+1] <= dat[0] {
			pleft += 1;
		}
		// value belongs in other partition
		else {
			dat.swap(pleft+1,pright-1);
			pright -= 1;
		}
	}

	// pivot lives at end of left partition
	dat.swap(0,pleft);
	pleft
}

/*
 * Quicksort with random pivoting.
 *
 * Kept it as a pure implementation; does not switch to a non-recursive sort at
 * small partition sizes.
 */
pub fn quick_sort(dat: &mut [i32]) {
	let max = dat.len();
	quick_sort_int(dat, 0, max);
}

fn quick_sort_int(dat: &mut [i32], min: usize, max: usize) {
	assert!(min <= max, "qsort min extent gt max extent");
	assert!(max <= dat.len(), "qsort max extent gt vector len");

	// recursion base case
	if max - min < 2 {
		return;
	}

	// random pivot
	dat.swap(min, rand::thread_rng().gen_range(min, max));
	let pidx = partition(&mut dat[min..max]) + min;

	// sort subpartitions
	quick_sort_int(dat, min, pidx);
	quick_sort_int(dat, pidx+1, max);
}

/*
 * Naive method for selecting k-th smallest element.
 *
 * Just sorts and grabs.
 */
pub fn naive_select(dat: &mut [i32], k: usize) -> i32 {
	assert!(k < dat.len(), "k-th element not in data bounds");
	quick_sort(dat);
	dat[k]
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
pub fn quick_select(dat: &mut [i32], k: usize) -> i32 {
	assert!(k < dat.len(), "k-th element not in data bounds");

	let max = dat.len();
	quick_select_int(dat, 0, max, k);
	dat[k]
}

fn quick_select_int(dat: &mut [i32], min: usize, max: usize, k: usize) {
	assert!(min <= max, "qselect min extent gt max extent");
	assert!(max <= dat.len(), "qselect max extent gt vector len");

	// recursion base case
	if max - min < 2 {
		return;
	}

	// random pivot
	dat.swap(min, rand::thread_rng().gen_range(min, max));
	let pidx = partition(&mut dat[min..max]) + min;

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
 * Make implicit binary max-heap data structure.
 *
 * Uses following packing scheme to map heap elements to array:
 *     root is at 0
 *     parent(i)     = floor( (i-1)/2 )
 *     leftchild(i)  = 2*i+1
 *     rightchild(i) = 2*i+2
 *
 */
pub fn make_implicit_max_heap(dat: &mut [i32]) {
	// nothing to do
	if dat.len() < 2 {
		return;
	}

	fn parent_idx(node_idx: usize) -> usize {
		return (node_idx-1)/2;
	}

	// start by establish heap property on far-right subtree
	// sweep there to begin of array, inclusive
	let end = dat.len();
	let far_right = parent_idx(end-1);
	for i in 0..far_right+1 {
		let root_idx = far_right - i;
		sift_down(&mut dat[..], root_idx);
	}
}

fn sift_down(dat: &mut [i32], start: usize) {
	fn left_child_idx(node_idx: usize) -> usize {
		return 2*node_idx+1;
	}

	// element may violate heap property; rest of array must be valid
	let mut wiggle_idx = start;

	// swap wiggle element downwards with largest child until heap property
	// re-established
	while left_child_idx(wiggle_idx) < dat.len() {

		let mut swap_target = wiggle_idx;

		let left_idx = left_child_idx(wiggle_idx);
		let right_idx = left_idx + 1;

		// loop condition ensures child exists
		if dat[left_idx] > dat[wiggle_idx] {
			swap_target = left_idx;
		}

		// right child greater than wiggle and left child
		if right_idx < dat.len() && dat[right_idx] > dat[swap_target] {
			swap_target = right_idx;
		}

		// heap property was established!
		if swap_target == wiggle_idx {
			return;
		}

		dat.swap(wiggle_idx, swap_target);
		wiggle_idx = swap_target;
	}
}

/*
 * Heap sort.
 */
pub fn heap_sort(dat: &mut [i32]) {
	// nothing to do
	if dat.len() < 2 {
		return;
	}

	make_implicit_max_heap(dat);

	// swap top of heap (max val) to sorted region being built at end
	// reestablish heap property by shifting down the element we swapped
	// inwards.
	let len = dat.len();
	for i in 0..len {
		dat.swap(0, len-i-1);
		sift_down(&mut dat[0..len-i-1], 0);
	}
}

/*
 * Merge sort.  This is a top-down implementation.
 *
 * Kept it as a pure implementation; does not switch to a non-recursive sort at
 * small partition sizes.
 */
pub fn merge_sort(dat: &mut [i32]) {
	// requires O(n) scratch space
	let mut scratch : Vec<i32> = Vec::with_capacity(dat.len());
	let max = dat.len();
	merge_sort_int(dat, 0, max, &mut scratch);
}

fn merge_sort_int(dat: &mut [i32], min: usize, max: usize, scratch: &mut Vec<i32>) {

	// empty and single-element list already sorted
	if max - min < 2 {
		return;
	}

	// split in two and sort each chunk
	let mid = min + (max-min)/2;
	merge_sort_int(dat, min, mid, scratch);
	merge_sort_int(dat, mid, max, scratch);

	// combine sorted chunks
	combine_chunks(dat, min, mid, max, scratch);
}

fn combine_chunks(dat: &mut [i32], lmin : usize, mid : usize, rmax : usize, scratch: &mut Vec<i32>) {

	scratch.clear();
	let mut li : usize = lmin;
	let mut ri : usize = mid;

	// merge two sorted lists
	while li < mid && ri < rmax {
		if dat[li] < dat[ri] {
			scratch.push(dat[li]);
			li += 1;
		} else {
			scratch.push(dat[ri]);
			ri += 1;
		}
	}

	// drain remainder
	if li < mid {
		scratch.extend_from_slice(&dat[li..mid]);
	} else if ri < rmax {
		scratch.extend_from_slice(&dat[ri..rmax]);
	}

	// scratch back to output
	dat[lmin..rmax].clone_from_slice(scratch);
}

/*
 * Insertion sort.  Stable and can be made online, but quadratic.
 */
pub fn insertion_sort(dat: &mut [i32]) {
	// nothing to do
	if dat.len() < 2 {
		return;
	}

	// place smallest value at 0th index as sentinel
	let mut sidx = 0;
	for i in 1..dat.len() {
		if dat[i] < dat[sidx] {
			sidx = i;
		}
	}
	dat.swap(0, sidx);

	// outer loop tracks sorted region
	for i in 2..dat.len() {

		// slide next unsorted element to correct place
		let tmp = dat[i];
		let mut j = i;
		while tmp < dat[j-1] {
			dat[j] = dat[j-1];
			j -= 1;
		}
		dat[j] = tmp;
	}
}

/*
 * Selection sort.  Quadratic and I have never encountered a real use case for
 * it.  Sedgewick points out that it does the minimum number of swaps, so is
 * plausible for huge items with small keys.  But honestly, just do an indirect
 * sort for that.
 */
pub fn selection_sort(dat: &mut [i32]) {
	for i in 0..dat.len() {
		let mut min = i;
		for j in i+1..dat.len() {
			if dat[j] < dat[min] {
				min = j;
			}
		}
		dat.swap(i,min);
	}
}

/*
 * Shaker sort.  Just for fun.
 *
 * This is a variant of bubble sort mentioned by Sedgewick where you basically
 * alternate the direction of bubbling in each pass.
 */
pub fn shaker_sort(dat: &mut [i32]) {

	// sorted bounds [0,min) and [max,len)
	let mut min : i64 = 0;
	let mut max : i64 = dat.len() as i64;

	// direction of shake
	let mut dir: i64 = 1;

	while min != max {

		if dir == 1 {
			for i in min..max-1 {
				let cur = i as usize;
				let nex = (i+dir) as usize;

				if dat[cur] > dat[nex] {
					dat.swap(cur,nex);
				}
			}
			max -= 1
		} else {
			for i in (min+1..max+1).rev() {
				let cur = i as usize;
				let nex = (i+dir) as usize;

				if dat[cur] < dat[nex] {
					dat.swap(cur,nex);
				}
			}
			min += 1
		}
		dir = -dir;
	}
}

/*
 * Bubble sort.  Just for completeness.
 */
pub fn bubble_sort(dat: &mut [i32]) {
	// nothing to do
	if dat.len() < 2 {
		return;
	}

	for i in 1..dat.len() {
		let mut swapped = false;
		for j in 0..dat.len()-i {
			if dat[j] > dat[j+1] {
				dat.swap(j,j+1);
				swapped = true;
			}
		}

		if !swapped {
			break;
		}
	}
}

/*
 * Bogosort!  Just for fun.  Optimized build can handle about size 10 inputs.
 */
pub fn bogo_sort(dat: &mut [i32]) {
	let mut sorted = false;
	while !sorted {
		// randomly shuffle input
		fisher_yates_shuffle(dat);

		// check if we got a sorted set
		sorted = is_sorted(&dat);
	}
}

/*
 * Binary search.  Input must already be sorted.
 */
pub fn binary_search(dat: &[i32], searchval: i32) -> Option<usize> {
	if dat.len() == 0 {
		return None;
	}

	// searchval, if it exists, is always in [min,max)
	let mut min = 0;
	let mut max = dat.len();
	while min < max {
		let mid = (max+min)/2;

		// excluded [mid, max) so search [min,mid)
		if dat[mid] > searchval {
			max = mid;
		}
		// excluded [min,mid] so search [mid+1,max)
		else if dat[mid] < searchval {
			min = mid+1;
		}
		// direct hit
		else {
			return Some(mid);
		}
	}

	// [min,max) now empty range, value can not exist
	return None;
}

/*
 * Jump search.  Input must already be sorted.
 *
 * This algorithm is just for fun since it is O(sqrt(n)) versus the O(log(n)) of
 * binary search.  Conceivably it would be useful if you had a physical tape
 * that you were scanning where performance was dominated by seek rather than
 * read time.
 */
pub fn jump_search(dat: &[i32], searchval: i32) -> Option<usize> {
	if dat.len() == 0 {
		return None;
	}

	let jumpdist = (dat.len() as f64).sqrt() as usize;
	let mut j = 0;
	while j < dat.len() && dat[j] <= searchval {
		j += jumpdist;
	}

	let i = if j >= jumpdist { j-jumpdist } else { 0 };
	j = if j <= dat.len() { j } else { dat.len() };

	for idx in i..j {
		if dat[idx] == searchval {
			return Some(idx);
		}
	}

	return None;
}



/*
 * incomplete - hyperghetto stub of interpolation search.
 */
pub fn interpolation_search(dat: &[i32], searchval: i32) -> Option<usize> {
	if dat.len() == 0 {
		return None;
	}

	// searchval, if it exists, is always in [min,max)
	let mut min = 0;
	let mut max = dat.len();
	while min < max {

		let x1 = min as f64;
		let x2 = (max-1) as f64;
		let y1 = dat[min] as f64;
		let y2 = dat[max-1] as f64;
		let m = (y1-y2)/(x1-x2);
		let b = y1 - m * x1;

		let mut xn = ((searchval as f64 - b) / m) as usize;
		xn = if xn < min { min } else { xn };
		xn = if xn >= max { max-1 } else { xn };

		// excluded [xn, max) so search [min,xn)
		if dat[xn] > searchval {
			max = xn;
		}
		// excluded [min,xn] so search [xn+1,max)
		else if dat[xn] < searchval {
			min = xn+1;
		}
		// direct hit
		else {
			return Some(xn);
		}
	}

	// [min,max) now empty range, value can not exist
	return None;
}



/*
 * Verify if vector is sorted.
 */
pub fn is_sorted(dat: &[i32]) -> bool {
	for i in 1..dat.len() {
		if dat[i] < dat[i-1] {
			return false;
		}
	}
	true
}

/*
 * Reverse order of elements in vector.
 *
 * Rust has a built-in for this; this implementation is just for fun.
 */
pub fn reverse(dat: &mut [i32]) {
	let len = dat.len();
	for i in 0..len/2 {
		dat.swap(i,len-i-1);
	}
}

/*
 * Checks if a string is an anagram.
 *
 * This code is clunkier and more inefficient than it needs to be.  Rust strings
 * are UTF8 sequences which means iteration over direct indexing.  I'm not too
 * familiar with my options yet, so I ended up with the cruft before.  Also note
 * that this iterates the full string rather than to the half-way point.
 */
pub fn is_anagram(dat: &str) -> bool {
	let fwd = dat.chars();
	let mut rev = dat.chars().rev();
	for ch in fwd {
		match rev.next() {
			None => panic!("rev iterator shorter than fwd"),
			Some(rch) => {
				if ch != rch { return false; }
			}
		}
	}
	true
}

/*
 * Remove duplicates from an array.
 *
 * Sorts to make life easy.  Does transform in-place, truncating away any dupes.
 */
pub fn dedupe(dat: &mut Vec<i32>) {
	if dat.len() == 0 {
		return;
	}

	quick_sort(dat);

	// next index to write non-dupe value to
	let mut widx: usize = 1;
	let mut last = dat[0];

	// shift left to overwrite dupes
	for ridx in 1..dat.len() {
		if dat[ridx] != last {
			dat[widx] = dat[ridx];
			last = dat[ridx];
			widx += 1;
		}
	}

	// truncate garbage at end
	dat.resize(widx, -1);
}


#[cfg(test)]
mod tests {
    use super::*;

	fn sort_eval<F>(randsize: i32, sortfn: F)
		// decently sized random vector
		where F: Fn(&mut [i32]) -> () {
		let mut dat: Vec<i32> = (0..randsize).collect();
		fisher_yates_shuffle(&mut dat);
		sortfn(&mut dat);
		assert!(is_sorted(&dat), "result not properly sorted");

		// try degenerate and small cases
		for n in 0..6 {
			dat = (0..n).collect();
			fisher_yates_shuffle(&mut dat);
			sortfn(&mut dat);
			assert!(is_sorted(&dat), "result not properly sorted");
		}
	}

	#[test]
	fn test_fisher_yates_shuffle() {
		// start with identical sequences
		let origin: Vec<i32> = (0..120).collect();
		let mut v1 = origin.clone();
		let mut v2 = origin.clone();
		assert!(v1 == v2, "start vectors unequal");

		// 120! permutations so odds of chance collision infinitesimal
		fisher_yates_shuffle(&mut v1);
		fisher_yates_shuffle(&mut v2);
		assert!(v1 != v2, "shuffled vectors equal");
		assert!(v1 != origin, "shuffled v1 equal still in order");
		assert!(v2 != origin, "shuffled v2 equal still in order");

		// verify all elements still there
		quick_sort(&mut v1);
		quick_sort(&mut v2);
		assert!(v1 == origin, "resorted v1 missing elements");
		assert!(v2 == origin, "resorted v2 missing elements");
	}

	#[test]
	fn test_fisher_yates_shuffle_degenerate() {
		let mut v1 : Vec<i32> = vec![];
		fisher_yates_shuffle(&mut v1);
		assert!(v1 == vec![], "empty vector unchanged by shuffle");

		let mut v2 : Vec<i32> = vec![6];
		fisher_yates_shuffle(&mut v2);
		assert!(v2 == vec![6], "one-element vector unchanged by shuffle");

		let mut v3 : Vec<i32> = vec![10,15];
		fisher_yates_shuffle(&mut v3);
		assert!(v3.len() == 2, "two-element vector same size post-shuffle");
	}

	#[test]
	fn test_bm_majority_vote() {
		// majority of 1s
		let v1 = vec![0,1,0,1,1];
		let m1 = bm_majority_vote(&v1);
		assert!(m1 == Some(1));

		// no majority
		let v2 = vec![0,1,0,1,1,0];
		let m2 = bm_majority_vote(&v2);
		assert!(m2 == None);

		// majority of 1s but not strict majority
		let v3 = vec![2,2,0,1,0,1,1];
		let m3 = bm_majority_vote(&v3);
		assert!(m3 == None);

		// empty input
		let v4 = vec![];
		let m4 = bm_majority_vote(&v4);
		assert!(m4 == None);

		// lonely majority
		let v5 = vec![6];
		let m5 = bm_majority_vote(&v5);
		assert!(m5 == Some(6));
	}

	#[test]
	fn test_partition() {
		fn left_ok(dat: &Vec<i32>, bound: i32, min: usize, max: usize) -> bool {
			for i in min..max {
				if dat[i] > bound {
					return false;
				}
			}
			true
		}
		fn right_ok(dat: &Vec<i32>, bound: i32, min: usize, max: usize) -> bool {
			for i in min..max {
				if dat[i] <= bound {
					return false;
				}
			}
			true
		}

		// empty right partition
		let mut v1 = vec![1, 0, 0, 0];
		let p1 = partition(&mut v1[..]);
		assert!(p1 == 3, "pivot not in right final location");
		assert!(left_ok(&v1, 1, 0, p1), "left partition invalid");
		assert!(right_ok(&v1, 1, p1+1, 4), "right partition invalid");

		// empty left partition
		let mut v2 = vec![0, 1, 1, 1];
		let p2 = partition(&mut v2[..]);
		assert!(p2 == 0, "pivot not in right final location");
		assert!(left_ok(&v2, 0, 0, p2), "left partition invalid");
		assert!(right_ok(&v2, 0, p2+1, 4), "right partition invalid");

		// partition on each side
		let mut v3 = vec![3, 5, 0, 1, 2, 4];
		let p3 = partition(&mut v3[..]);
		assert!(p3 == 3, "pivot not in right final location");
		assert!(left_ok(&v3, 3, 0, p3), "left partition invalid");
		assert!(right_ok(&v3, 3, p3+1, 6), "right partition invalid");

		// both partitions empty
		let mut v4 = vec![42];
		let p4 = partition(&mut v4[..]);
		assert!(p4 == 0, "pivot not in right final location");
		assert!(left_ok(&v4, 42, 0, p4), "left partition invalid");
		assert!(right_ok(&v4, 42, p4+1, 1), "right partition invalid");
	}

	#[test]
	fn test_quick_sort() {
		sort_eval(5000, quick_sort);
	}

	#[test]
	fn test_naive_select() {
		let mut dat: Vec<i32> = (0..100).collect();
		fisher_yates_shuffle(&mut dat);

		let k10 = naive_select(&mut dat, 10);
		assert!(k10 == 10, "naive select did not pick 10th element");

		let k99 = naive_select(&mut dat, 99);
		assert!(k99 == 99, "naive select did not pick 99th element");
	}

	#[test]
	fn test_quick_select() {
		let mut dat: Vec<i32> = (0..100).collect();
		fisher_yates_shuffle(&mut dat);

		let k10 = quick_select(&mut dat, 10);
		assert!(k10 == 10, "qselect did not pick 10th element");

		let k99 = quick_select(&mut dat, 99);
		assert!(k99 == 99, "qselect did not pick 99th element");
	}

	#[test]
	fn test_quick_select_degenerate() {
		let mut v1: Vec<i32> = vec![42];
		assert!(quick_select(&mut v1, 0) == 42);
	}

	#[test]
	fn test_make_implicit_max_heap() {
		// try degenerate and balanced/unbalanced cases
		for n in 0..10 {
			let mut dat: Vec<i32> = (0..n).collect();
			make_implicit_max_heap(&mut dat);

			// verify heap property, parent >= child
			for i in 1..dat.len() {
				let parent_idx = (i-1)/2;
				assert!(dat[parent_idx] >= dat[i], "max heap property violated");
			}
		}
	}

	#[test]
	fn test_heap_sort() {
		sort_eval(5000, heap_sort);
	}

	#[test]
	fn test_merge_sort() {
		sort_eval(5000, merge_sort);
	}

	#[test]
	fn test_insertion_sort() {
		sort_eval(5000, insertion_sort);
	}

	#[test]
	fn test_selection_sort() {
		sort_eval(5000, selection_sort);
	}

	#[test]
	fn test_shaker_sort() {
		sort_eval(5000, shaker_sort);
	}

	#[test]
	fn test_bubble_sort() {
		sort_eval(5000, bubble_sort);
	}

	#[test]
	fn test_bogo_sort() {
		sort_eval(6, bogo_sort);
	}

	#[test]
	fn test_binary_search() {
		let dat: Vec<i32> = (0..10).collect();
		for i in 0..dat.len() {
			let res = binary_search(&dat, dat[i]);
			assert!(res == Some(i), "binary search should have hit");
		}

		let r1 = binary_search(&dat, -1);
		assert!(r1 == None, "binary search should have missed");

		let r2 = binary_search(&dat, 1000);
		assert!(r2 == None, "binary search should have missed");
	}

	#[test]
	fn test_jump_search() {
		// jump over arrays of different sizes
		for n in 1..101 as usize {
			// make sure we can hit every value inside
			let dat: Vec<i32> = (0..n as i32).collect();
			for i in 0..n {
				let res = jump_search(&dat, dat[i]);
				assert!(res == Some(i), "jump search should have hit");
			}

			// miss on purpose
			let r1 = jump_search(&dat, -1);
			assert!(r1 == None, "jump search should have missed");

			let r2 = jump_search(&dat, 1000);
			assert!(r2 == None, "jump search should have missed");
		}

		// degenerate case
		let degen: Vec<i32> = vec![];
		let dres = jump_search(&degen, 100);
		assert!(dres == None);
	}

	#[test]
	fn test_interpolation_search() {
		let dat: Vec<i32> = (0..10).collect();
		for i in 0..dat.len() {
			let res = interpolation_search(&dat, dat[i]);
			assert!(res == Some(i), "interpolation search should have hit");
		}

		let r1 = interpolation_search(&dat, -1);
		assert!(r1 == None, "interpolation search should have missed");

		let r2 = interpolation_search(&dat, 1000);
		assert!(r2 == None, "interpolation search should have missed");
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

	#[test]
	fn test_reverse() {
		let mut v1: Vec<i32> = vec![];
		reverse(&mut v1);
		assert!(v1 == vec![], "identity for reversed empty vector");

		let mut v2: Vec<i32> = vec![6];
		reverse(&mut v2);
		assert!(v2 == vec![6], "identity for reversed one element vector");

		let mut v3: Vec<i32> = vec![1,2,3,4];
		reverse(&mut v3);
		assert!(v3 == vec![4,3,2,1], "reversed even-count vector");

		let mut v4: Vec<i32> = vec![1,2,3,4,5];
		reverse(&mut v4);
		assert!(v4 == vec![5,4,3,2,1], "reversed odd-count vector");

		let mut v5: Vec<i32> = (0..10000).collect();
		fisher_yates_shuffle(&mut v5);
		let v5_orig = v5.clone();
		reverse(&mut v5);
		reverse(&mut v5);
		assert!(v5 == v5_orig, "double reverse should equal start");
	}

	#[test]
	fn test_is_anagram() {
		assert!(!is_anagram("potato"));
		assert!(is_anagram("racecar"));
		assert!(is_anagram("amanaplanacanalpanama"));
		assert!(is_anagram(""));
		assert!(is_anagram("a"));
		assert!(is_anagram("moo oom"));
		assert!(!is_anagram("hello world"));
	}

	#[test]
	fn test_dedupe() {
		let ins = vec![
			vec![],
			vec![1,2,3,4,5],
			vec![1,1,1,2,2,2],
			vec![1,2,3,4,2],
			vec![1,1,1,1,1]];

		let truths = vec![
			vec![],
			vec![1,2,3,4,5],
			vec![1,2],
			vec![1,2,3,4],
			vec![1]];

		for i in 0..ins.len() {
			let mut dat = ins[i].clone();
			dedupe(&mut dat);
			let msg = format!("{:?} != {:?}", dat, truths[i]);
			assert!(dat == truths[i], msg);
		}
	}
}

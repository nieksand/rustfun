/*
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
use std::collections::HashMap;


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
    for _ in 0..dat.len()-1 {
        let pick = rng.gen_range(0,j);
        dat.swap(pick,j-1);
        j -= 1;
    }
}

/*
 * Heap's permutations
 *
 * Generates all permutations of a sequence with each iteration requiring only a
 * single swap of elements.
 *
 * The inductive reasoning:
 *
 * - permutation of 1 element array is a base case (n=1)
 *
 * - assume you have method to permute an n-1 element array
 *
 * - generate permutation for n element array by holding nth element fixed and
 *   permuting n-1 elements.  Then swap n and one of n-1 and permute again.
 *   Repeat this process until each of the n-1th elements has held the nth
 *   position.
 *
 * A direct implementation leads to a very understandable algorithm but with an
 * extra swap.  Generate permutations using the nth element as-is.  Then looping
 * with i=0..upto_idx: swap the ith and upto_idx values, generate permutations,
 * swap ith and upto_idx values back.
 *
 * The actual algorithm avoids the swap-back by exploiting the specific n-1
 * permutation to place a distinct element in the nth position.
 *
 * Specifically if the subsequence is...
 *   Odd-length:  swap [0,n] after each iteration
 *   Even-length: swap [i,n] after each iteration
 *
 *
 * Why?!?
 *
 *
 * Heap, B. R. (1963). "Permutations by Interchanges". The Computer
 * Journal. 6 (3): 293–4. doi:10.1093/comjnl/6.3.293.
 */
pub fn heaps_permutations(dat: &mut [i32], gathercb: &Fn(&mut [i32]) -> ()) {

	// degenerate case
	if dat.len() == 0 {
		return
	}

	// upto_idx is the inclusive index up to which we permute values
    let upto_idx = dat.len()-1;
    heaps_permutations_int(dat, upto_idx, gathercb)
}

fn heaps_permutations_int(dat: &mut [i32], upto_idx: usize, gathercb: &Fn(&mut [i32]) -> ()) {

    // permuting up to index 0 (inclusive) is base case
    if upto_idx == 0 {
		// INVOKE CALLBACK
        return;
    }

    // generate permutations given last element
    heaps_permutations_int(dat, upto_idx-1, gathercb);

    // swap from n-1th to nth, generate permutations
    for i in 0..upto_idx {
        // swap in
        let t = dat[i];
        dat[i] = dat[upto_idx];
        dat[upto_idx] = t;

        heaps_permutations_int(dat, upto_idx-1, gathercb);

        // swap out
        let t = dat[i];
        dat[i] = dat[upto_idx];
        dat[upto_idx] = t;
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
    let mut candidate: i32 = dat[0];
    let mut count: i64 = 1;

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
    let threshold: i64 = (dat.len() as i64) / 2;
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
 * Hash table majority vote algorithm.
 *
 * Indicates which element has a strict majority--that is more occurrences than
 * all other elements combined.  The result is None if no strict majority
 * element exists.
 *
 * It is linear time assuming no pathological hash behavior (1 pass over data)
 * and has memory usage proportional to the number of distinct candidates.
 */
pub fn hash_majority_vote(dat: &[i32]) -> Option<i32> {
    // sum votes for each candidate
    let mut counts = HashMap::new();
    for v in dat {
        let cnt = counts.entry(v).or_insert(0);
        *cnt += 1;
    }

    // check tallies for majority holder
    let threshold: i64 = (dat.len() as i64) / 2;
    for (v, cnt) in counts {
        if cnt > threshold {
            return Some(*v);
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
 * Average case performance of O(n log n) with worst case O(n^2).  Using random
 * pivot selection makes worst case extremely unlikely.
 *
 * This is a pure implementation that does not switch to a non-recursive sort on
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
 * Make array an implicit binary max-heap data structure.
 *
 * The max-heap enforces that each node is GTE to its children and that the tree
 * is complete.
 *
 * The implicit part comes from packing the tree into an array with the
 * following index scheme:
 *
 *     root is at 0
 *     parent(i)     = floor( (i-1)/2 )
 *     leftchild(i)  = 2*i+1
 *     rightchild(i) = 2*i+2
 *
 * The construction process has linear time complexity, but that is not at all
 * obvious from the code.  A good explanation is here:
 *
 *    http://stackoverflow.com/questions/9755721/
 */
pub fn make_implicit_max_heap(dat: &mut [i32]) {
    // nothing to do
    if dat.len() < 2 {
        return;
    }

    fn parent_idx(node_idx: usize) -> usize {
        return (node_idx-1)/2;
    }

    // start by establishing heap property on far-right subtree
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
 *
 * This has O(n log n) time complexity and constant space complexity.  Note that
 * heap construction is called once O(n) and sift_down is invoked n times with
 * O(log n) complexity.  Hence, O(n + n log n) = O(n log n).
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
 * Merge sequential sorted runs.
 *
 * Takes two sorted sequences (runs) and combines them to a single sorted
 * sequence.
 *
 * Uses a scratch array to hold the intermediate output.  Scratch size consumed
 * equals the sum of the two run lengths.  The sequential constraint makes both
 * the interface and implementation a bit simpler, but isn't strictly speaking
 * necessary.
 */
fn merge_sequential_runs(dat: &mut [i32], lmin: usize, mid: usize, rmax: usize, scratch: &mut Vec<i32>) {

    scratch.clear();
    let mut li: usize = lmin;
    let mut ri: usize = mid;

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
 * Merge sort.
 *
 * This is a top-down implementation.  It has O(n log n) time complexity and
 * O(n) space complexity.
 *
 * Kept it as a pure implementation; does not switch to a non-recursive sort at
 * small partition sizes.
 */
pub fn merge_sort(dat: &mut [i32]) {
    // requires O(n) scratch space
    let mut scratch: Vec<i32> = Vec::with_capacity(dat.len());
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
    merge_sequential_runs(dat, min, mid, max, scratch);
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
 * alternate the direction of bubbling in each pass.  It handles the case where
 * small elements are near the end of the input and otherwise require up to n-1
 * passes to get to the beginning of the array.
 */
pub fn shaker_sort(dat: &mut [i32]) {

    // sorted bounds [0,min) and [max,len)
    let mut min: i64 = 0;
    let mut max: i64 = dat.len() as i64;

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
 * Naive method for selecting k-th smallest element.
 *
 * Just sorts and grabs, so it inherits its time and space complexity from the
 * underlying sort.
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
 *
 * Average case time complexity is O(n) but like quicksort it has the worst case
 * of O(n^2).
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
        let mid = min + (max-min)/2;

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
 * This algorithm is O(sqrt(n)) vs the O(log n) of binary search.  It can be
 * applicable when binary search is not possible or if performance is dominated
 * by seek rather than read time.
 *
 * Jump Searching: A Fast Sequential Search Technique, CACM, 21(10):831-834,
 * October 1978.
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
 * incomplete - hyper-trashy stub of interpolation search.
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

/*
 * Naive O(n^2) algorithm for finding largest subsequence.
 *
 * Enumerates all possible subsequences and returns longest one.
 */
pub fn largest_subseq_naive(dat: &[i32]) -> (usize, usize) {

    if dat.len() == 0 {
        return (0,0);
    }

    let mut lidx = 0;
    let mut ridx = 1;
    let mut maxsum = dat[0];

    for i in 0..dat.len() {
        let mut cursum = 0;
        for j in i..dat.len() {
            cursum += dat[j];

            if cursum > maxsum || (cursum == maxsum && (j-i+1) > (ridx-lidx)) {
                lidx = i;
                ridx = j+1;
                maxsum = cursum;
            }
        }
    }

    (lidx, ridx)
}


#[cfg(test)]
mod tests {
    use super::*;
	use std::collections::HashSet;

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
    fn test_fisher_yates_shuffle_two() {
        // 1 in 2^1000 chance of getting same shuffle 1000 times
        let mut dat = vec![1,2];
        let mut first_ones = 0;
        for _ in 0..1000 {
            fisher_yates_shuffle(&mut dat);
            if dat[0] == 1 {
                first_ones += 1;
            }
        }
        assert!(first_ones != 1000, "dubious shuffling of two element deck")
    }

    #[test]
    fn test_fisher_yates_shuffle_degenerate() {
        let mut v1: Vec<i32> = vec![];
        fisher_yates_shuffle(&mut v1);
        assert!(v1 == vec![], "empty vector unchanged by shuffle");

        let mut v2: Vec<i32> = vec![6];
        fisher_yates_shuffle(&mut v2);
        assert!(v2 == vec![6], "one-element vector unchanged by shuffle");

        let mut v3: Vec<i32> = vec![10,15];
        fisher_yates_shuffle(&mut v3);
        assert!(v3.len() == 2, "two-element vector same size post-shuffle");
    }

    #[test]
    fn test_heaps_permutations() {
		for n in 1..6 {
			// n factorial
			let expected_cnt: usize = (1..n+1).fold(1, |acc, val| acc * val);

			// gathers generated permutations in hashset
			let mut results = HashSet::new();
			results.insert("potato");

			{
				let gathercb = |x| {
					results.insert(x);
				};
	
				let mut input: Vec<i32> = (0..n as i32).collect(); 
				//heaps_permutations(&mut input, &mut gathercb);
			}

			let errmsg = format!("distinct permutation count !={} for n={}", expected_cnt, n);
			assert!(results.len() == expected_cnt, errmsg);
		}

		// test degenerate case
    }

    // runs arbitrary majority vote function through test battery
    fn majority_eval<F>(majorityfn: F)
        where F: Fn(&[i32]) -> Option<i32> {
        // majority of 1s
        let v1 = vec![0,1,0,1,1];
        let m1 = majorityfn(&v1);
        assert!(m1 == Some(1));

        // no majority
        let v2 = vec![0,1,0,1,1,0];
        let m2 = majorityfn(&v2);
        assert!(m2 == None);

        // majority of 1s but not strict majority
        let v3 = vec![2,2,0,1,0,1,1];
        let m3 = majorityfn(&v3);
        assert!(m3 == None);

        // empty input
        let v4 = vec![];
        let m4 = majorityfn(&v4);
        assert!(m4 == None);

        // lonely majority
        let v5 = vec![6];
        let m5 = majorityfn(&v5);
        assert!(m5 == Some(6));
    }

    #[test]
    fn test_bm_majority_vote() {
        majority_eval(bm_majority_vote);
    }

    #[test]
    fn test_hash_majority_vote() {
        majority_eval(hash_majority_vote);
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

    // runs arbitrary sort function through test battery
    fn sort_eval<F>(randsize: i32, sortfn: F)
        where F: Fn(&mut [i32]) -> () {

        // decently sized random vector
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
    fn test_quick_sort() {
        sort_eval(5000, quick_sort);
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

    // runs arbitrary selection function through test battery
    fn select_eval<F>(selectfn: F)
        where F: Fn(&mut [i32], usize) -> i32 {

        // ensure correct selection of each element
        let mut dat: Vec<i32> = (0..100).collect();
        fisher_yates_shuffle(&mut dat);

        for i in 0..dat.len() {
            let kval = selectfn(&mut dat, i);
            assert!(kval == (i as i32), "select did not pick correct element");
        }

        // test single element degenerate case
        let mut v1: Vec<i32> = vec![42];
        assert!(selectfn(&mut v1, 0) == 42);
    }

    #[test]
    fn test_naive_select() {
        select_eval(naive_select);
    }

    #[test]
    fn test_quick_select() {
        select_eval(quick_select);
    }

    fn search_eval<F>(searchfn: F)
        where F: Fn(&[i32], i32) -> Option<usize> {

        // search over arrays of different sizes
        for n in 1..101 as usize {
            // make sure we can hit every value inside
            let dat: Vec<i32> = (0..n as i32).collect();
            for i in 0..n {
                let res = searchfn(&dat, dat[i]);
                assert!(res == Some(i), "jump search should have hit");
            }

            // miss on purpose
            let r1 = searchfn(&dat, -1);
            assert!(r1 == None, "jump search should have missed");

            let r2 = searchfn(&dat, 1000);
            assert!(r2 == None, "jump search should have missed");
        }

        // degenerate case
        let degen: Vec<i32> = vec![];
        let dres = searchfn(&degen, 100);
        assert!(dres == None);
    }

    #[test]
    fn test_binary_search() {
        search_eval(binary_search);
    }

    #[test]
    fn test_jump_search() {
        search_eval(jump_search);
    }

    #[test]
    fn test_interpolation_search() {
        search_eval(interpolation_search);
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
        let ins = vec![vec![],
                       vec![1,2,3,4,5],
                       vec![1,1,1,2,2,2],
                       vec![1,2,3,4,2],
                       vec![1,1,1,1,1]];

        let truths = vec![vec![],
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

    // runs arbitrary largest subsequence function through test battery
    fn largest_subseq_eval<F>(subseqfn: F)
        where F: Fn(&[i32]) -> (usize, usize) {

        let (a, b) = subseqfn(&vec![]);
        assert!(a == 0 && b == 0, "failed degenerate case");

        let (a, b) = subseqfn(&vec![42]);
        assert!(a == 0 && b == 1, "failed single-element case");

        let test = vec![10, -5, 6];
        let (a, b) = subseqfn(&test);
        assert!(a == 0 && b == 3, "failed crossing valley");

        let test = vec![3, -100, 5];
        let (a, b) = subseqfn(&test);
        assert!(a == 2 && b == 3, "failed crossing deep valley");

        let test = vec![0, 10, -10, 10, 0];
        let (a, b) = subseqfn(&test);
        assert!(a == 0 && b == 5, "failed crossing symmetric valley");

        let test = vec![0, 1, 10, -100, 0, 0, 1, 10];
        let (a, b) = subseqfn(&test);
        assert!(a == 4 && b == 8, "failed on long-tailed valley");

        let test = vec![-1, 10, -5, 6, -10, 100, -1];
        let (a, b) = subseqfn(&test);
        assert!(a == 1 && b == 6, "failed crossing double valley");

        let test = vec![10, 0, 0, 1, 0];
        let (a, b) = subseqfn(&test);
        assert!(a == 0 && b == 5, "failed including trailing zeros");
    }

    #[test]
    fn test_largest_subseq_naive() {
        largest_subseq_eval(largest_subseq_naive);
    }
}

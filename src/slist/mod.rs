/*
 * Singly Linked List Implementation.
 *
 * Design choices:
 *
 *   - [TODO] Wrapper struct for list rather than using head node directly.
 *            Gives us a place to store list length and changes that operation
 *            from O(n) to O(1).
 *
 *   - [TODO] Templatized list/node type to embed contained value directly in
 *            list nodes.  Avoids unneeded pointer redirection.
 *
 */

pub struct Slist {
	head:	Option<Box<Node>>,
	len:	usize,
}

impl Slist {
	pub fn head(&self) -> &Option<Box<Node>> {
		&self.head
	}

	pub fn len(&self) -> usize {
		self.len
	}

	pub fn reverse(&mut self) {
		// let mut node = list.head();
		// loop {
		// 	match *node {
		// 		None	 	 => break,
		// 		Some(ref nb) => { res.push(nb.val); node = &nb.next; },
		// 	}
		// }
	}
}

pub struct Node {
	val:	i32,
	next:	Option<Box<Node>>,
}

impl Node {
	pub fn get_val(&self) -> i32 {
		self.val
	}
}

/*
 * Convert vector to linked list representation.
 */
pub fn vec_to_list(dat: &Vec<i32>) -> Slist {
	if dat.len() == 0 {
		return Slist{head: None, len: 0};
	}

	let mut prev_node: Option<Box<Node>> = None;
	for i in 0..dat.len() {
		let n = Node{val: dat[dat.len()-i-1], next: prev_node};
		prev_node = Some(Box::new(n));
	}

	return Slist{head: prev_node, len: dat.len()}
}

/*
 * Convert linked list to vector representation.
 */
pub fn list_to_vec(list: &Slist) -> Vec<i32> {
	let mut res: Vec<i32> = Vec::with_capacity(list.len());
	let mut node = list.head();
	loop {
		match *node {
			None	 	 => break,
			Some(ref nb) => { res.push(nb.val); node = &nb.next; },
		}
	}
	res
}

/*
 * Check if element in list.
 */
pub fn list_contains(list: &Slist, val: i32) -> bool {
	let mut found = false;
	let mut cur = list.head();
	while !found {
		match *cur {
			None	     => break,
			Some(ref nb) => { found = nb.val == val;
							  cur = &nb.next; },
		}
	}
	found
}


#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_slist_vec_list_convert() {
		let v1: Vec<i32> = vec![5,3,1,0,4,2];
		let l = vec_to_list(&v1);
		let v2 = list_to_vec(&l);
		assert!(v1 == v2);
	}

	#[test]
	fn test_slist_contains() {
		let v: Vec<i32> = vec![3,2,1,0,4,5,6,7];
		let list = vec_to_list(&v);

		assert!(list.len() == v.len());
		assert!(list_contains(&list, 7));
		assert!(list_contains(&list, 2));
		assert!(!list_contains(&list, 8));
	}

	#[test]
	fn test_slist_len() {
		// empty list
		let l0 = Slist{head: None, len: 0};
		assert!(l0.len() == 0);

		// one element list
		let l1 = vec_to_list(&vec![1]);
		assert!(l1.len() == 1);

		// two element list
		let l2 = vec_to_list(&vec![1,2]);
		assert!(l2.len() == 2);
	}

	#[test]
	fn test_slist_reverse() {
		let v_in: Vec<i32> = (0..10).collect();
		let mut l = vec_to_list(&v_in);

		l.reverse();
		let v_out = list_to_vec(&l);

		let v_truth: Vec<i32> = (0..10).rev().collect();
		assert!(v_out == v_truth);
	}
}

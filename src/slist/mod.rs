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
	pub fn len(&self) -> usize {
		self.len
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
pub fn vec_to_list(dat: &Vec<i32>) -> Option<Box<Node>> {
	if dat.len() == 0 {
		return None;
	}

	let mut prev_node: Option<Box<Node>> = None;
	for i in 0..dat.len() {
		let n = Node{val: dat[dat.len()-i-1], next: prev_node};
		prev_node = Some(Box::new(n));
	}

	return prev_node;
}

/*
 * Get length of list.
 */
pub fn list_len(n: &Option<Box<Node>>) -> usize {
	let mut len = 0;
	let mut cur = n;
	loop {
		match *cur {
			None	     => break,
			Some(ref nb) => cur = &nb.next,
		}
		len += 1;
	}
	len
}

/*
 * Check if element in list.
 */
pub fn list_contains(n: &Option<Box<Node>>, val: i32) -> bool {
	let mut found = false;
	let mut cur = n;
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
	fn test_vec_to_list() {
		let v: Vec<i32> = vec![3,2,1,0,4,5,6,7];
		let y = vec_to_list(&v);

		assert!(list_len(&y) == v.len());
		assert!(list_contains(&y, 7));
		assert!(list_contains(&y, 2));
		assert!(!list_contains(&y, 8));
	}
}

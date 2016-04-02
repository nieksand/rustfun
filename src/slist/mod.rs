/*
 * Singly Linked List Implementation.
 */

pub struct Node {
	val:	i32,
	next:	Option<Box<Node>>,
}

impl Node {
	pub fn get_val(&self) -> i32 {
		self.val
	}
}


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

//
//pub fn list_len(n: &Node) -> usize {
//	
//
//
//}

//pub fn list_to_vec(head: &Option<Box<Node>>) -> Vec<i32> {
//
//	let mut v: Vec<i32> = Vec::new();
//
//	let mut nptr = head;
//	loop {
//		match *nptr {
//			None 	=> break,
//			Some(n) => {
//				v.push(n.val);
//				nptr = &n.next;
//			}	
//		}
//	}
//	return v;
//}


#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_vec_to_list() {
		let v: Vec<i32> = vec![3,2,1,0,4,5,6,7];
		let y = vec_to_list(&v);
		assert!(list_len(&y) == v.len());
	}
}

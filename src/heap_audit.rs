//! Opt-in measurement of Rust heap operations inside the native callback body.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<Option<[usize; 3]>> = const { Cell::new(None) };
    static TOTALS: Cell<[[usize; 4]; 2]> = const { Cell::new([[0; 4]; 2]) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub(crate) struct Scope;

impl Scope {
    pub(crate) fn new() -> Self {
        ACTIVE.set(Some([0; 3]));
        Self
    }

    pub(crate) fn finish(self, successful: bool) {
        let counts = ACTIVE.replace(None).expect("active callback audit");
        TOTALS.with(|totals| {
            let mut value = totals.get();
            let row = &mut value[usize::from(!successful)];
            row[0] += 1;
            for (total, count) in row[1..].iter_mut().zip(counts) {
                *total += count;
            }
            totals.set(value);
        });
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        // Do not leave counting enabled while unwinding a panicking callback.
        ACTIVE.set(None);
    }
}

/// Return and clear successful/failed callback rows: calls, allocs, reallocs, frees.
#[pyo3::pyfunction]
pub(crate) fn take_callback_audit() -> [[usize; 4]; 2] {
    TOTALS.replace([[0; 4]; 2])
}

fn count(kind: usize) {
    ACTIVE.with(|active| {
        if let Some(mut counts) = active.get() {
            counts[kind] += 1;
            active.set(Some(counts));
        }
    });
}

struct CountingAllocator;

// SAFETY: Every operation forwards its original arguments to the System allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(0);
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(0);
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(1);
        // SAFETY: The caller supplies a System-owned pointer and its layout.
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(2);
        // SAFETY: The caller supplies a System-owned pointer and its layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::{Scope, take_callback_audit};
    use std::hint::black_box;

    #[test]
    fn audit_counts_all_heap_operations_and_separates_failures() {
        take_callback_audit();
        let audit = Scope::new();
        let mut bytes = Vec::with_capacity(1);
        bytes.push(1u8);
        bytes.resize(128, 2);
        drop(black_box(bytes));
        audit.finish(true);
        let audit = Scope::new();
        drop(black_box(Box::new(42)));
        audit.finish(false);
        let [successful, failed] = take_callback_audit();
        assert_eq!(successful[0], 1);
        assert!(successful[1..].iter().all(|c| *c > 0));
        assert_eq!(failed[0], 1);
        assert!(failed[1] > 0 && failed[3] > 0);
        assert_eq!(take_callback_audit(), [[0; 4]; 2]);
    }
}

//! Separate Rust heap measurements for native call and detached DSP boundaries.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<[Option<[usize; 3]>; 2]> = const { Cell::new([None; 2]) };
    static TOTALS: Cell<[[[usize; 4]; 2]; 2]> = const { Cell::new([[[0; 4]; 2]; 2]) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub(crate) struct Scope {
    boundary: usize,
}

impl Scope {
    pub(crate) fn new() -> Self {
        Self::start(0)
    }

    pub(crate) fn native_call() -> Self {
        Self::start(1)
    }

    fn start(boundary: usize) -> Self {
        let mut active = ACTIVE.get();
        active[boundary] = Some([0; 3]);
        ACTIVE.set(active);
        Self { boundary }
    }

    pub(crate) fn finish(self, successful: bool) {
        let mut active = ACTIVE.get();
        let counts = active[self.boundary].take().expect("active callback audit");
        ACTIVE.set(active);
        TOTALS.with(|totals| {
            let mut value = totals.get();
            let row = &mut value[self.boundary][usize::from(!successful)];
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
        let mut active = ACTIVE.get();
        active[self.boundary] = None;
        ACTIVE.set(active);
    }
}

/// Return and clear successful/failed callback rows: calls, allocs, reallocs, frees.
#[pyo3::pyfunction]
pub(crate) fn take_callback_audit() -> [[usize; 4]; 2] {
    take(0)
}

/// Whole native calls include array conversion, copying and Python error construction.
#[pyo3::pyfunction]
pub(crate) fn take_native_call_audit() -> [[usize; 4]; 2] {
    take(1)
}

fn take(boundary: usize) -> [[usize; 4]; 2] {
    let mut totals = TOTALS.get();
    let result = std::mem::take(&mut totals[boundary]);
    TOTALS.set(totals);
    result
}

fn count(kind: usize) {
    ACTIVE.with(|active| {
        let mut counts = active.get();
        for count in counts.iter_mut().flatten() {
            count[kind] += 1;
        }
        active.set(counts);
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
    use super::{Scope, take_callback_audit, take_native_call_audit};
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

    #[test]
    fn native_call_audit_includes_work_outside_nested_dsp_scope() {
        take_callback_audit();
        take_native_call_audit();
        let outer = Scope::native_call();
        drop(black_box(Box::new(1)));
        let inner = Scope::new();
        drop(black_box(Box::new(2)));
        inner.finish(true);
        drop(black_box(Box::new(3)));
        outer.finish(true);
        let [dsp, failed] = take_callback_audit();
        let [call, call_failed] = take_native_call_audit();
        assert_eq!(dsp, [1, 1, 0, 1]);
        assert_eq!(call, [1, 3, 0, 3]);
        assert_eq!(failed, [0; 4]);
        assert_eq!(call_failed, [0; 4]);
    }
}

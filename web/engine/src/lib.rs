//! The real game for the browser demo, as a handful of C-style exports (no bindings generator).
//!
//! It is sabbia itself, at the same revision the training data came from, plus the generator's
//! random starting scenes. The page reads the class grid straight out of WebAssembly memory: the
//! same ids the dreamer sees and predicts. The exports are prefixed because sabbia, built for
//! WebAssembly, already exports its own `init`, `step`, `paint` and `clear`.

use sabbia::world::Rng;
use sabbia::{Element, World};
use std::cell::RefCell;

thread_local! {
    static WORLD: RefCell<World> = RefCell::new(World::new(1, 1, 1));
    static CLASSES: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn with_world<T>(f: impl FnOnce(&mut World) -> T) -> T {
    WORLD.with(|world| f(&mut world.borrow_mut()))
}

/// Creates a `width` x `height` world holding a random starting scene, as in the training data.
#[unsafe(no_mangle)]
pub extern "C" fn engine_init(width: u32, height: u32, seed: u32) {
    let (width, height) = (
        width.clamp(8, 1024) as usize,
        height.clamp(8, 1024) as usize,
    );
    let mut rng = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    let mut world = World::new(width, height, rng.next_u32());
    sogno_gen::scene(&mut world, &mut rng);
    with_world(|w| *w = world);
}

/// Advances `steps` simulation steps.
#[unsafe(no_mangle)]
pub extern "C" fn engine_step(steps: u32) {
    with_world(|world| {
        for _ in 0..steps {
            world.step();
        }
    });
}

/// Paints a disc of `element` with sabbia's brush. Unknown numbers are ignored.
#[unsafe(no_mangle)]
pub extern "C" fn engine_paint(x: i32, y: i32, radius: i32, element: u32) {
    if let Some(element) = u8::try_from(element).ok().and_then(Element::from_u8) {
        with_world(|world| world.paint(x, y, radius, element));
    }
}

/// Empties the world.
#[unsafe(no_mangle)]
pub extern "C" fn engine_clear() {
    with_world(World::clear);
}

/// Pointer to `width * height` class ids, row by row: the element ids, with fire, smoke, steam and
/// embers split by age (`sogno_gen::classes`). Valid until the next call into the module.
#[unsafe(no_mangle)]
pub extern "C" fn engine_classes() -> *const u8 {
    CLASSES.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        with_world(|world| sogno_gen::class_grid(world, &mut buffer));
        buffer.as_ptr()
    })
}

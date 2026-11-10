//! The scripted painter behind sogno's training data, shared by the generator (`sogno-gen`) and the
//! browser demo (`web/engine`): random starting scenes and brush strokes that play like a person.

use sabbia::world::Rng;
use sabbia::{Element, World};

/// What the painter can draw, and how often it picks each (out of the sum of the weights).
const PALETTE: [(Element, u32); 15] = [
    (Element::Sand, 30),
    (Element::Water, 30),
    (Element::Fire, 18),
    (Element::Wall, 14),
    (Element::Wood, 10),
    (Element::Oil, 10),
    (Element::Lava, 10),
    (Element::Acid, 9),
    (Element::Stone, 9),
    (Element::Empty, 8),
    (Element::Ice, 7),
    (Element::Plant, 7),
    (Element::Smoke, 3),
    (Element::Steam, 3),
    (Element::Glass, 3),
];

pub fn pick_element(rng: &mut Rng) -> Element {
    let total: u32 = PALETTE.iter().map(|&(_, w)| w).sum();
    let mut roll = rng.next_u32() % total;
    for &(element, weight) in &PALETTE {
        if roll < weight {
            return element;
        }
        roll -= weight;
    }
    Element::Sand
}

/// A uniform integer in `low..=high`.
pub fn between(rng: &mut Rng, low: i32, high: i32) -> i32 {
    low + (rng.next_u32() % (high - low + 1) as u32) as i32
}

/// Draws a thick line of `element`, cell by cell, the way a dragged brush does.
pub fn line(world: &mut World, from: (i32, i32), to: (i32, i32), radius: i32, element: Element) {
    let steps = (to.0 - from.0).abs().max((to.1 - from.1).abs()).max(1);
    for i in 0..=steps {
        let x = from.0 + (to.0 - from.0) * i / steps;
        let y = from.1 + (to.1 - from.1) * i / steps;
        world.paint(x, y, radius, element);
    }
}

/// A random starting scene: a few platforms and slopes, a few piles and pools.
pub fn scene(world: &mut World, rng: &mut Rng) {
    let (w, h) = (world.width() as i32, world.height() as i32);
    for _ in 0..between(rng, 0, 4) {
        let solid = [
            Element::Wall,
            Element::Wall,
            Element::Wood,
            Element::Glass,
            Element::Ice,
        ][rng.next_u32() as usize % 5];
        let x = between(rng, 0, w - 1);
        let y = between(rng, h / 4, h - 1);
        let length = between(rng, w / 8, w / 2);
        let slope = between(rng, -length / 2, length / 2);
        line(
            world,
            (x, y),
            (x + length, y + slope),
            between(rng, 0, 1),
            solid,
        );
    }
    for _ in 0..between(rng, 2, 9) {
        let element = pick_element(rng);
        let (x, y) = (between(rng, 0, w - 1), between(rng, 0, h - 1));
        world.paint(x, y, between(rng, 3, 12), element);
    }
    // Let the scene settle a little, so episodes do not all start with everything in mid-air.
    for _ in 0..between(rng, 0, 60) {
        world.step();
    }
}

/// One brush stroke in progress.
pub struct Stroke {
    pub element: Element,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub radius: i32,
    pub frames_left: u32,
}

impl Stroke {
    pub fn new(rng: &mut Rng, w: i32, h: i32) -> Stroke {
        let element = pick_element(rng);
        // People pour falling things from the top and build solids anywhere.
        let top = matches!(
            element,
            Element::Sand | Element::Water | Element::Stone | Element::Oil
        ) && rng.chance(600);
        let y = if top {
            between(rng, 0, h / 4)
        } else {
            between(rng, 0, h - 1)
        };
        let speed = between(rng, 0, 20) as f32 / 10.0;
        let angle = (rng.next_u32() % 6283) as f32 / 1000.0;
        Stroke {
            element,
            x: between(rng, 0, w - 1) as f32,
            y: y as f32,
            vx: speed * angle.cos(),
            vy: speed * angle.sin() * if top { 0.2 } else { 1.0 },
            radius: between(rng, 0, 6),
            frames_left: between(rng, 1, 40) as u32,
        }
    }
}

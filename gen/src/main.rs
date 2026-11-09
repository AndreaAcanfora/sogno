//! Plays sabbia with a scripted painter and records what happens, as training data for sogno.
//!
//! Each episode is a random scene that a painter keeps drawing into, one stroke at a time, the way a
//! person plays: pour sand from the top, draw a wall, set the wood on fire. The output is three files:
//!
//! - `states.npy`, `u8 [episodes, frames + 1, height, width]`: element ids. `states[e, 0]` is the
//!   starting scene; `states[e, t + 1]` is the world `stride` steps after the painter acted at frame t.
//! - `paints.npy`, `i32 [n, 4]`: the cells the painter changed, as `(episode, frame, cell, element)`
//!   rows sorted by episode and frame. The model's input at frame t is `states[e, t]` with these applied.
//! - `meta.json`: sizes, stride and seed.
//!
//! So every frame gives one training pair, painted or not: what the world looked like after the
//! painter acted, and what it looked like `stride` steps later.

use sabbia::world::Rng;
use sabbia::{Element, World};
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

struct Options {
    episodes: usize,
    frames: usize,
    width: usize,
    height: usize,
    stride: usize,
    seed: u32,
    out: PathBuf,
}

const USAGE: &str = "usage: sogno-gen --out DIR [--episodes 200] [--frames 300] [--width 96] \
[--height 64] [--stride 2] [--seed 1]";

fn parse_options() -> Result<Options, String> {
    let mut options = Options {
        episodes: 200,
        frames: 300,
        width: 96,
        height: 64,
        stride: 2,
        seed: 1,
        out: PathBuf::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.into());
        }
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("{flag}: not a number: {value}"))
        };
        match flag.as_str() {
            "--episodes" => options.episodes = number()?,
            "--frames" => options.frames = number()?,
            "--width" => options.width = number()?,
            "--height" => options.height = number()?,
            "--stride" => options.stride = number()?,
            "--seed" => options.seed = number()? as u32,
            "--out" => options.out = PathBuf::from(&value),
            _ => return Err(format!("unknown option {flag}\n{USAGE}")),
        }
    }
    if options.out.as_os_str().is_empty() {
        return Err(USAGE.into());
    }
    if options.episodes == 0 || options.frames == 0 || options.stride == 0 {
        return Err("--episodes, --frames and --stride must be at least 1".into());
    }
    if !(8..=1024).contains(&options.width) || !(8..=1024).contains(&options.height) {
        return Err("--width and --height must be between 8 and 1024".into());
    }
    Ok(options)
}

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

fn pick_element(rng: &mut Rng) -> Element {
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
fn between(rng: &mut Rng, low: i32, high: i32) -> i32 {
    low + (rng.next_u32() % (high - low + 1) as u32) as i32
}

/// Draws a thick line of `element`, cell by cell, the way a dragged brush does.
fn line(world: &mut World, from: (i32, i32), to: (i32, i32), radius: i32, element: Element) {
    let steps = (to.0 - from.0).abs().max((to.1 - from.1).abs()).max(1);
    for i in 0..=steps {
        let x = from.0 + (to.0 - from.0) * i / steps;
        let y = from.1 + (to.1 - from.1) * i / steps;
        world.paint(x, y, radius, element);
    }
}

/// A random starting scene: a few platforms and slopes, a few piles and pools.
fn scene(world: &mut World, rng: &mut Rng) {
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
struct Stroke {
    element: Element,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    radius: i32,
    frames_left: u32,
}

impl Stroke {
    fn new(rng: &mut Rng, w: i32, h: i32) -> Stroke {
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

/// Writes a little-endian `.npy` file (format version 1.0).
fn write_npy(path: &Path, dtype: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let mut header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
    // Magic (6) + version (2) + length (2) + header + newline must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut file = BufWriter::new(fs::File::create(path)?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    file.write_all(data)?;
    file.flush()
}

fn run(options: &Options) -> io::Result<()> {
    let Options {
        episodes,
        frames,
        width,
        height,
        stride,
        seed,
        ..
    } = *options;
    let cells = width * height;
    let mut states = vec![0u8; episodes * (frames + 1) * cells];
    let mut paints: Vec<[i32; 4]> = Vec::new();
    let mut rng = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    let mut before = vec![Element::Empty; cells];

    for episode in 0..episodes {
        let mut world = World::new(width, height, rng.next_u32());
        scene(&mut world, &mut rng);
        let mut stroke: Option<Stroke> = None;
        let base = episode * (frames + 1) * cells;
        let mut record = |world: &World, frame: usize| {
            let at = base + frame * cells;
            for y in 0..height {
                for x in 0..width {
                    states[at + y * width + x] = world.get(x, y) as u8;
                }
            }
        };
        record(&world, 0);
        for frame in 0..frames {
            if stroke.is_none() && rng.chance(120) {
                stroke = Some(Stroke::new(&mut rng, width as i32, height as i32));
            }
            if let Some(s) = &mut stroke {
                for (i, cell) in before.iter_mut().enumerate() {
                    *cell = world.get(i % width, i / width);
                }
                world.paint(s.x.round() as i32, s.y.round() as i32, s.radius, s.element);
                for (i, &old) in before.iter().enumerate() {
                    let new = world.get(i % width, i / width);
                    if new != old {
                        paints.push([episode as i32, frame as i32, i as i32, new as i32]);
                    }
                }
                s.x = (s.x + s.vx).clamp(0.0, width as f32 - 1.0);
                s.y = (s.y + s.vy).clamp(0.0, height as f32 - 1.0);
                if rng.chance(100) {
                    s.vx += between(&mut rng, -5, 5) as f32 / 10.0;
                    s.vy += between(&mut rng, -5, 5) as f32 / 10.0;
                }
                s.frames_left -= 1;
                if s.frames_left == 0 {
                    stroke = None;
                }
            }
            for _ in 0..stride {
                world.step();
            }
            record(&world, frame + 1);
        }
    }

    fs::create_dir_all(&options.out)?;
    write_npy(
        &options.out.join("states.npy"),
        "|u1",
        &[episodes, frames + 1, height, width],
        &states,
    )?;
    let flat: Vec<u8> = paints
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_npy(
        &options.out.join("paints.npy"),
        "<i4",
        &[paints.len(), 4],
        &flat,
    )?;
    let meta = format!(
        "{{\n  \"game\": \"sabbia\",\n  \"classes\": {},\n  \"episodes\": {episodes},\n  \"frames\": {frames},\n  \
\"width\": {width},\n  \"height\": {height},\n  \"stride\": {stride},\n  \"seed\": {seed},\n  \"paints\": {}\n}}\n",
        Element::ALL.len(),
        paints.len()
    );
    fs::write(options.out.join("meta.json"), meta)
}

fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&options) {
        Ok(()) => {
            println!(
                "wrote {} episodes x {} frames of {}x{} to {}",
                options.episodes,
                options.frames,
                options.width,
                options.height,
                options.out.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sogno-gen: {error}");
            ExitCode::FAILURE
        }
    }
}

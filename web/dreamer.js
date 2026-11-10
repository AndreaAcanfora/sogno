// The dreamer in the browser: the same U-Net as sogno/model.py, run on the GPU with WebGPU.
//
// No ML library: the network is five small compute shaders (an embedding stem, a convolution, group
// norm + SiLU in two halves, and a channel concat), recorded once and replayed for every model call.
// Activations are float32, channel-last ([H, W, C]) like MLX. Sampling (MaskGIT's confident-cells-
// first reveal) runs in JavaScript on the logits, exactly as `sample` does in Python.

const WORKGROUP = 64;

const SHADERS = {
  // Stem: a 3x3 conv over one-hot inputs is a sum of weight columns, picked by each neighbour's class.
  stem: /* wgsl */ `
    struct P { h: u32, w: u32, cout: u32, classes: u32, history: u32, cin: u32 }
    @group(0) @binding(0) var<uniform> p: P;
    @group(0) @binding(1) var<storage, read> ids: array<u32>;
    @group(0) @binding(2) var<storage, read> weight: array<f32>;
    @group(0) @binding(3) var<storage, read> bias: array<f32>;
    @group(0) @binding(4) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(${WORKGROUP})
    fn main(@builtin(global_invocation_id) g: vec3u) {
      let i = g.x;
      let cells = p.h * p.w;
      if (i >= cells * p.cout) { return; }
      let o = i % p.cout;
      let cell = i / p.cout;
      let y = i32(cell / p.w);
      let x = i32(cell % p.w);
      var acc = bias[o];
      for (var ky = 0; ky < 3; ky++) {
        let iy = y + ky - 1;
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0; kx < 3; kx++) {
          let ix = x + kx - 1;
          if (ix < 0 || ix >= i32(p.w)) { continue; }
          let at = u32(iy) * p.w + u32(ix);
          let base = ((o * 3u + u32(ky)) * 3u + u32(kx)) * p.cin;
          for (var j = 0u; j <= p.history; j++) {
            acc += weight[base + j * p.classes + ids[j * cells + at]];
          }
        }
      }
      out[i] = acc;
    }`,

  // Convolution, channel-last, with optional stride, nearest 2x upsampling of the input, and residual.
  // Each thread computes four output channels, so every input value read is used four times.
  conv: /* wgsl */ `
    struct P { ih: u32, iw: u32, cin: u32, oh: u32, ow: u32, cout: u32, k: u32, stride: u32,
               pad: u32, up: u32, residual: u32, _pad: u32 }
    @group(0) @binding(0) var<uniform> p: P;
    @group(0) @binding(1) var<storage, read> inp: array<vec4f>;
    @group(0) @binding(2) var<storage, read> weight: array<vec4f>;
    @group(0) @binding(3) var<storage, read> bias: array<vec4f>;
    @group(0) @binding(4) var<storage, read> res: array<vec4f>;
    @group(0) @binding(5) var<storage, read_write> out: array<vec4f>;
    @compute @workgroup_size(${WORKGROUP})
    fn main(@builtin(global_invocation_id) g: vec3u) {
      let quads = p.cout / 4u;
      let i = g.x;
      if (i >= p.oh * p.ow * quads) { return; }
      let q = i % quads;
      let cell = i / quads;
      let y = i32(cell / p.ow);
      let x = i32(cell % p.ow);
      let eh = i32(p.ih << p.up);
      let ew = i32(p.iw << p.up);
      var acc = bias[q];
      let cin4 = p.cin / 4u;
      for (var ky = 0u; ky < p.k; ky++) {
        let iy = y * i32(p.stride) + i32(ky) - i32(p.pad);
        if (iy < 0 || iy >= eh) { continue; }
        for (var kx = 0u; kx < p.k; kx++) {
          let ix = x * i32(p.stride) + i32(kx) - i32(p.pad);
          if (ix < 0 || ix >= ew) { continue; }
          let src = ((u32(iy) >> p.up) * p.iw + (u32(ix) >> p.up)) * cin4;
          // weight is repacked as [cout/4, k, k, cin/4, 4 inputs] of vec4 (4 outputs each).
          var w = ((q * p.k + ky) * p.k + kx) * cin4 * 4u;
          for (var c = 0u; c < cin4; c++) {
            let v = inp[src + c];
            acc += weight[w] * v.x + weight[w + 1u] * v.y + weight[w + 2u] * v.z + weight[w + 3u] * v.w;
            w += 4u;
          }
        }
      }
      if (p.residual != 0u) { acc += res[i]; }
      out[i] = acc;
    }`,

  // Group norm, first half: mean and 1/std of each group, one workgroup per group.
  stats: /* wgsl */ `
    struct P { cells: u32, c: u32, per: u32, _pad: u32 }
    @group(0) @binding(0) var<uniform> p: P;
    @group(0) @binding(1) var<storage, read> inp: array<f32>;
    @group(0) @binding(2) var<storage, read_write> stats: array<f32>;
    var<workgroup> s1: array<f32, 256>;
    var<workgroup> s2: array<f32, 256>;
    @compute @workgroup_size(256)
    fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
      let group = wg.x;
      let n = p.cells * p.per;
      var a = 0.0;
      var b = 0.0;
      for (var j = l.x; j < n; j += 256u) {
        let v = inp[(j / p.per) * p.c + group * p.per + j % p.per];
        a += v;
        b += v * v;
      }
      s1[l.x] = a;
      s2[l.x] = b;
      workgroupBarrier();
      for (var s = 128u; s > 0u; s >>= 1u) {
        if (l.x < s) { s1[l.x] += s1[l.x + s]; s2[l.x] += s2[l.x + s]; }
        workgroupBarrier();
      }
      if (l.x == 0u) {
        let mean = s1[0] / f32(n);
        let variance = max(s2[0] / f32(n) - mean * mean, 0.0);
        stats[group * 2u] = mean;
        stats[group * 2u + 1u] = inverseSqrt(variance + 1e-5);
      }
    }`,

  // Group norm, second half, fused with SiLU (the network always applies them together).
  norm: /* wgsl */ `
    struct P { cells: u32, c: u32, per: u32, _pad: u32 }
    @group(0) @binding(0) var<uniform> p: P;
    @group(0) @binding(1) var<storage, read> inp: array<f32>;
    @group(0) @binding(2) var<storage, read> stats: array<f32>;
    @group(0) @binding(3) var<storage, read> gamma: array<f32>;
    @group(0) @binding(4) var<storage, read> beta: array<f32>;
    @group(0) @binding(5) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(${WORKGROUP})
    fn main(@builtin(global_invocation_id) g: vec3u) {
      let i = g.x;
      if (i >= p.cells * p.c) { return; }
      let ch = i % p.c;
      let group = ch / p.per;
      let v = (inp[i] - stats[group * 2u]) * stats[group * 2u + 1u] * gamma[ch] + beta[ch];
      out[i] = v / (1.0 + exp(-v));
    }`,

  concat: /* wgsl */ `
    struct P { cells: u32, ca: u32, cb: u32, _pad: u32 }
    @group(0) @binding(0) var<uniform> p: P;
    @group(0) @binding(1) var<storage, read> a: array<f32>;
    @group(0) @binding(2) var<storage, read> b: array<f32>;
    @group(0) @binding(3) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(${WORKGROUP})
    fn main(@builtin(global_invocation_id) g: vec3u) {
      let c = p.ca + p.cb;
      let i = g.x;
      if (i >= p.cells * c) { return; }
      let cell = i / c;
      let ch = i % c;
      out[i] = select(b[cell * p.cb + ch - p.ca], a[cell * p.ca + ch], ch < p.ca);
    }`,
};

function halfToFloat(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) out[i] = sign * frac * 2 ** -24;
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}

// [cout, k, k, cin] -> [cout/4, k, k, cin/4, 4 inputs, 4 outputs], the layout the conv shader reads.
function packConv(w, [cout, k, , cin]) {
  const out = new Float32Array(w.length);
  let n = 0;
  for (let q = 0; q < cout / 4; q++)
    for (let t = 0; t < k * k; t++)
      for (let c = 0; c < cin; c++)
        for (let r = 0; r < 4; r++) out[n++] = w[((q * 4 + r) * k * k + t) * cin + c];
  return out;
}

export class Dreamer {
  static async load(url, device) {
    const base = new URL(url, location.href);
    const meta = await (await fetch(new URL("model.json", base))).json();
    const bytes = await (await fetch(new URL("weights.bin", base))).arrayBuffer();
    return new Dreamer(device, meta, halfToFloat(new Uint16Array(bytes)));
  }

  constructor(device, meta, weights) {
    this.device = device;
    this.meta = meta;
    const cfg = (this.config = meta.config);
    const [h, w] = (this.frame = meta.frame);
    this.cells = h * w;
    this.tensor = (name) => {
      const t = meta.tensors[name];
      if (!t) throw new Error(`missing tensor ${name}`);
      const size = t.shape.reduce((a, b) => a * b, 1);
      return { shape: t.shape, data: weights.subarray(t.offset, t.offset + size) };
    };
    this.pipelines = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      const module = device.createShaderModule({ code });
      this.pipelines[name] = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    }
    this.steps = [];
    this.none = this.buffer(16); // stands in for the residual when there is none
    this.ids = this.buffer((cfg.history + 1) * this.cells * 4, GPUBufferUsage.COPY_DST);
    const logits = this.build(h, w);
    this.logitsBuffer = logits.buf;
    this.stride = logits.c; // classes, rounded up to a multiple of 4
    this.readback = device.createBuffer({
      size: this.cells * this.stride * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  buffer(size, extra = 0, data) {
    if (data) size = data.byteLength;
    const buf = this.device.createBuffer({
      size: Math.max(16, Math.ceil(size / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | extra,
      mappedAtCreation: Boolean(data),
    });
    if (data) {
      new data.constructor(buf.getMappedRange()).set(data);
      buf.unmap();
    }
    return buf;
  }

  uniform(values) {
    const data = new Uint32Array(Math.ceil(values.length / 4) * 4);
    data.set(values);
    const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }

  step(name, buffers, params, threads, workgroups) {
    const pipeline = this.pipelines[name];
    const entries = [this.uniform(params), ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } }));
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    this.steps.push({ pipeline, bindGroup, workgroups: workgroups ?? Math.ceil(threads / WORKGROUP) });
  }

  // ---------- the network, mirroring Dreamer.__call__ in sogno/model.py ----------

  build(h, w) {
    const cfg = this.config;
    const channels = Array.from({ length: cfg.levels }, (_, i) => cfg.width * 2 ** i);
    const stemW = this.tensor("stem.weight");
    let x = { buf: this.buffer(h * w * channels[0] * 4), h, w, c: channels[0] };
    this.step(
      "stem",
      [this.ids, this.buffer(0, 0, stemW.data), this.buffer(0, 0, this.tensor("stem.bias").data), x.buf],
      [h, w, channels[0], cfg.classes, cfg.history, stemW.shape[3]],
      h * w * channels[0],
    );
    const skips = [];
    channels.forEach((ch, i) => {
      for (let b = 0; b < cfg.blocks; b++) x = this.resblock(`down.${i}.blocks.${b}`, x, ch);
      if (i < channels.length - 1) {
        skips.push(x);
        x = this.conv(`down.${i}.pool`, x, { stride: 2 });
      }
    });
    for (let b = 0; b < 2; b++) x = this.resblock(`middle.${b}`, x, channels.at(-1));
    for (let u = 0; u < channels.length - 1; u++) {
      const i = channels.length - 2 - u;
      x = this.conv(`up.${u}.lift`, x, { up: true });
      x = this.concat(x, skips.pop());
      for (let b = 0; b < cfg.blocks; b++) x = this.resblock(`up.${u}.blocks.${b}`, x, channels[i]);
    }
    x = this.normSilu("norm", x);
    return this.conv("head", x);
  }

  conv(name, x, { stride = 1, up = false, residual = null } = {}) {
    let wt = this.tensor(`${name}.weight`);
    let bias = this.tensor(`${name}.bias`).data;
    const [, k, , cin] = wt.shape;
    if (cin !== x.c || cin % 4) throw new Error(`${name}: ${cin} inputs on ${x.c} channels`);
    // The shader writes 4 outputs per thread: pad to a multiple of 4 with zero filters. Only the head
    // needs it (sabbia has 93 classes); its extra logits are dropped in `logits`.
    const cout = Math.ceil(wt.shape[0] / 4) * 4;
    if (cout !== wt.shape[0]) {
      const data = new Float32Array(cout * k * k * cin);
      data.set(wt.data);
      wt = { shape: [cout, k, k, cin], data };
      const padded = new Float32Array(cout);
      padded.set(bias);
      bias = padded;
    }
    const pad = (k - 1) >> 1;
    const ih = x.h, iw = x.w;
    const eh = up ? ih * 2 : ih, ew = up ? iw * 2 : iw;
    const oh = Math.floor((eh + 2 * pad - k) / stride) + 1;
    const ow = Math.floor((ew + 2 * pad - k) / stride) + 1;
    const out = { buf: this.buffer(oh * ow * cout * 4), h: oh, w: ow, c: cout };
    this.step(
      "conv",
      [x.buf, this.buffer(0, 0, packConv(wt.data, wt.shape)), this.buffer(0, 0, bias),
        residual ? residual.buf : this.none, out.buf],
      [ih, iw, cin, oh, ow, cout, k, stride, pad, up ? 1 : 0, residual ? 1 : 0, 0],
      (oh * ow * cout) / 4,
    );
    return out;
  }

  normSilu(name, x) {
    const groups = Math.min(32, x.c / 4);
    const per = x.c / groups;
    const stats = this.buffer(groups * 8);
    const out = { ...x, buf: this.buffer(x.h * x.w * x.c * 4) };
    const params = [x.h * x.w, x.c, per, 0];
    this.step("stats", [x.buf, stats], params, 0, groups);
    this.step(
      "norm",
      [x.buf, stats, this.buffer(0, 0, this.tensor(`${name}.weight`).data), this.buffer(0, 0, this.tensor(`${name}.bias`).data), out.buf],
      params,
      x.h * x.w * x.c,
    );
    return out;
  }

  resblock(name, x, cout) {
    let h = this.conv(`${name}.conv1`, this.normSilu(`${name}.norm1`, x));
    const skip = this.meta.tensors[`${name}.skip.weight`] ? this.conv(`${name}.skip`, x) : x;
    return this.conv(`${name}.conv2`, this.normSilu(`${name}.norm2`, h), { residual: skip });
  }

  concat(a, b) {
    const out = { buf: this.buffer(a.h * a.w * (a.c + b.c) * 4), h: a.h, w: a.w, c: a.c + b.c };
    this.step("concat", [a.buf, b.buf, out.buf], [a.h * a.w, a.c, b.c, 0], a.h * a.w * out.c);
    return out;
  }

  // ---------- running it ----------

  /** Logits `[H * W, classes]` for the next frame. `past` is an array of `history` Uint8Arrays (oldest
   * first); `partial` holds class ids, with `classes` marking a cell still to be predicted. */
  async logits(past, partial) {
    const ids = new Uint32Array((this.config.history + 1) * this.cells);
    past.forEach((frame, j) => ids.set(frame, j * this.cells));
    ids.set(partial, this.config.history * this.cells);
    this.device.queue.writeBuffer(this.ids, 0, ids);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const s of this.steps) {
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.dispatchWorkgroups(s.workgroups);
    }
    pass.end();
    encoder.copyBufferToBuffer(this.logitsBuffer, 0, this.readback, 0, this.readback.size);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    const c = this.config.classes;
    if (this.stride === c) return raw;
    const out = new Float32Array(this.cells * c);
    for (let cell = 0; cell < this.cells; cell++) out.set(raw.subarray(cell * this.stride, cell * this.stride + c), cell * c);
    return out;
  }

  /** Dreams the next frame (a Uint8Array of class ids), revealed over `passes` model calls, the most
   * confident cells first. Mirrors `sample` in sogno/model.py. */
  async sample(past, { passes = 4, temperature = 1, random = Math.random } = {}) {
    const c = this.config.classes;
    const cells = this.cells;
    const partial = new Uint8Array(cells).fill(c);
    const confidence = new Float32Array(cells);
    const choice = new Uint8Array(cells);
    const weights = new Float64Array(c);
    const gumbel = () => -Math.log(-Math.log(Math.max(random(), 1e-9)));
    for (let i = 0; i < passes; i++) {
      const logits = await this.logits(past, partial);
      const noise = temperature * (1 - (i + 1) / passes);
      const masked = [];
      for (let cell = 0; cell < cells; cell++) {
        if (partial[cell] !== c) continue;
        const row = cell * c;
        let max = -Infinity;
        for (let k = 0; k < c; k++) max = Math.max(max, logits[row + k]);
        let sum = 0;
        for (let k = 0; k < c; k++) sum += Math.exp(logits[row + k] - max);
        let pick = 0;
        if (temperature > 0) {
          // One uniform draw against the cumulative softmax(logits / T).
          let total = 0;
          for (let k = 0; k < c; k++) total += (weights[k] = Math.exp((logits[row + k] - max) / temperature));
          let u = random() * total;
          pick = c - 1;
          for (let k = 0; k < c; k++) {
            u -= weights[k];
            if (u < 0) {
              pick = k;
              break;
            }
          }
        } else {
          for (let k = 1; k < c; k++) if (logits[row + k] > logits[row + pick]) pick = k;
        }
        choice[cell] = pick;
        confidence[cell] = logits[row + pick] - max - Math.log(sum) + (noise > 0 ? noise * gumbel() : 0);
        masked.push(cell);
      }
      const keep = Math.floor(Math.cos((((i + 1) / passes) * Math.PI) / 2) * cells);
      let threshold = -Infinity;
      if (keep > 0) {
        const sorted = Float32Array.from(masked, (cell) => confidence[cell]).sort();
        threshold = keep - 1 < sorted.length ? sorted[keep - 1] : Infinity;
      }
      for (const cell of masked) if (confidence[cell] > threshold) partial[cell] = choice[cell];
    }
    return partial;
  }
}

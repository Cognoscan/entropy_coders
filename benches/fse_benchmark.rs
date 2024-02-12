use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn gen_sequence(prob: f64, size: usize) -> Vec<u8> {
    const LUT_SIZE: usize = 4096;
    let mut lut = [0u8; 4096];
    let prob = prob.clamp(0.005, 0.995);
    let mut remaining = LUT_SIZE;
    let mut idx = 0;
    let mut s = 0u8;
    while remaining > 0 {
        let n = ((remaining as f64 * prob) as usize).max(1);
        for _ in 0..n {
            lut[idx] = s;
            idx += 1;
        }
        s += 1;
        remaining -= n;
    }
    let mut out = Vec::with_capacity(size);
    let mut rng = rand::thread_rng();
    for _ in 0..size {
        let i: u16 = rng.gen();
        out.push(lut[i as usize & (LUT_SIZE-1)]);
    }
    out
}

fn time_benchmark(c: &mut Criterion) {
    let len = 1<<15;
    let src = gen_sequence(0.2, len);
    let mut dst = Vec::with_capacity(src.len()*2);
    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Bytes(len as u64));
    group.bench_function("compress_20", |b| b.iter(|| {
        dst.clear();
        entropy_coders::fse_compress2(black_box(src.as_slice()), &mut dst)
    }));
    group.finish();
}

criterion_group!{
    name = time_bench;
    config = Criterion::default();
    targets = time_benchmark
}
criterion_main!(time_bench);
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use entropy_coders::{bitstream::BitStackWriter, *};
use rand::Rng;
use criterion_perf_events::Perf;
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;

fn fse_compress2(src: &[u8], dst: &mut Vec<u8>) -> usize {
    let mut writer = BitStackWriter::new(dst);
    let hist = histogram::NormHistogram::new(src);
    let fse_table = FseTable::new(&hist);
    let mut src_iter = src.chunks(2).rev();
    let first = src_iter.next().unwrap();
    let (mut encode0, mut encode1) = if first.len() == 1 {
        let next = src_iter.next().unwrap();
        let mut encode0 = FseEncode::new_first_symbol(&fse_table, first[0]);
        let encode1 = FseEncode::new_first_symbol(&fse_table, next[1]);
        encode0.encode(&mut writer, next[0]);
        (encode0, encode1)
    }
    else {
        let encode0 = FseEncode::new_first_symbol(&fse_table, first[0]);
        let encode1 = FseEncode::new_first_symbol(&fse_table, first[1]);
        (encode0, encode1)
    };

    for n in src_iter {
        encode0.encode(&mut writer, unsafe { *n.get_unchecked(0) });
        encode1.encode(&mut writer, unsafe { *n.get_unchecked(1) });
    }

    encode0.finish(&mut writer);
    encode1.finish(&mut writer);
    writer.finish()
}

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

fn cycles_benchmark(c: &mut Criterion<Perf>) {
    let src = gen_sequence(0.2, 1<<15);
    let mut dst = Vec::with_capacity(src.len());
    c.bench_function("compress_20", |b| b.iter(|| {
        dst.clear();
        entropy_coders::fse_compress(black_box(src.as_slice()), &mut dst)
    }));
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

//criterion_group!(benches, criterion_benchmark);
//criterion_main!(benches);


//criterion_group!{
//    name = cycles_bench;
//    config = Criterion::default()
//        .with_measurement(Perf::new(Builder::from_hardware_event(Hardware::CPUCycles)));
//    targets = cycles_benchmark
//}
criterion_group!{
    name = time_bench;
    config = Criterion::default();
    targets = time_benchmark
}
criterion_main!(time_bench);
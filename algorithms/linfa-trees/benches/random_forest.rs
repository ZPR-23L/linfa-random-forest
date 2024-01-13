use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::traits::Fit;
use linfa_datasets::generate::make_dataset;
use linfa_trees::RandomForestClassifier;
use statrs::distribution::{DiscreteUniform, Uniform};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_forest");
    config::set_default_benchmark_configs(&mut group);

    let params: [(usize, usize); 3] = [(100, 4), (1_000, 4), (10_000, 4)];

    let feat_distr = Uniform::new(0., 5.).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();

    let hyperparams = RandomForestClassifier::params();
    for (size, num_feat) in params {
        let dataset = make_dataset(size, num_feat, 1, feat_distr, target_distr);
        let dataset = dataset.into_single_target();
        let dataset = dataset.map_targets(|y| *y as usize);

        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, dataset| {
            b.iter(|| hyperparams.fit(&dataset));
        });
    }
    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);

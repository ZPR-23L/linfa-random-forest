use linfa::prelude::*;
use linfa_trees::{DecisionTree, RandomForestClassifier, Result, SplitQuality};
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

fn main() -> Result<()> {
    // Load dataset
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("Training model with default params...");
    let default_model = RandomForestClassifier::params().fit(&train)?;
    let default_predict = default_model.predict(&test);
    let conf_matrix = default_predict.confusion_matrix(&test)?;

    println!("{:?}", conf_matrix);

    println!(
        "Test accuracy with default params: {:.2}%",
        100.0 * conf_matrix.accuracy()
    );

    println!("Training model with custom tree params...");
    let trees_params = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0);
    let custom_model = RandomForestClassifier::params()
        .trees_params(trees_params)
        .fit(&train)?;
    let custom_predict = custom_model.predict(&test);
    let conf_matrix = custom_predict.confusion_matrix(&test)?;

    println!("{:?}", conf_matrix);

    println!(
        "Test accuracy with custom tree params: {:.2}%",
        100.0 * conf_matrix.accuracy()
    );

    println!("Training decision tree with the same params for comparison...");
    let tree_model = trees_params.fit(&train)?;
    let tree_predict = tree_model.predict(&test);
    let conf_matrix = tree_predict.confusion_matrix(&test)?;

    println!("{:?}", conf_matrix);

    println!(
        "Test accuracy with decision tree: {:.2}%",
        100.0 * conf_matrix.accuracy()
    );

    Ok(())
}

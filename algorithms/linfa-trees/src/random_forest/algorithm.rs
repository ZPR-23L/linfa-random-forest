//! Random forest
//!
use crate::{DecisionTree, MaxFeatures, RandomForestValidParams};
use linfa::prelude::Fit;
use linfa::dataset::{AsTargets, Records};
use linfa::traits::{Predict, PredictInplace};
use std::collections::HashMap;
use linfa::{
    dataset::{AsSingleTargets, Labels},
    error::Error,
    error::Result,
    DatasetBase, Float, Label,
};
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand::Rng;
use rayon::scope;
use std::sync::{Arc, Mutex};

/// A random forest model for classification
///
/// ### Structure
///
/// A random forest is an ensamble of decision trees. Each tree is fitted with some random noise,
/// which ensures variability between the trees. That in turn, makes it statistically a more
/// accurate model, in comparison to a single decision tree.
///
/// ### Algorithm
///
/// To create a tree, a subset of the original dataset is chosen. When bootstrapping is enabled, a number of
/// samples can be specified. The samples are then drawn with replacement from the original dataset. This means
/// that even when the sample size is equal to the size of the dataset, not all samples will be used - there will
/// be repeats of the same rows. With bootstrapping disabled, all samples will be used to fit every tree.
///
/// To ensure variability of trees, when creating each tree, at every split only a subset of features is chosen randomly.
///
/// The number of features and samples can be specified as [hyperparameters](crate::RandomForestParams).
///
/// ### Predictions
///
/// Prediction is made by picking the most common label from the predictions of all decision trees of the forest.
///
/// ### Example
///
/// Below is an example on how to train a random forest with default hyperparams:
///
/// ```rust
/// use linfa_trees::RandomForestClassifier;
/// use linfa::prelude::*;
/// use linfa_datasets;
///
/// // Load the dataset
/// let dataset = linfa_datasets::iris();
/// // Fit the random forest
/// let forest = RandomForestClassifier::params().fit(&dataset).unwrap();
/// // Get accuracy on training set
/// let accuracy = forest.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();
/// ```

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct RandomForestClassifier<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
    oob_score: Option<f64>,
}

impl<F: Float, L: Label> RandomForestClassifier<F, L> {

    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    fn calculate_oob_score<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>
    (dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, indices: &Vec<Vec<usize>>, trees: &Vec<DecisionTree<F, L>>)
    -> Option<f64> {
        // For every sample find all trees that didn't use it for training
        // At sample's index holds a vector of tree indices that didn't use the sample
        let mut trees_without_samples = Vec::default();
        for sample_idx in 0..dataset.nsamples() {
            let mut trees_without_sample_idx = Vec::default();
            for tree_idx in 0..indices.len() {
                if !indices[tree_idx].contains(&sample_idx) {
                    trees_without_sample_idx.push(tree_idx);
                }
            }
            trees_without_samples.push(trees_without_sample_idx);
        }
        
        let mut correct = 0;
        
        // For every sample, make a prediction with trees trained without this sample
        for sample_idx in 0..dataset.nsamples() {
            let sample = dataset.records().select(Axis(0), &[sample_idx]);
            let target = dataset.as_targets().select(Axis(0), &[sample_idx]);
            let mut predictions = Vec::default();

            for tree_idx in &trees_without_samples[sample_idx] {
                predictions.extend(trees[*tree_idx].predict(&sample));
            }

            let predictions = Array::from_vec(predictions);
            let pred = most_common(predictions);

            // If majority predicted correct label, add to correct predictions counter
            if &pred == target.get(0).unwrap() { correct += 1; }
        }

        Some(correct as f64 / dataset.nsamples() as f64)
    }

    fn bootstrap_indices(nsamples: usize, max_samples: usize) -> Vec<usize> {
        let mut rng = thread_rng();

        // Sample with replacement
        let indices = (0..nsamples)
            .map(|_| rng.gen_range(0..nsamples))
            .take(max_samples)
            .collect::<Vec<_>>();

        indices
    }

    fn bootstrap<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        indices: &Vec<Vec<usize>>,
    ) -> Vec<DatasetBase<Array<F, Ix2>, Array<L, Ix1>>> {
        let mut bootstrapped_samples = Vec::new();

        for i in 0..indices.len() {
            let records = dataset.records().select(Axis(0), &indices[i]);
            let targets = dataset.as_targets().select(Axis(0), &indices[i]);

            // Create a bootstrapped dataset
            let bootstrapped_dataset = DatasetBase::new(records, targets)
                .with_feature_names(dataset.feature_names());
            bootstrapped_samples.push(bootstrapped_dataset);
        }

        bootstrapped_samples
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug + Sync + Send, D, T>
    Fit<ArrayBase<D, Ix2>, T, Error> for RandomForestValidParams<F, L>
where
    D: Data<Elem = F> + Send + Sync,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L> + Send + Sync,
{
    type Object = RandomForestClassifier<F, L>;

    /// Using specified hyperparameters, fit a random forest on a dataset with a matrix of features and an array of labels
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

        let num_features = dataset.feature_names().len();

        let max_features = match self.max_features() {
            MaxFeatures::Sqrt => f64::sqrt(num_features as f64) as usize,
            MaxFeatures::Log2 => f64::log2(num_features as f64) as usize,
            MaxFeatures::Float(n) => std::cmp::max(1, ((num_features as f32) * n) as usize),
            MaxFeatures::None => num_features,
        };

        let num_samples = dataset.records().len();

        let max_samples = match self.max_samples() {
            Some(n) => std::cmp::max(1, ((num_samples as f32) * n) as usize),
            None => num_samples,
        };

        let indices: Vec<Vec<usize>> = (0..self.num_trees())
            .map(|_| Self::Object::bootstrap_indices(dataset.nsamples(), max_samples))
            .collect();

        let samples = if self.bootstrap() {
            Self::Object::bootstrap(&dataset, &indices)} else {
                vec![DatasetBase::new(
                        dataset.records().to_owned(),
                        dataset.as_targets().to_owned()
                    ).with_feature_names(dataset.feature_names());
                    self.num_trees()]
            };
        
        let fitted_trees = Mutex::new(vec![None; self.num_trees()]);

        // Using concurrency for fitting trees
        scope(|s| {
            for (i, sample) in samples.iter().enumerate() {
                let fitted_trees_ref = &fitted_trees; // Borrow a reference to the Mutex
                s.spawn(move |_| {
                    let tree = self.trees_params()
                        .max_features(Some(max_features))
                        .fit(sample)
                        .unwrap();
        
                    // Lock the Mutex and insert the tree at the correct index
                    fitted_trees_ref.lock().unwrap()[i] = Some(tree);
                });
            }
        });

        let fitted_trees = fitted_trees.into_inner()
            .unwrap()
            .iter()
            .map(|x| x.clone().unwrap())
            .collect();

        let oob_score = if self.oob_score() { Self::Object::calculate_oob_score(&dataset, &indices, &fitted_trees) }
            else { None };

        Ok(RandomForestClassifier {
            trees: fitted_trees,
            oob_score: oob_score,
        })
    }
}

impl<F: Float, L: Label + Default + Copy + Send + Sync, D: Data<Elem = F> + Send + Sync>
    PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for RandomForestClassifier<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        // 2D array holds predictions for each row of `x` from every tree
        let trees_targets = Arc::new(Mutex::new(Array2::<L>::default((0, x.nrows()))));

        // Each tree makes a prediction concurrently
        scope(|s| {
            for tree in &self.trees {
                let trees_targets = Arc::clone(&trees_targets);
                s.spawn(move |_| {
                    let targets = tree.predict(x);
                    let mut trees_targets = trees_targets.lock().unwrap();
                    trees_targets.push_row(targets.view()).unwrap();
                });
            }
        });

        let trees_targets = trees_targets.lock().unwrap();

        // Search for most frequent label in each column
        for (idx, target) in y.iter_mut().enumerate() {
            *target = most_common::<L>(trees_targets.column(idx).to_owned()).clone();
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

fn most_common<L: std::hash::Hash + Eq>(targets: Array1<L>) -> L {
    let mut map = HashMap::new();
    for target in targets {
        let counter = map.entry(target).or_insert(0);
        *counter += 1;
    }
    let (most_common, _) = map.into_iter().max_by_key(|(_, v)| *v).unwrap();
    most_common
}


#[cfg(test)]
mod tests {
    use crate::{
        DecisionTree, MaxFeatures, RandomForestClassifier, RandomForestParams,
        RandomForestValidParams, SplitQuality,
    };
    use linfa::dataset::Records;
    use linfa::traits::{Fit, PredictInplace};
    use linfa::Dataset;
    use linfa::ParamGuard;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand::prelude::SmallRng;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<RandomForestClassifier<f64, bool>>();
        has_autotraits::<RandomForestValidParams<f64, bool>>();
        has_autotraits::<RandomForestParams<f64, bool>>();
    }

    #[test]
    fn default_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let valid_params = params.check().unwrap();
        assert_eq!(valid_params.num_trees(), 100);
        assert_eq!(valid_params.bootstrap(), true);
        assert_eq!(valid_params.oob_score(), false);
        assert_eq!(valid_params.max_samples(), None);
        assert_eq!(valid_params.max_features(), MaxFeatures::Sqrt);
    }

    #[test]
    fn custom_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let valid_params = params
            .num_trees(50)
            .oob_score(true)
            .max_samples(Some(0.5))
            .max_features(MaxFeatures::None)
            .check()
            .unwrap();
        assert_eq!(valid_params.num_trees(), 50);
        assert_eq!(valid_params.bootstrap(), true);
        assert_eq!(valid_params.oob_score(), true);
        assert_eq!(valid_params.max_samples(), Some(0.5));
        assert_eq!(valid_params.max_features(), MaxFeatures::None);
    }

    #[test]
    fn custom_tree_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let tree_params = DecisionTree::params().split_quality(SplitQuality::Entropy);
        let valid_params = params.trees_params(tree_params).check().unwrap();
        assert_eq!(
            valid_params.trees_params().check().unwrap().split_quality(),
            SplitQuality::Entropy
        );
    }

    #[test]
    fn custom_invalid_tree_params() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let tree_params = DecisionTree::params().min_impurity_decrease(0.);
        let params = params.trees_params(tree_params);
        let result = params.check();
        assert!(result.is_err());
    }

    #[test]
    fn invalid_max_samples() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_samples(Some(1.5));
        let result = params.check();
        assert!(result.is_err());
    }

    #[test]
    fn invalid_max_features() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_features(MaxFeatures::Float(1.5));
        let result = params.check();
        assert!(result.is_err());
    }

    #[test]
    fn oob_without_bootstrap_error() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.bootstrap(false).oob_score(true);
        let result = params.check();
        assert!(result.is_err());
    }

    #[test]
    fn max_samples_without_bootstrap_error() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.bootstrap(false).max_samples(Some(0.5));
        let result = params.check();
        assert!(result.is_err());
    }

    #[test]
    fn bootstrap_test() {
        let data = array![
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, -14.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 5.0, 3.0, 0.0, -4.0, 0.0, 0.0, 1.0, -5.0, 0.2, 0.0, 4.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, -4.5, 0.0, 0.0, 2.1, 1.0, 0.0, 0.0, -4.5, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
            [-1.0, -2.0, 0.0, 4.0, -3.0, 10.0, 4.0, 0.0, -3.2, 0.0, 4.0, 3.0, -4.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, 0.0,],
            [2.0, 8.0, 5.0, 1.0, 0.5, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 2.0, 0.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -2.0, 3.0, 0.0, 1.0, 0.0,],
            [2.0, 0.0, 1.0, 2.0, 3.0, -1.0, 10.0, 2.0, 0.0, -1.0, 1.0, 2.0, 2.0, 0.0,],
            [1.0, 1.0, 0.0, 2.0, 2.0, -1.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 3.0, 0.0,],
            [3.0, 1.0, 0.0, 3.0, 0.0, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 1.5, 1.0, -1.0, -1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 10.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, -1.0,],
            [2.0, 0.0, 5.0, 1.0, 0.5, -2.0, 10.0, 0.0, 1.0, -5.0, 3.0, 1.0, 0.0, -1.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -2.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0,],
            [2.0, 1.0, 1.0, 1.0, 2.0, -1.0, 10.0, 2.0, 0.0, -1.0, 0.0, 2.0, 1.0, 1.0,],
            [1.0, 1.0, 0.0, 0.0, 1.0, -3.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 1.0, 1.0,],
            [3.0, 1.0, 0.0, 1.0, 0.0, -4.0, 1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0, 0.0,]
        ];

        let targets = array![1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0];

        let dataset = Dataset::new(data, targets);
        let indices = vec![RandomForestClassifier::<f64, usize>::bootstrap_indices(dataset.nsamples(), 10); 10];
        let bootstrapped = RandomForestClassifier::bootstrap(&dataset, &indices);
        assert_eq!(bootstrapped.len(), 10);
        assert!(bootstrapped.iter().all(|x| x.nsamples() == 10));
    }

    #[test]
    fn fit_test() {
        let mut rng = SmallRng::seed_from_u64(42);

        let (train, _) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);
        let classifier_model = RandomForestClassifier::params().fit(&train);
        assert!(classifier_model.is_ok());
        let classifier_model = classifier_model.unwrap();
        assert_eq!(classifier_model.trees.len(), 100);
    }

    #[test]
    fn most_common_test() {
        let test_array = array![1, 2, 1, 1, 4, 1, 2, 2, 2, 1];
        let most_common_val = super::most_common(test_array);
        assert_eq!(most_common_val, 1);
    }

    #[test]
    fn predict_inplace_test() {
        let mut rng = SmallRng::seed_from_u64(42);

        let (train, _) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);
        let classifier_model = RandomForestClassifier::params().fit(&train).unwrap();
        let mut targets = classifier_model.default_target(&train.records);
        let original_targets = targets.to_owned();
        classifier_model.predict_inplace(&train.records, &mut targets);
        assert_ne!(targets, original_targets);
    }
}

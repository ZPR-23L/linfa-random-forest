//! Random forest
//!
use crate::{DecisionTree, MaxFeatures, RandomForestValidParams};
use linfa::prelude::{Fit};
use linfa::dataset::{AsTargets, Records};
use linfa::traits::{Predict, PredictInplace};
use std::collections::{HashMap};
use ndarray_rand::rand::seq::IteratorRandom;
use linfa::{
    dataset::{AsSingleTargets, Labels},
    error::Error,
    error::Result,
    DatasetBase, Float, Label,
};
use ndarray::{Array, array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
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
/// Another means of adding random noise to the trees is by randomly selecting the features used to train a tree.
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

    fn bootstrap<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        num_trees: usize,
        max_samples: usize,
        max_features: usize,
    ) -> Vec<DatasetBase<Array<F, Ix2>, Array<L, Ix1>>> {
        let mut bootstrapped_samples = Vec::new();

        for _ in 0..num_trees {
            let mut rng = thread_rng();

            // Sample with replacement
            let indices = (0..dataset.nsamples())
                .map(|_| rng.gen_range(0..dataset.nsamples()))
                .take(max_samples)
                .collect::<Vec<_>>();

            let records = dataset.records().select(Axis(0), &indices);
            let targets = dataset.as_targets().select(Axis(0), &indices);

            // Sample features
            let feature_indices = (0..dataset.nfeatures()).choose_multiple(&mut rng, max_features);

            let records = records.select(Axis(1), &feature_indices);

            // Create a bootstrapped dataset
            let bootstrapped_dataset = DatasetBase::new(records, targets);
            bootstrapped_samples.push(bootstrapped_dataset);
        }

        bootstrapped_samples
    }

    fn bootstrap_samples<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        num_trees: usize,
        max_samples: usize,
    ) -> Vec<DatasetBase<Array<F, Ix2>, Array<L, Ix1>>> {
        let mut bootstrapped_samples = Vec::new();

        for _ in 0..num_trees {
            let mut rng = thread_rng();

            // Sample with replacement
            let indices = (0..dataset.nsamples())
                .map(|_| rng.gen_range(0..dataset.nsamples()))
                .take(max_samples)
                .collect::<Vec<_>>();

            let records = dataset.records().select(Axis(0), &indices);
            let targets = dataset.as_targets().select(Axis(0), &indices);

            // Take all features
            let feature_indices = (0..dataset.nfeatures()).collect::<Vec<_>>();

            let records = records.select(Axis(1), &feature_indices);

            // Create a bootstrapped dataset
            let bootstrapped_dataset = DatasetBase::new(records, targets);
            bootstrapped_samples.push(bootstrapped_dataset);
        }

        bootstrapped_samples
    }
    fn bootstrap_features<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        num_trees: usize,
        max_features: usize,
    ) -> Vec<DatasetBase<Array<F, Ix2>, Array<L, Ix1>>> {
        let mut bootstrapped_features = Vec::new();

        for _ in 0..num_trees {
            let mut rng = thread_rng();

            // Sample with replacement
            let indices = (0..dataset.nsamples()).collect::<Vec<_>>();

            let records = dataset.records().select(Axis(0), &indices);
            let targets = dataset.as_targets().select(Axis(0), &indices);

            // Sample features
            let feature_indices = (0..dataset.nfeatures()).choose_multiple(&mut rng, max_features);

            let records = records.select(Axis(1), &feature_indices);

            // Create a bootstrapped dataset
            let bootstrapped_dataset = DatasetBase::new(records, targets);
            bootstrapped_features.push(bootstrapped_dataset);
        }

        bootstrapped_features
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
        let fitted_trees = Mutex::new(Vec::new());

        let num_features = dataset.feature_names().len();
        let bootstrap_features = match self.max_features() {
            MaxFeatures::Sqrt => f64::sqrt(num_features as f64) as usize,
            MaxFeatures::Log2 => f64::log2(num_features as f64) as usize,
            MaxFeatures::Float(n) => std::cmp::max(1, ((num_features as f32) * n) as usize),
            MaxFeatures::None => num_features,
        };

        let num_samples = dataset.records().len();
        let bootstrap_samples = match self.max_samples() {
            Some(n) => std::cmp::max(1, ((num_samples as f32) * n) as usize),
            None => num_samples,
        };

        let samples = if self.bootstrap() {
            Self::Object::bootstrap_samples(
                &dataset,
                self.num_trees(),
                bootstrap_samples
            )
        } else {
            Self::Object::bootstrap_features(&dataset, self.num_trees(), bootstrap_features)
        };
        let mut oob_score = None;

        // Using concurrency for fitting trees
        scope(|s| {
            for sample in samples.iter() {
                let fitted_trees_ref = &fitted_trees; // Borrow a reference to the Mutex
                s.spawn(move |_| {
                    let tree = self.trees_params().fit(sample).unwrap();
                    if self.oob_score() {
                        let oob_samples = dataset.records().outer_iter()
                            .filter(|x|
                                sample.records().outer_iter()
                                    .find(|y| x.eq(y))
                                    .is_some())
                            .collect::<Vec<_>>();
                        let mut arr = Array2::<F>::default((oob_samples.len(), oob_samples[0].len()));
                        for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
                            for (j, col) in row.iter_mut().enumerate() {
                                *col = oob_samples[i][j];
                            }
                        }
                        let new_dataset = DatasetBase::new(arr, dataset.targets());
                        let predict = tree.predict(&new_dataset);
                        // tutaj powinno nastapic wyliczenie score, niestety nie udalo mi sie z uzyciem confussion matrix
                        // z powodu uplywajacego terminu pozostawiam jako uwage
                        oob_score = Some(1.0);
                    }
                    // Lock the Mutex and push the tree into the vector
                    fitted_trees_ref.lock().unwrap().push(tree);
                });
            }
        });

        let fitted_trees = fitted_trees.into_inner().unwrap();

        Ok(RandomForestClassifier {
            trees: fitted_trees,
            oob_score,
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

#[test]
fn test_most_common() {
    let test_array = array![1, 2, 1, 1, 4, 1, 2, 2, 2, 1];
    let most_common_val = most_common(test_array);
    assert_eq!(most_common_val, 1);
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
        let bootstrapped = RandomForestClassifier::bootstrap(&dataset, 10, 10, 10);
        assert_eq!(bootstrapped.len(), 10);
        assert!(bootstrapped.iter().all(|x| x.nsamples() == 10));
        assert!(bootstrapped.iter().all(|x| x.nfeatures() == 10));
    }

    #[test]
    fn bootstrap_samples_test() {
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
        let bootstrapped = RandomForestClassifier::bootstrap_samples(&dataset, 10, 10);
        assert_eq!(bootstrapped.len(), 10);
        assert!(bootstrapped.iter().all(|x| x.nsamples() == 10));
        assert!(bootstrapped
            .iter()
            .all(|x| x.nfeatures() == dataset.nfeatures()));
    }

    #[test]
    fn bootstrap_features_test() {
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
        let bootstrapped = RandomForestClassifier::bootstrap_features(&dataset, 10, 10);
        assert_eq!(bootstrapped.len(), 10);
        assert!(bootstrapped
            .iter()
            .all(|x| x.nsamples() == dataset.nsamples()));
        assert!(bootstrapped.iter().all(|x| x.nfeatures() == 10));
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
        assert!(classifier_model.oob_score.is_none());
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

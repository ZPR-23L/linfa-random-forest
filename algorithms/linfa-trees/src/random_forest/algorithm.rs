//! Random forest
//! 
use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use linfa::{
    dataset::{Labels, AsSingleTargets},
    error::Error,
    error::Result,
    DatasetBase, Float, Label,
};
use linfa::prelude::Fit;
use linfa::traits::{PredictInplace, Predict};
use crate::{DecisionTree, RandomForestValidParams, MaxFeatures};


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

pub struct RandomForestClassifier<F: Float, L: Label> {
    trees: Vec<DecisionTree<F, L>>, // collection of fitted decision trees of the forest
    oob_score: Option<f32>
}

impl<F: Float, L: Label> RandomForestClassifier<F, L> {
    fn calculate_oob_score() -> Option<f32> {
        // TODO implement
        // TODO correct function signature
        None
    }

    fn bootstrap<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, num_trees: usize,
                             max_samples: usize, max_features: usize) -> Vec<&DatasetBase<ArrayBase<D, Ix2>, T>> {
        // TODO implement
        Vec::default()
    }
    fn bootstrap_features<D: Data<Elem = F>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>>(dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, num_trees: usize,
        max_features: usize) -> Vec<&DatasetBase<ArrayBase<D, Ix2>, T>> {
        // TODO implement
        Vec::default()
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
for RandomForestValidParams<F, L>
    where
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = RandomForestClassifier<F, L>;

    /// Using specified hyperparameters, fit a random forest on a dataset with a matrix of features and an array of labels
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

        let mut fitted_trees: Vec<DecisionTree<F, L>> = Vec::new();

        let num_features = dataset.feature_names().len();
        let bootstrap_features = match self.max_features() {
            MaxFeatures::Sqrt => f64::sqrt(num_features as f64) as usize,
            MaxFeatures::Log2 => f64::log2(num_features as f64) as usize,
            MaxFeatures::Float(n) => std::cmp::max(1, ((num_features as f32) * n) as usize),
            MaxFeatures::None => num_features
        };

        let num_samples = dataset.records().len();
        let bootstrap_samples = match self.max_samples() {
            Some(n) => std::cmp::max(1, ((num_samples as f32) * n) as usize), 
            None => num_samples
        };

        let samples = if self.bootstrap()
            {Self::Object::bootstrap(&dataset, self.num_trees(), bootstrap_samples, bootstrap_features)}
            else {Self::Object::bootstrap_features(&dataset, self.num_trees(), bootstrap_features)};

        for sample in samples {
            let tree = self.trees_params().fit(&sample)?;
            fitted_trees.push(tree);
        }

        let oob_score = if self.oob_score() {Self::Object::calculate_oob_score()} else {None};

        Ok(RandomForestClassifier{
            trees: fitted_trees,
            oob_score
        })
    }

}

impl<F: Float, L: Label + Default + Copy, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
for RandomForestClassifier<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        // 2D array holds predictions for each row of `x` from every tree
        let mut trees_targets = Array2::<L>::default((0, x.nrows()));

        for tree in &self.trees {
            let targets = tree.predict(x);
            trees_targets.push_row(targets.view()).unwrap();
        }

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
    use linfa::ParamGuard;
    use crate::{RandomForestClassifier, RandomForestValidParams, RandomForestParams, DecisionTree, MaxFeatures, SplitQuality};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin> () {}
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
        let valid_params = params.num_trees(50)
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
        assert_eq!(valid_params.trees_params().split_quality(), SplitQuality::Entropy);
    }

    #[test]
    fn invalid_max_samples() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_samples(Some(1.5));
        let result = params.check_ref();
        assert!(result.is_err());
    }

    #[test]
    fn invalid_max_features() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.max_features(MaxFeatures::Float(1.5));
        let result = params.check_ref();
        assert!(result.is_err());
    }

    #[test]
    fn oob_without_bootstrap_error() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.bootstrap(false).oob_score(true);
        let result = params.check_ref();
        assert!(result.is_err());
    }

    #[test]
    fn max_samples_without_bootstrap_error() {
        let params = RandomForestClassifier::<f64, bool>::params();
        let params = params.bootstrap(false).max_samples(Some(0.5));
        let result = params.check_ref();
        assert!(result.is_err());
    }
}
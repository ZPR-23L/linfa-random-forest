use crate::{DecisionTree, DecisionTreeParams, RandomForestClassifier};
use linfa::{
    error::{Error, Result},
    Float, Label, ParamGuard,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]

/// Specifies the amount of features which will be used to build each tree
/// of the random forest
pub enum MaxFeatures {
    /// The number of features used is the square root of the number of all features
    Sqrt,
    /// The number of features used is base 2 logarithm of the number of all features
    Log2,
    /// The number of features used is the result of multiplying this number
    /// and the number of all features. This number has to be in the range (0, 1)
    Float(f32),
    /// This will mean the number of features is equal to the number of all features
    None,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
/// The set of hyperparameters that can be specified for fitting a [random forest](RandomForestClassifier).
/// You can also change all the hyperparameters that are specific for the [decision tree](DecisionTreeParams).
///
/// ### Example
///
/// ```rust
/// use linfa_trees::{RandomForestClassifier, MaxFeatures, DecisionTree};
/// use linfa_datasets::iris;
/// use linfa::prelude::*;
///
/// // Initialize the default set of parameters
/// let params = RandomForestClassifier::params();
/// // Set the parameters to the desired values
/// let params = params.num_trees(150).max_samples(Some(0.8)).max_features(MaxFeatures::Sqrt);
/// // You can also change the parameters of decision trees
/// let tree_params = DecisionTree::params();
/// let tree_params = tree_params.max_depth(Some(5)).min_weight_leaf(2.);
/// let params = params.trees_params(tree_params);
/// ```
pub struct RandomForestValidParams<F, L> {
    trees_params: DecisionTreeParams<F, L>,
    num_trees: usize,          // number of estimators
    bootstrap: bool,           // is bootstrapping enabled
    oob_score: bool,           // is oob score enabled
    max_samples: Option<f32>,  // number of samples to bootstrap
    max_features: MaxFeatures, // number of features
}

impl<F: Float, L: Label> RandomForestValidParams<F, L> {
    /// Returns [DecisionTreeParams]. These are the hyperparameters used for all the
    /// [decision trees](DecisionTree) of the [random forest](RandomForestClassifier).
    pub fn trees_params(&self) -> DecisionTreeParams<F, L> {
        self.trees_params.clone()
    }

    /// Returns the number of trees in the [random forest](RandomForestClassifier).
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Returns a boolean - whether bootstrapping is used in the forest or not. If set to false,
    /// all samples in the dataset are used to create a single tree.
    pub fn bootstrap(&self) -> bool {
        self.bootstrap
    }

    /// Returns a boolean - whether OOB score is used in the forest or not.
    pub fn oob_score(&self) -> bool {
        self.oob_score
    }

    /// Returns a float in range (0, 1) or `None`. The result of multiplying this float by
    /// the number of all samples in the dataset is the number of samples used to create a single
    /// [decision tree](DecisionTree) of the [random forest](RandomForestClassifier). If it is `None`,
    /// then the number of samples is the number of all samples in the dataset.
    pub fn max_samples(&self) -> Option<f32> {
        self.max_samples
    }

    /// Returns enum [MaxFeatures]
    pub fn max_features(&self) -> MaxFeatures {
        self.max_features
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomForestParams<F, L>(RandomForestValidParams<F, L>);

impl<F: Float, L: Label> RandomForestParams<F, L> {
    pub fn new() -> Self {
        Self(RandomForestValidParams {
            trees_params: DecisionTree::params(),
            num_trees: 100,
            bootstrap: true,
            oob_score: false,
            max_samples: None,
            max_features: MaxFeatures::Sqrt,
        })
    }

    /// Sets the parameters of trees in the random forest.
    pub fn trees_params(mut self, trees_params: DecisionTreeParams<F, L>) -> Self {
        self.0.trees_params = trees_params;
        self
    }

    /// Sets the number of trees in the [random forest](RandomForestClassifier).
    pub fn num_trees(mut self, num_trees: usize) -> Self {
        self.0.num_trees = num_trees;
        self
    }

    /// Sets a boolean - whether bootstrapping is used in the forest or not. Setting bootstrapping to `false`
    /// will mean that each tree of the forest is fitted using all samples from the original dataset.
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.0.bootstrap = bootstrap;
        self
    }

    /// Sets a boolean - whether Out-of-Bag score is returned or not.
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.0.oob_score = oob_score;
        self
    }

    /// This parameter can only be used when `bootstrap = true`.
    /// Sets the number of samples used to construct each tree of the random forest.
    /// `max_samples` should be a float in range (0, 1) or `None`.
    /// The result of multiplying this float by the number of all samples in
    /// the dataset is the number of samples used to create a single
    /// [decision tree](DecisionTree) of the [random forest](RandomForestClassifier). If it is `None`,
    /// then the number of samples is the number of all samples in the dataset.
    pub fn max_samples(mut self, max_samples: Option<f32>) -> Self {
        self.0.max_samples = max_samples;
        self
    }

    /// Sets the number of features used to construct each tree of the random forest.
    /// `max_features` should be an enum [MaxFeatures].
    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.0.max_features = max_features;
        self
    }
}

impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L: Label> RandomForestClassifier<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `num_trees = 100`
    /// * `bootstrap = true`
    /// * `oob_score = false`
    /// * `max_samples = None`
    /// * `max_features = MaxFeatures::Sqrt`
    ///
    /// ([Decision tree](DecisionTree) default parameters)
    ///
    /// * `split_quality = SplitQuality::Gini`
    /// * `max_depth = None`
    /// * `min_weight_split = 2.0`
    /// * `min_weight_leaf = 1.0`
    /// * `min_impurity_decrease = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    pub fn params() -> RandomForestParams<F, L> {
        RandomForestParams::new()
    }
}

impl<F: Float, L> ParamGuard for RandomForestParams<F, L> {
    type Checked = RandomForestValidParams<F, L>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        self.0.trees_params.check_ref()?;
        if let MaxFeatures::Float(value) = self.0.max_features {
            if value <= 0.0 || value >= 1.0 {
                return Err(Error::Parameters(format!(
                    "Max features should be in range (0, 1), but was {}",
                    value
                )));
            }
        }
        if let Some(value) = self.0.max_samples {
            if value <= 0.0 || value >= 1.0 {
                return Err(Error::Parameters(format!(
                    "Max samples should be in range (0, 1), but was {}",
                    value
                )));
            }
        }
        if !self.0.bootstrap && self.0.oob_score {
            return Err(Error::Parameters(format!(
                "Cannot have oob_score without bootstrap"
            )));
        }
        if !self.0.bootstrap && self.0.max_samples != None {
            return Err(Error::Parameters(format!(
                "Cannot set max_samples without bootstrap"
            )));
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
